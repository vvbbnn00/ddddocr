/// Limits applied before decoding untrusted images.
pub const MAX_IMAGE_BYTES: usize = 8 * 1024 * 1024;
pub const MAX_IMAGE_SIDE: u32 = 4096;

fn load_image(data: &[u8]) -> anyhow::Result<image::DynamicImage> {
    anyhow::ensure!(
        !data.is_empty() && data.len() <= MAX_IMAGE_BYTES,
        "image must contain 1..=8 MiB"
    );
    let mut reader = image::ImageReader::new(std::io::Cursor::new(data)).with_guessed_format()?;
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(MAX_IMAGE_SIDE);
    limits.max_image_height = Some(MAX_IMAGE_SIDE);
    limits.max_alloc = Some(128 * 1024 * 1024);
    reader.limits(limits);
    Ok(reader.decode()?)
}

fn check_match_work(
    target: &image::GrayImage,
    background: &image::GrayImage,
) -> anyhow::Result<()> {
    let positions = u64::from(background.width() - target.width() + 1)
        * u64::from(background.height() - target.height() + 1);
    anyhow::ensure!(
        positions * u64::from(target.width()) * u64::from(target.height()) <= 100_000_000,
        "template matching exceeds the 100 million pixel-comparison limit"
    );
    Ok(())
}

fn decode_ctc(indices: impl IntoIterator<Item = usize>, charset: &[String]) -> String {
    let mut previous = None;
    let mut text = String::new();
    for index in indices {
        if previous != Some(index) && index != 0 {
            if let Some(character) = charset.get(index) {
                text.push_str(character);
            }
        }
        previous = Some(index);
    }
    text
}

fn softmax(row: &[f32]) -> Vec<f32> {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut probabilities: Vec<f32> = row.iter().map(|value| (value - max).exp()).collect();
    let sum: f64 = probabilities.iter().map(|&value| f64::from(value)).sum();
    for value in &mut probabilities {
        *value = (f64::from(*value) / sum) as f32;
    }
    probabilities
}

fn cpu_session(model: &[u8], threads: usize) -> anyhow::Result<ort::Session> {
    anyhow::ensure!(threads > 0, "inference threads must be positive");
    Ok(ort::Session::builder()?
        .with_intra_threads(threads)?
        .with_inter_threads(1)?
        .with_parallel_execution(false)?
        .commit_from_memory(model)?)
}

/// 初始化内容识别。
#[cfg(feature = "inline-model")]
pub fn ddddocr_classification() -> anyhow::Result<Ddddocr<'static>> {
    Ddddocr::new(
        include_bytes!("../model/common.onnx"),
        serde_json::from_str(include_str!("../model/common.json")).unwrap(),
    )
}

/// 初始化内容识别。
#[cfg(not(feature = "inline-model"))]
pub fn ddddocr_classification() -> anyhow::Result<Ddddocr<'static>> {
    Ddddocr::new(
        std::fs::read("model/common.onnx")?,
        serde_json::from_str(&std::fs::read_to_string("model/common.json")?)?,
    )
}

/// 初始化内容识别。
#[cfg(feature = "cuda")]
pub fn ddddocr_classification_cuda(device_id: i32) -> anyhow::Result<Ddddocr<'static>> {
    Ddddocr::new_cuda(
        include_bytes!("../model/common.onnx"),
        serde_json::from_str(include_str!("../model/common.json")).unwrap(),
        device_id,
    )
}

/// 使用旧模型初始化内容识别。
#[cfg(feature = "inline-model")]
pub fn ddddocr_classification_old() -> anyhow::Result<Ddddocr<'static>> {
    Ddddocr::new(
        include_bytes!("../model/common_old.onnx"),
        serde_json::from_str(include_str!("../model/common_old.json")).unwrap(),
    )
}

/// 使用旧模型初始化内容识别。
#[cfg(not(feature = "inline-model"))]
pub fn ddddocr_classification_old() -> anyhow::Result<Ddddocr<'static>> {
    Ddddocr::new(
        std::fs::read("model/common_old.onnx")?,
        serde_json::from_str(&std::fs::read_to_string("model/common_old.json")?)?,
    )
}

/// 使用旧模型初始化内容识别。
#[cfg(feature = "cuda")]
pub fn ddddocr_classification_old_cuda(device_id: i32) -> anyhow::Result<Ddddocr<'static>> {
    Ddddocr::new_cuda(
        include_bytes!("../model/common_old.onnx"),
        serde_json::from_str(include_str!("../model/common_old.json")).unwrap(),
        device_id,
    )
}

/// 初始化目标检测。
#[cfg(feature = "inline-model")]
pub fn ddddocr_detection() -> anyhow::Result<Ddddocr<'static>> {
    Ddddocr::new_model(include_bytes!("../model/common_det.onnx"))
}

/// 初始化目标检测。
#[cfg(not(feature = "inline-model"))]
pub fn ddddocr_detection() -> anyhow::Result<Ddddocr<'static>> {
    Ddddocr::new_model(std::fs::read("model/common_det.onnx")?)
}

/// 初始化目标检测。
#[cfg(feature = "cuda")]
pub fn ddddocr_detection_cuda(device_id: i32) -> anyhow::Result<Ddddocr<'static>> {
    Ddddocr::new_model_cuda(include_bytes!("../model/common_det.onnx"), device_id)
}

/// 滑块匹配。
pub fn slide_match<I1, I2>(target_image: I1, background_image: I2) -> anyhow::Result<SlideBBox>
where
    I1: AsRef<[u8]>,
    I2: AsRef<[u8]>,
{
    let target_image = load_image(target_image.as_ref())?;
    let background_image = load_image(background_image.as_ref())?;

    anyhow::ensure!(
        background_image.width() >= target_image.width(),
        "背景图片的宽度必须大于等于目标图片的宽度"
    );

    anyhow::ensure!(
        background_image.height() >= target_image.height(),
        "背景图片的高度必须大于等于目标图片的高度"
    );

    let target_image = target_image.to_rgba8();

    // 裁剪图片，只保留不透明部分
    let width = target_image.width();
    let height = target_image.height();
    let mut start_x = width;
    let mut start_y = height;
    let mut end_x = 0;
    let mut end_y = 0;

    for x in 0..width {
        for y in 0..height {
            let p = target_image[(x, y)];

            if p[3] != 0 {
                if x < start_x {
                    start_x = x;
                }

                if y < start_y {
                    start_y = y;
                }

                if x > end_x {
                    end_x = x;
                }

                if y > end_y {
                    end_y = y;
                }
            }
        }
    }

    anyhow::ensure!(
        start_x <= end_x && start_y <= end_y,
        "target image is fully transparent"
    );
    let cropped_image = image::imageops::crop_imm(
        &target_image,
        start_x,
        start_y,
        end_x - start_x + 1,
        end_y - start_y + 1,
    )
    .to_image();

    // 图片转换到灰度图
    let target_image = image::imageops::grayscale(&cropped_image);

    // 使用 canny 进行边缘检测。然后对背景图片进行同样的处理
    // 接着，使用 match_template 函数进行模板匹配，得到匹配结果矩阵
    // 然后使用 find_extremes 函数找到结果矩阵中的最大值和最小值
    // 并得到最大值所在的位置 loc，根据目标图片的大小和 loc 计算出目标物体的位置信息
    let background_image = background_image.to_luma8();
    check_match_work(&target_image, &background_image)?;
    let target_image = imageproc::edges::canny(&target_image, 100.0, 200.0);
    let background_image = imageproc::edges::canny(&background_image, 100.0, 200.0);
    let result =
        imageproc::template_matching::find_extremes(&imageproc::template_matching::match_template(
            &background_image,
            &target_image,
            imageproc::template_matching::MatchTemplateMethod::CrossCorrelationNormalized,
        ));

    Ok(SlideBBox {
        target_x: start_x,
        target_y: start_y,
        x1: result.max_value_location.0,
        y1: result.max_value_location.1,
        x2: result.max_value_location.0 + target_image.width(),
        y2: result.max_value_location.1 + target_image.height(),
    })
}

/// 滑块匹配。
pub fn slide_match_with_path<P1, P2>(
    target_image: P1,
    background_image: P2,
) -> anyhow::Result<SlideBBox>
where
    P1: AsRef<std::path::Path>,
    P2: AsRef<std::path::Path>,
{
    slide_match(
        std::fs::read(target_image)?,
        std::fs::read(background_image)?,
    )
}

/// 如果小图无过多背景部分，可以使用简单滑块匹配。
pub fn simple_slide_match<I1, I2>(
    target_image: I1,
    background_image: I2,
) -> anyhow::Result<SlideBBox>
where
    I1: AsRef<[u8]>,
    I2: AsRef<[u8]>,
{
    let target_image = load_image(target_image.as_ref())?;
    let background_image = load_image(background_image.as_ref())?;

    anyhow::ensure!(
        background_image.width() >= target_image.width(),
        "背景图片的宽度必须大于等于目标图标的宽度"
    );

    anyhow::ensure!(
        background_image.height() >= target_image.height(),
        "背景图片的高度必须大于等于目标图标的高度"
    );

    // 使用 canny 进行边缘检测。然后对背景图片进行同样的处理
    // 接着，使用 match_template 函数进行模板匹配，得到匹配结果矩阵
    // 然后使用 find_extremes 函数找到结果矩阵中的最大值和最小值
    // 并得到最大值所在的位置 loc，根据目标图片的大小和 loc 计算出目标物体的位置信息
    let target_image = target_image.to_luma8();
    let background_image = background_image.to_luma8();
    check_match_work(&target_image, &background_image)?;
    let target_image = imageproc::edges::canny(&target_image, 100.0, 200.0);
    let background_image = imageproc::edges::canny(&background_image, 100.0, 200.0);
    let result =
        imageproc::template_matching::find_extremes(&imageproc::template_matching::match_template(
            &background_image,
            &target_image,
            imageproc::template_matching::MatchTemplateMethod::CrossCorrelationNormalized,
        ));

    Ok(SlideBBox {
        target_x: 0,
        target_y: 0,
        x1: result.max_value_location.0,
        y1: result.max_value_location.1,
        x2: result.max_value_location.0 + target_image.width(),
        y2: result.max_value_location.1 + target_image.height(),
    })
}

/// 如果小图无过多背景部分，可以使用简单滑块匹配。
pub fn simple_slide_match_with_path<P1, P2>(
    target_image: P1,
    background_image: P2,
) -> anyhow::Result<SlideBBox>
where
    P1: AsRef<std::path::Path>,
    P2: AsRef<std::path::Path>,
{
    simple_slide_match(
        std::fs::read(target_image)?,
        std::fs::read(background_image)?,
    )
}

/// 坑位匹配。
pub fn slide_comparison<I1, I2>(
    target_image: I1,
    background_image: I2,
) -> anyhow::Result<(u32, u32)>
where
    I1: AsRef<[u8]>,
    I2: AsRef<[u8]>,
{
    let target_image = load_image(target_image.as_ref())?;
    let background_image = load_image(background_image.as_ref())?;

    anyhow::ensure!(
        target_image.width() == background_image.width()
            && target_image.height() == background_image.height(),
        "图片尺寸不相等"
    );

    let target = target_image.to_rgb8();
    let background = background_image.to_rgb8();
    let mut start_x = 0;
    let mut start_y = 0;
    for x in 0..target.width() {
        let mut count = 0;
        let mut first_y = None;
        for y in 0..target.height() {
            if target[(x, y)]
                .0
                .iter()
                .zip(background[(x, y)].0.iter())
                .any(|(a, b)| a.abs_diff(*b) > 80)
            {
                first_y.get_or_insert(y);
                count += 1;
            }
        }
        if count >= 5 {
            start_x = (x + 2).min(target.width() - 1);
            start_y = first_y.unwrap_or(0);
            break;
        }
    }

    Ok((start_x, start_y))
}

/// 坑位匹配。
pub fn slide_comparison_with_path<P1, P2>(
    target_image: P1,
    background_image: P2,
) -> anyhow::Result<(u32, u32)>
where
    P1: AsRef<std::path::Path>,
    P2: AsRef<std::path::Path>,
{
    slide_comparison(
        std::fs::read(target_image)?,
        std::fs::read(background_image)?,
    )
}

/// 判断是否为自定义模型。
pub fn is_diy<MODEL>(model: MODEL) -> bool
where
    MODEL: AsRef<[u8]>,
{
    // 比较 common.onnx 和 common_old.onnx 的 sha256
    let sha256 = sha256::digest(model.as_ref());

    sha256 != "33b5cd351ee94e73a6bf8fa18c415ed8b819b3ffd342e267c30d8ad8334e34e8"
        && sha256 != "b8f2ad9cbc1f2e3922a6cb9459e30824e7e2467f3fb4fd61420640e34ea0bf68"
}

/// 将图片的透明部分用白色填充。
fn png_rgba_black_preprocess(image: &image::DynamicImage) -> image::DynamicImage {
    let (width, height) = image::GenericImageView::dimensions(image);
    let mut new_image = image::DynamicImage::new_rgb8(width, height);

    for y in 0..height {
        for x in 0..width {
            let rgba_pixel = image::GenericImageView::get_pixel(image, x, y);

            let alpha = u16::from(rgba_pixel[3]);
            let composite =
                |v: u8| ((u16::from(v) * alpha + 255 * (255 - alpha) + 127) / 255) as u8;
            let rgb_pixel = image::Rgba([
                composite(rgba_pixel[0]),
                composite(rgba_pixel[1]),
                composite(rgba_pixel[2]),
                255,
            ]);

            image::GenericImage::put_pixel(&mut new_image, x, y, rgb_pixel);
        }
    }

    new_image
}

/// 内容识别需要用到的配置。
///
/// `../model/common_charset.json`
/// `../model/common_old_charset.json`
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Charset {
    /// 是否为 cnn 模型。
    pub word: bool,

    /// 宽度，高度，如果宽度为 -1，则自动调整，高度必须为 16 的倍数。
    pub image: [i64; 2],

    /// 通道数量。
    pub channel: i64,

    /// 字符集。
    pub charset: Vec<String>,
}

impl std::str::FromStr for Charset {
    type Err = serde_json::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        serde_json::from_str(s)
    }
}

/// 文字坐标。
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct BBox {
    pub x1: u32,
    pub y1: u32,
    pub x2: u32,
    pub y2: u32,
}

/// 滑块坐标。
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct SlideBBox {
    /// 透明部分的 x 偏移。
    pub target_x: u32,

    /// 透明部分的 y 偏移。
    pub target_y: u32,

    pub x1: u32,
    pub y1: u32,
    pub x2: u32,
    pub y2: u32,
}

/// 字符集和概率。
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CharacterProbability {
    pub text: Option<String>,
    pub charset: Vec<String>,
    pub probability: Vec<Vec<f32>>,
    pub confidence: Option<f64>,
}

/// 有时候只想获取 probability，而不获取 text 和 confidence。
impl CharacterProbability {
    pub fn get_text(&mut self) -> &str {
        self.text.get_or_insert_with(|| {
            // A blank resets CTC repetition, but is not part of the text.
            let mut previous = None;
            let mut text = String::new();
            for row in &self.probability {
                let index = row
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.total_cmp(b))
                    .map(|(i, _)| i);
                if index != previous {
                    if let Some(character) = index.and_then(|i| self.charset.get(i)) {
                        text.push_str(character);
                    }
                }
                previous = index;
            }
            text
        })
    }

    pub fn get_confidence(&mut self) -> f64 {
        *self.confidence.get_or_insert_with(|| {
            let mut max_sum = 0.0;
            let mut count = 0usize;

            for i in &self.probability {
                if let Some(v) = i.iter().fold(None, |acc: Option<f64>, &i| {
                    acc.map_or(Some(i as f64), |max| Some(max.max(i as f64)))
                }) {
                    max_sum += v;
                    count += 1;
                }
            }

            if count == 0 {
                0.0
            } else {
                max_sum / count as f64
            }
        })
    }
}

pub trait MapJson {
    fn json(&self) -> String;
}

impl MapJson for BBox {
    fn json(&self) -> String {
        serde_json::to_string(self).unwrap()
    }
}

impl MapJson for Vec<BBox> {
    fn json(&self) -> String {
        serde_json::to_string(self).unwrap()
    }
}

impl MapJson for SlideBBox {
    fn json(&self) -> String {
        serde_json::to_string(self).unwrap()
    }
}

impl MapJson for (u32, u32) {
    fn json(&self) -> String {
        serde_json::to_string(self).unwrap()
    }
}

impl MapJson for CharacterProbability {
    fn json(&self) -> String {
        serde_json::to_string(self).unwrap()
    }
}

impl MapJson for Vec<(BBox, String)> {
    fn json(&self) -> String {
        serde_json::to_string(self).unwrap()
    }
}

pub trait MapBBox {
    fn to_tuple(&self) -> Vec<(u32, u32, u32, u32)>;

    fn to_vec(&self) -> Vec<Vec<u32>>;
}

impl MapBBox for Vec<BBox> {
    fn to_tuple(&self) -> Vec<(u32, u32, u32, u32)> {
        self.iter()
            .map(|v| (v.x1, v.y1, v.x2, v.y2))
            .collect::<Vec<_>>()
    }

    fn to_vec(&self) -> Vec<Vec<u32>> {
        self.iter()
            .map(|v| vec![v.x1, v.y1, v.x2, v.y2])
            .collect::<Vec<_>>()
    }
}

const NMS_THR: f32 = 0.45;
const SCORE_THR: f32 = 0.1;
const MODEL_WIDTH: u32 = 416;
const MODEL_HEIGHT: u32 = 416;
const STRIDES: [u32; 3] = [8, 16, 32];

/// 字符集范围。
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum CharsetRange {
    /// 纯整数 0-9。
    Digit,

    /// 纯小写字母 a-z。
    Lowercase,

    /// 纯大写字母 a-z。
    Uppercase,

    /// 小写字母 a-z + 大写字母 A-Z。
    LowercaseUppercase,

    /// 小写字母 a-z + 整数 0-9。
    LowercaseDigit,

    /// 大写字母 A-Z + 整数 0-9。
    UppercaseDigit,

    /// 小写字母 a-z + 大写字母 A-Z + 整数 0-9。
    LowercaseUppercaseDigit,

    /// 默认字符集，删除小写字母 a-z、大写字母 A-Z、整数 0-9。
    DefaultCharsetLowercaseUppercaseDigit,

    /// 自定义字符集，例如 `0123456789+-x/=`。
    Other(String),

    /// 直接设置字符集，即 set_ranges 处理后的结果。
    Charset(Vec<String>),
}

impl From<i32> for CharsetRange {
    fn from(value: i32) -> Self {
        match value {
            0 => Self::Digit,
            1 => Self::Lowercase,
            2 => Self::Uppercase,
            3 => Self::LowercaseUppercase,
            4 => Self::LowercaseDigit,
            5 => Self::UppercaseDigit,
            6 => Self::LowercaseUppercaseDigit,
            7 => Self::DefaultCharsetLowercaseUppercaseDigit,
            _ => panic!("invalid charset range: {}", value),
        }
    }
}

impl From<&str> for CharsetRange {
    fn from(value: &str) -> Self {
        CharsetRange::Other(value.to_string())
    }
}

impl From<String> for CharsetRange {
    fn from(value: String) -> Self {
        CharsetRange::Other(value)
    }
}

impl From<&String> for CharsetRange {
    fn from(value: &String) -> Self {
        CharsetRange::Other(value.clone())
    }
}

impl From<Vec<String>> for CharsetRange {
    fn from(value: Vec<String>) -> Self {
        CharsetRange::Charset(value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Color {
    /// 红色。
    Red,

    /// 蓝色。
    Blue,

    /// 绿色。
    Green,

    /// 黄色。
    Yellow,

    /// 橙色。
    Orange,

    /// 紫色。
    Purple,

    /// 青色。
    Cyan,

    /// 黑色。
    Black,

    /// 白色。
    White,

    /// 灰色。
    Gray,
}

impl<T: AsRef<str>> From<T> for Color {
    fn from(value: T) -> Self {
        match value.as_ref().to_ascii_lowercase().as_str() {
            "red" => Color::Red,
            "blue" => Color::Blue,
            "green" => Color::Green,
            "yellow" => Color::Yellow,
            "orange" => Color::Orange,
            "purple" => Color::Purple,
            "cyan" => Color::Cyan,
            "black" => Color::Black,
            "white" => Color::White,
            "gray" => Color::Gray,
            _ => panic!("unknown color: {}", value.as_ref()),
        }
    }
}

pub type HsvRange = ((u8, u8, u8), (u8, u8, u8));

pub trait IntoHsvRange {
    fn into_hsv_ranges(self) -> Vec<HsvRange>;
}

impl IntoHsvRange for Color {
    fn into_hsv_ranges(self) -> Vec<HsvRange> {
        match self {
            Color::Red => vec![
                ((0, 50, 50), (10, 255, 255)),
                ((170, 50, 50), (180, 255, 255)),
            ],
            Color::Blue => vec![((100, 50, 50), (140, 255, 255))],
            Color::Green => vec![((40, 50, 50), (80, 255, 255))],
            Color::Yellow => vec![((20, 50, 50), (40, 255, 255))],
            Color::Orange => vec![((10, 50, 50), (20, 255, 255))],
            Color::Purple => vec![((140, 50, 50), (170, 255, 255))],
            Color::Cyan => vec![((80, 50, 50), (100, 255, 255))],
            Color::Black => vec![((0, 0, 0), (180, 255, 30))],
            Color::White => vec![((0, 0, 200), (180, 30, 255))],
            Color::Gray => vec![((0, 0, 30), (180, 30, 200))],
        }
    }
}

/// 颜色过滤。
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(untagged)]
pub enum ColorFilter {
    /// HSV 范围，每个元素是一个 (min_hsv, max_hsv) 的元组。
    /// 例如: `[((0, 50, 50), (10, 255, 255))]`。
    HSVRanges(Vec<HsvRange>),

    /// 颜色范围，例如: `["red", "blue"]`。
    ColorRanges(Vec<Color>),

    /// 单个颜色。
    Color(Color),
}

impl IntoHsvRange for ColorFilter {
    fn into_hsv_ranges(self) -> Vec<HsvRange> {
        match self {
            ColorFilter::HSVRanges(v) => v,
            ColorFilter::ColorRanges(v) => ColorFilter::from(v).into_hsv_ranges(),
            ColorFilter::Color(v) => v.into_hsv_ranges(),
        }
    }
}

impl IntoHsvRange for &str {
    fn into_hsv_ranges(self) -> Vec<HsvRange> {
        Color::from(self).into_hsv_ranges()
    }
}

impl IntoHsvRange for String {
    fn into_hsv_ranges(self) -> Vec<HsvRange> {
        Color::from(self).into_hsv_ranges()
    }
}

impl IntoHsvRange for &String {
    fn into_hsv_ranges(self) -> Vec<HsvRange> {
        Color::from(self).into_hsv_ranges()
    }
}

impl IntoHsvRange for ((u8, u8, u8), (u8, u8, u8)) {
    fn into_hsv_ranges(self) -> Vec<HsvRange> {
        vec![self]
    }
}

impl From<&str> for ColorFilter {
    fn from(value: &str) -> Self {
        Color::from(value).into_hsv_ranges().into()
    }
}

impl From<String> for ColorFilter {
    fn from(value: String) -> Self {
        value.as_str().into()
    }
}

impl From<&String> for ColorFilter {
    fn from(value: &String) -> Self {
        value.as_str().into()
    }
}

impl From<Color> for ColorFilter {
    fn from(value: Color) -> Self {
        value.into_hsv_ranges().into()
    }
}

impl From<&[Color]> for ColorFilter {
    fn from(value: &[Color]) -> Self {
        value.to_vec().into()
    }
}

impl From<&[&str]> for ColorFilter {
    fn from(value: &[&str]) -> Self {
        value
            .iter()
            .map(|v| Color::from(*v))
            .collect::<Vec<_>>()
            .into()
    }
}

impl<T, const N: usize> From<[T; N]> for ColorFilter
where
    T: IntoHsvRange,
{
    fn from(value: [T; N]) -> Self {
        value
            .into_iter()
            .flat_map(|v| v.into_hsv_ranges())
            .collect::<Vec<_>>()
            .into()
    }
}

impl<'a, T, const N: usize> From<&'a [T; N]> for ColorFilter
where
    T: IntoHsvRange + Clone,
{
    fn from(value: &'a [T; N]) -> Self {
        value
            .iter()
            .flat_map(|v| v.clone().into_hsv_ranges())
            .collect::<Vec<_>>()
            .into()
    }
}

impl From<&[String]> for ColorFilter {
    fn from(values: &[String]) -> Self {
        values.to_vec().into()
    }
}

impl From<Vec<&str>> for ColorFilter {
    fn from(value: Vec<&str>) -> Self {
        value
            .into_iter()
            .flat_map(|v| Color::from(v).into_hsv_ranges())
            .collect::<Vec<_>>()
            .into()
    }
}

impl From<Vec<String>> for ColorFilter {
    fn from(value: Vec<String>) -> Self {
        value
            .iter()
            .flat_map(|v| Color::from(v.as_str()).into_hsv_ranges())
            .collect::<Vec<_>>()
            .into()
    }
}

impl From<Vec<Color>> for ColorFilter {
    fn from(value: Vec<Color>) -> Self {
        value
            .into_iter()
            .flat_map(|v| v.into_hsv_ranges())
            .collect::<Vec<_>>()
            .into()
    }
}

impl From<Vec<HsvRange>> for ColorFilter {
    fn from(value: Vec<HsvRange>) -> Self {
        ColorFilter::HSVRanges(value)
    }
}

impl ColorFilter {
    /// 过滤颜色，例如 ColorFilter::from("green").filter(image) 表示只保留绿色。
    pub fn filter<I>(&self, image: I) -> anyhow::Result<image::DynamicImage>
    where
        I: AsRef<[u8]>,
    {
        Ok(self.filter_decoded(load_image(image.as_ref())?))
    }

    fn filter_decoded(&self, image: image::DynamicImage) -> image::DynamicImage {
        let mut image = image.to_rgb8();
        let ranges = self.clone().into_hsv_ranges();
        for pixel in image.pixels_mut() {
            let r = f32::from(pixel[0]) / 255.0;
            let g = f32::from(pixel[1]) / 255.0;
            let b = f32::from(pixel[2]) / 255.0;
            let max = r.max(g).max(b);
            let delta = max - r.min(g).min(b);
            let saturation = if max == 0.0 { 0.0 } else { delta / max };
            let hue = if delta == 0.0 {
                0.0
            } else if max == r {
                60.0 * ((g - b) / delta).rem_euclid(6.0)
            } else if max == g {
                60.0 * ((b - r) / delta + 2.0)
            } else {
                60.0 * ((r - g) / delta + 4.0)
            };
            let hsv = [
                (hue / 2.0).round() as u8,
                (saturation * 255.0).round() as u8,
                (max * 255.0).round() as u8,
            ];
            if !ranges.iter().any(|(lo, hi)| {
                hsv[0] >= lo.0
                    && hsv[0] <= hi.0
                    && hsv[1] >= lo.1
                    && hsv[1] <= hi.1
                    && hsv[2] >= lo.2
                    && hsv[2] <= hi.2
            }) {
                *pixel = image::Rgb([255, 255, 255]);
            }
        }
        image::DynamicImage::ImageRgb8(image)
    }
}

#[derive(Debug)]
pub struct Ddddocr<'a> {
    diy: bool,
    session: ort::Session,
    charset: Option<std::borrow::Cow<'a, Charset>>,
    charset_range: Vec<String>,
}

/// 因为自带模型和自定义模型的参数不同，所以在创建模型的时候会自动判断是否为自定义模型。
impl<'a> Ddddocr<'a> {
    /// 设置运行库的路径。
    ///
    /// - Unix: `/etc/.../libonnxruntime.so`
    /// - Windows: `C:\\Program Files\\...\\onnxruntime.dll`
    ///
    /// 不需要手动调用此函数，因为程序会自动寻找运行库路径。
    #[cfg(feature = "load-dynamic")]
    pub fn set_onnxruntime_path<P>(path: P) -> anyhow::Result<()>
    where
        P: AsRef<std::path::Path>,
    {
        let path = path.as_ref();

        let save = std::panic::take_hook();

        std::panic::set_hook(Box::new(|_| {}));

        let result = std::panic::catch_unwind(|| {
            ort::init_from(path.to_string_lossy().to_string())
                .commit()
                .unwrap()
        });

        std::panic::set_hook(save);

        result.map_err(|v| {
            anyhow::anyhow!(
                "{}",
                v.downcast::<String>().unwrap_or(Box::new(format!(
                    "failed to load the runtime library: {}",
                    path.display()
                )))
            )
        })
    }

    /// 从内存加载模型和字符集，只能使用内容识别，使用目标检测会恐慌。
    pub fn new<MODEL>(model: MODEL, charset: Charset) -> anyhow::Result<Self>
    where
        MODEL: AsRef<[u8]>,
    {
        Self::new_with_threads(model, charset, 1)
    }

    pub fn new_with_threads<MODEL: AsRef<[u8]>>(
        model: MODEL,
        charset: Charset,
        threads: usize,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(
            matches!(charset.channel, 1 | 3)
                && charset.image[1] > 0
                && charset.image[1] <= 4096
                && (charset.image[0] == -1 || (charset.image[0] > 0 && charset.image[0] <= 8192))
                && !charset.charset.is_empty(),
            "invalid model charset configuration"
        );
        Ok(Self {
            diy: is_diy(model.as_ref()),
            session: cpu_session(model.as_ref(), threads)?,
            charset: Some(std::borrow::Cow::Owned(charset)),
            charset_range: Vec::new(),
        })
    }

    /// 从内存加载模型和字符集，只能使用内容识别，使用目标检测会恐慌。
    pub fn new_ref<MODEL>(model: MODEL, charset: &'a Charset) -> anyhow::Result<Self>
    where
        MODEL: AsRef<[u8]>,
    {
        Ok(Self {
            diy: is_diy(model.as_ref()),
            session: cpu_session(model.as_ref(), 1)?,
            charset: Some(std::borrow::Cow::Borrowed(charset)),
            charset_range: Vec::new(),
        })
    }

    /// 从内存加载模型和字符集，只能使用内容识别，使用目标检测会恐慌。
    #[cfg(feature = "cuda")]
    pub fn new_cuda<MODEL>(model: MODEL, charset: Charset, device_id: i32) -> anyhow::Result<Self>
    where
        MODEL: AsRef<[u8]>,
    {
        let builder = ort::Session::builder()?;

        let cuda = ort::CUDAExecutionProvider::default()
            .with_device_id(device_id)
            .with_arena_extend_strategy(ort::ArenaExtendStrategy::NextPowerOfTwo)
            .with_memory_limit(2 * 1024 * 1024 * 1024)
            .with_conv_algorithm_search(ort::CUDAExecutionProviderCuDNNConvAlgoSearch::Exhaustive)
            .with_copy_in_default_stream(true);

        if !ort::ExecutionProvider::is_available(&cuda)? {
            anyhow::bail!("please compile ONNX Runtime with CUDA!")
        }

        ort::ExecutionProvider::register(&cuda, &builder)?;

        Ok(Self {
            diy: is_diy(model.as_ref()),
            session: builder.commit_from_memory(model.as_ref())?,
            charset: Some(std::borrow::Cow::Owned(charset)),
            charset_range: Vec::new(),
        })
    }

    /// 从内存加载模型和字符集，只能使用内容识别，使用目标检测会恐慌。
    #[cfg(feature = "cuda")]
    pub fn new_cuda_ref<MODEL>(
        model: MODEL,
        charset: &'a Charset,
        device_id: i32,
    ) -> anyhow::Result<Self>
    where
        MODEL: AsRef<[u8]>,
    {
        let builder = ort::Session::builder()?;

        let cuda = ort::CUDAExecutionProvider::default()
            .with_device_id(device_id)
            .with_arena_extend_strategy(ort::ArenaExtendStrategy::NextPowerOfTwo)
            .with_memory_limit(2 * 1024 * 1024 * 1024)
            .with_conv_algorithm_search(ort::CUDAExecutionProviderCuDNNConvAlgoSearch::Exhaustive)
            .with_copy_in_default_stream(true);

        if !ort::ExecutionProvider::is_available(&cuda)? {
            anyhow::bail!("please compile ONNX Runtime with CUDA!")
        }

        ort::ExecutionProvider::register(&cuda, &builder)?;

        Ok(Self {
            diy: is_diy(model.as_ref()),
            session: builder.commit_from_memory(model.as_ref())?,
            charset: Some(std::borrow::Cow::Borrowed(charset)),
            charset_range: Vec::new(),
        })
    }

    /// 从内存加载模型，只能使用目标检测，使用内容识别会恐慌。
    pub fn new_model<MODEL>(model: MODEL) -> anyhow::Result<Self>
    where
        MODEL: AsRef<[u8]>,
    {
        Self::new_model_with_threads(model, 1)
    }

    pub fn new_model_with_threads<MODEL: AsRef<[u8]>>(
        model: MODEL,
        threads: usize,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            diy: is_diy(model.as_ref()),
            session: cpu_session(model.as_ref(), threads)?,
            charset: None,
            charset_range: Vec::new(),
        })
    }

    /// 从内存加载模型，只能使用目标检测，使用内容识别会恐慌。
    #[cfg(feature = "cuda")]
    pub fn new_model_cuda<MODEL>(model: MODEL, device_id: i32) -> anyhow::Result<Self>
    where
        MODEL: AsRef<[u8]>,
    {
        let builder = ort::Session::builder()?;

        let cuda = ort::CUDAExecutionProvider::default()
            .with_device_id(device_id)
            .with_arena_extend_strategy(ort::ArenaExtendStrategy::NextPowerOfTwo)
            .with_memory_limit(2 * 1024 * 1024 * 1024)
            .with_conv_algorithm_search(ort::CUDAExecutionProviderCuDNNConvAlgoSearch::Exhaustive)
            .with_copy_in_default_stream(true);

        if !ort::ExecutionProvider::is_available(&cuda)? {
            anyhow::bail!("please compile ONNX Runtime with CUDA!")
        }

        ort::ExecutionProvider::register(&cuda, &builder)?;

        Ok(Self {
            diy: is_diy(model.as_ref()),
            session: builder.commit_from_memory(model.as_ref())?,
            charset: None,
            charset_range: Vec::new(),
        })
    }

    /// 从文件加载模型和字符集，只能使用内容识别，使用目标检测会恐慌。
    pub fn with_model_charset<PATH1, PATH2>(model: PATH1, charset: PATH2) -> anyhow::Result<Self>
    where
        PATH1: AsRef<std::path::Path>,
        PATH2: AsRef<std::path::Path>,
    {
        Self::new(
            std::fs::read(model)?,
            serde_json::from_str(&std::fs::read_to_string(charset)?)?,
        )
    }

    /// 从文件加载模型和字符集，只能使用内容识别，使用目标检测会恐慌。
    #[cfg(feature = "cuda")]
    pub fn with_model_charset_cuda<PATH1, PATH2>(
        model: PATH1,
        charset: PATH2,
        device_id: i32,
    ) -> anyhow::Result<Self>
    where
        PATH1: AsRef<std::path::Path>,
        PATH2: AsRef<std::path::Path>,
    {
        Self::new_cuda(
            std::fs::read(model)?,
            serde_json::from_str(&std::fs::read_to_string(charset)?)?,
            device_id,
        )
    }

    /// 从文件加载模型，只能使用目标检测，使用内容识别会恐慌。
    pub fn with_model<P>(model: P) -> anyhow::Result<Self>
    where
        P: AsRef<std::path::Path>,
    {
        Self::new_model(std::fs::read(model)?)
    }

    /// 从文件加载模型，只能使用目标检测，使用内容识别会恐慌。
    #[cfg(feature = "cuda")]
    pub fn with_model_cuda<P>(model: P, device_id: i32) -> anyhow::Result<Self>
    where
        P: AsRef<std::path::Path>,
    {
        Self::new_model_cuda(std::fs::read(model)?, device_id)
    }

    /// 根据给定 ranges 计算字符集范围。
    pub fn calc_ranges<R>(&self, ranges: R) -> Vec<String>
    where
        R: Into<CharsetRange>,
    {
        let mut new_charset = match ranges.into() {
            CharsetRange::Digit => "0123456789"
                .chars()
                .map(|v| v.to_string())
                .collect::<Vec<String>>(),
            CharsetRange::Lowercase => "abcdefghijklmnopqrstuvwxyz"
                .chars()
                .map(|v| v.to_string())
                .collect::<Vec<String>>(),
            CharsetRange::Uppercase => "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                .chars()
                .map(|v| v.to_string())
                .collect::<Vec<String>>(),
            CharsetRange::LowercaseUppercase => {
                "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
                    .chars()
                    .map(|v| v.to_string())
                    .collect::<Vec<String>>()
            }
            CharsetRange::LowercaseDigit => "abcdefghijklmnopqrstuvwxyz0123456789"
                .chars()
                .map(|v| v.to_string())
                .collect::<Vec<String>>(),
            CharsetRange::UppercaseDigit => "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                .chars()
                .map(|v| v.to_string())
                .collect::<Vec<String>>(),
            CharsetRange::LowercaseUppercaseDigit => {
                "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                    .chars()
                    .map(|v| v.to_string())
                    .collect::<Vec<String>>()
            }
            CharsetRange::DefaultCharsetLowercaseUppercaseDigit => {
                let delete_range = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                    .chars()
                    .collect::<Vec<char>>();

                self.charset
                    .as_ref()
                    .expect("only the ocr model can be used")
                    .charset
                    .clone()
                    .into_iter()
                    .filter(|v| v.chars().all(|c| !delete_range.contains(&c)))
                    .collect::<Vec<String>>()
            }
            CharsetRange::Other(v) => v.chars().map(|v| v.to_string()).collect::<Vec<String>>(),
            CharsetRange::Charset(v) => return v,
        };

        let mut seen = std::collections::HashSet::new();
        new_charset.retain(|character| !character.is_empty() && seen.insert(character.clone()));
        new_charset.push(String::new());

        new_charset
    }

    /// 限定 classification_probability 的字符范围，只能使用内容识别，使用目标检测会恐慌。
    pub fn set_ranges<R>(&mut self, ranges: R)
    where
        R: Into<CharsetRange>,
    {
        self.charset_range = self.calc_ranges(ranges)
    }

    /// 内容识别，返回全字符表的概率，可以通过 `set_ranges` 限定字符范围，仅限于使用官方模型。
    pub fn classification_probability<I>(&self, image: I) -> anyhow::Result<CharacterProbability>
    where
        I: AsRef<[u8]>,
    {
        self.classification_probability_with_options(image, false, None, None)
    }

    /// 内容识别，返回全字符表的概率，可以通过 `set_ranges` 限定字符范围，仅限于使用官方模型。
    /// 如果 `png_fix` 为 true，则支持透明黑色背景的 png 图片。
    pub fn classification_probability_with_png_fix<I>(
        &self,
        image: I,
        png_fix: bool,
    ) -> anyhow::Result<CharacterProbability>
    where
        I: AsRef<[u8]>,
    {
        self.classification_probability_with_options(image, png_fix, None, None)
    }

    /// 内容识别，返回全字符表的概率，可以通过 `set_ranges` 限定字符范围，仅限于使用官方模型。
    /// 如果 `filter` 为 red，则表示只识别红色。
    pub fn classification_probability_with_filter<I, F>(
        &self,
        image: I,
        filter: F,
    ) -> anyhow::Result<CharacterProbability>
    where
        I: AsRef<[u8]>,
        F: Into<ColorFilter>,
    {
        self.classification_probability_with_options(image, false, Some(filter.into()), None)
    }

    /// 内容识别，返回指定字符范围的概率，可通过 `ranges` 限定字符范围。
    pub fn classification_probability_with_ranges<I, R>(
        &self,
        image: I,
        ranges: R,
    ) -> anyhow::Result<CharacterProbability>
    where
        I: AsRef<[u8]>,
        R: Into<CharsetRange>,
    {
        self.classification_probability_with_options(image, false, None, Some(ranges.into()))
    }

    /// 内容识别，返回指定字符范围的概率，可通过 `png_fix` 支持透明黑色背景的 png 图片，并通过 `ranges` 限定字符范围。
    pub fn classification_probability_with_png_fix_and_ranges<I, R>(
        &self,
        image: I,
        png_fix: bool,
        ranges: R,
    ) -> anyhow::Result<CharacterProbability>
    where
        I: AsRef<[u8]>,
        R: Into<CharsetRange>,
    {
        self.classification_probability_with_options(image, png_fix, None, Some(ranges.into()))
    }

    /// 内容识别，返回指定字符范围的概率，可通过 `filter` 指定颜色过滤，并通过 `ranges` 限定字符范围。
    pub fn classification_probability_with_filter_and_ranges<I, F, R>(
        &self,
        image: I,
        filter: F,
        ranges: R,
    ) -> anyhow::Result<CharacterProbability>
    where
        I: AsRef<[u8]>,
        F: Into<ColorFilter>,
        R: Into<CharsetRange>,
    {
        self.classification_probability_with_options(
            image,
            false,
            Some(filter.into()),
            Some(ranges.into()),
        )
    }

    /// 内容识别，返回全字符表的概率，可以通过 set_ranges 限定字符范围，仅限于使用官方模型。
    /// 如果 png_fix 为 true，则支持透明黑色背景的 png 图片。
    /// 如果 filter 为 red，则表示只识别红色。
    /// 如果 ranges 为 None，则使用 set_ranges 的字符范围。
    pub fn classification_probability_with_options<I>(
        &self,
        image: I,
        png_fix: bool,
        filter: Option<ColorFilter>,
        ranges: Option<CharsetRange>,
    ) -> anyhow::Result<CharacterProbability>
    where
        I: AsRef<[u8]>,
    {
        anyhow::ensure!(!self.diy, "probability requires an official OCR model");
        let calculated;
        let charset_ranges = match ranges {
            Some(value) => {
                calculated = self.calc_ranges(value);
                &calculated
            }
            None => &self.charset_range,
        };

        let tensor = self.prepare_tensor(image.as_ref(), png_fix, filter)?;
        anyhow::ensure!(
            tensor.shape()[3] <= 1024,
            "probability output requires resized width <= 1024 pixels"
        );
        let charset = &self
            .charset
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("OCR model required"))?
            .charset;

        let ort_outs = &self.session.run(ort::inputs![tensor]?)?;

        let ort_outs = &ort_outs[0].try_extract_tensor::<f32>()?;

        anyhow::ensure!(
            ort_outs.shape().last() == Some(&charset.len()),
            "model output does not match charset"
        );
        let lookup: std::collections::HashMap<&str, usize> = if charset_ranges.is_empty() {
            std::collections::HashMap::new()
        } else {
            charset
                .iter()
                .enumerate()
                .map(|(index, character)| (character.as_str(), index))
                .collect()
        };
        let indices: Vec<Option<usize>> = charset_ranges
            .iter()
            .map(|c| lookup.get(c.as_str()).copied())
            .collect();
        let mut probabilities = Vec::new();
        for row in ort_outs.rows() {
            let row = row
                .as_slice()
                .ok_or_else(|| anyhow::anyhow!("non-contiguous model output"))?;
            anyhow::ensure!(row.iter().all(|v| v.is_finite()), "non-finite model output");
            let values = softmax(row);
            probabilities.push(if charset_ranges.is_empty() {
                values
            } else {
                indices
                    .iter()
                    .map(|index| index.map_or(-1.0, |i| values[i]))
                    .collect()
            });
        }
        Ok(CharacterProbability {
            text: None,
            charset: if charset_ranges.is_empty() {
                charset.clone()
            } else {
                charset_ranges.clone()
            },
            probability: probabilities,
            confidence: None,
        })
    }

    fn prepare_tensor(
        &self,
        data: &[u8],
        png_fix: bool,
        filter: Option<ColorFilter>,
    ) -> anyhow::Result<ndarray::Array4<f32>> {
        self.prepare_decoded_tensor(load_image(data)?, png_fix, filter)
    }

    fn prepare_decoded_tensor(
        &self,
        mut image: image::DynamicImage,
        png_fix: bool,
        filter: Option<ColorFilter>,
    ) -> anyhow::Result<ndarray::Array4<f32>> {
        let config = self
            .charset
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("OCR model required"))?;
        if png_fix {
            image = png_rgba_black_preprocess(&image);
        }
        if let Some(filter) = filter {
            image = filter.filter_decoded(image);
        }
        let height = config.image[1] as u32;
        let width = if config.image[0] == -1 {
            if config.word {
                height
            } else {
                (u64::from(image.width()) * u64::from(height) / u64::from(image.height())).max(1)
                    as u32
            }
        } else {
            config.image[0] as u32
        };
        anyhow::ensure!(
            height > 0 && height <= 4096 && width > 0 && width <= 8192,
            "resized OCR image exceeds limits"
        );
        let image = image.resize_exact(width, height, image::imageops::FilterType::Lanczos3);
        let channels = config.channel as usize;
        anyhow::ensure!(matches!(channels, 1 | 3), "model channels must be 1 or 3");
        let mut tensor = ndarray::Array4::zeros((1, channels, height as usize, width as usize));
        if channels == 1 {
            for (x, y, pixel) in image.to_luma8().enumerate_pixels() {
                let value = f32::from(pixel[0]) / 255.0;
                tensor[[0, 0, y as usize, x as usize]] = if self.diy {
                    (value - 0.456) / 0.224
                } else {
                    (value - 0.5) / 0.5
                };
            }
        } else {
            for (x, y, pixel) in image.to_rgb8().enumerate_pixels() {
                for c in 0..3 {
                    tensor[[0, c, y as usize, x as usize]] = (f32::from(pixel[c]) / 255.0
                        - [0.485, 0.456, 0.406][c])
                        / [0.229, 0.224, 0.225][c];
                }
            }
        }
        Ok(tensor)
    }

    /// 内容识别。
    pub fn classification<I>(&self, image: I) -> anyhow::Result<String>
    where
        I: AsRef<[u8]>,
    {
        self.classification_with_options(image, false, None)
    }

    /// 内容识别。
    pub fn classification_with_path<P>(&self, path: P) -> anyhow::Result<String>
    where
        P: AsRef<std::path::Path>,
    {
        self.classification(std::fs::read(path)?)
    }

    /// 内容识别，如果 png_fix 为 true，则支持透明黑色背景的 png 图片。
    pub fn classification_with_png_fix<I>(&self, image: I, png_fix: bool) -> anyhow::Result<String>
    where
        I: AsRef<[u8]>,
    {
        self.classification_with_options(image, png_fix, None)
    }

    /// 内容识别，如果 png_fix 为 true，则支持透明黑色背景的 png 图片。
    pub fn classification_with_path_png_fix<P>(
        &self,
        path: P,
        png_fix: bool,
    ) -> anyhow::Result<String>
    where
        P: AsRef<std::path::Path>,
    {
        self.classification_with_png_fix(std::fs::read(path)?, png_fix)
    }

    /// 内容识别，如果 filter 为 red，则表示只识别红色。
    pub fn classification_with_filter<I, F>(&self, image: I, filter: F) -> anyhow::Result<String>
    where
        I: AsRef<[u8]>,
        F: Into<ColorFilter>,
    {
        self.classification_with_options(image, false, Some(filter.into()))
    }

    /// 内容识别，如果 filter 为 red，则表示只识别红色。
    pub fn classification_with_path_filter<P, F>(
        &self,
        path: P,
        filter: F,
    ) -> anyhow::Result<String>
    where
        P: AsRef<std::path::Path>,
        F: Into<ColorFilter>,
    {
        self.classification_with_filter(std::fs::read(path)?, filter)
    }

    /// 内容识别，如果 png_fix 为 true，则支持透明黑色背景的 png 图片，如果 filter 为 red，则表示只识别红色。
    pub fn classification_with_options<I>(
        &self,
        image: I,
        png_fix: bool,
        filter: Option<ColorFilter>,
    ) -> anyhow::Result<String>
    where
        I: AsRef<[u8]>,
    {
        let tensor = self.prepare_tensor(image.as_ref(), png_fix, filter)?;
        self.classify_tensor(tensor)
    }

    fn classify_tensor(&self, tensor: ndarray::Array4<f32>) -> anyhow::Result<String> {
        let config = self
            .charset
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("OCR model required"))?;
        let word = config.word;
        let charset = &config.charset;

        if word {
            Ok(self.session.run(ort::inputs![tensor]?)?[1]
                .try_extract_tensor::<i64>()?
                .iter()
                .map(|&v| charset[v as usize].to_string())
                .collect::<String>())
        } else if self.diy {
            // todo: 自定义模型未经测试
            let result = &self.session.run(ort::inputs![tensor]?)?[0];
            let result = result.try_extract_tensor::<u32>()?;
            Ok(decode_ctc(
                result.iter().map(|&value| value as usize),
                charset,
            ))
        } else {
            let result = &self.session.run(ort::inputs![tensor]?)?[0];
            let result = result.try_extract_tensor::<f32>()?;

            // 输入长这样 [[[1,2,3,4], [1,2,3,4], [1,2,3,4]]]
            // 我们要获取   ^^^^^^^^^  ^^^^^^^^^  ^^^^^^^^^
            // 最后结果 [3, 3, 3]
            // 这是最大值的索引
            let result = result.rows().into_iter().map(|v| {
                // 找出数组中元素值最大的那个，然后获取他在数组中的索引
                v.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.total_cmp(b))
                    .unwrap_or((0, &0.0))
                    .0
            });

            Ok(decode_ctc(result, charset))
        }
    }

    /// 根据坐标裁剪图片，然后进行内容识别。
    pub fn classification_bbox<I>(
        &self,
        image: I,
        bbox: &Vec<BBox>,
    ) -> anyhow::Result<Vec<(BBox, String)>>
    where
        I: AsRef<[u8]>,
    {
        let image = load_image(image.as_ref())?;

        let mut result = Vec::new();

        for i in bbox {
            anyhow::ensure!(
                i.x1 < i.x2 && i.y1 < i.y2 && i.x2 <= image.width() && i.y2 <= image.height(),
                "invalid bounding box"
            );
            let cropped = image.crop_imm(i.x1, i.y1, i.x2 - i.x1, i.y2 - i.y1);
            let tensor = self.prepare_decoded_tensor(cropped, false, None)?;
            result.push((*i, self.classify_tensor(tensor)?));
        }

        Ok(result)
    }

    /// 根据坐标裁剪图片，然后进行内容识别。
    pub fn classification_bbox_with_path<P>(
        &self,
        path: P,
        bbox: &Vec<BBox>,
    ) -> anyhow::Result<Vec<(BBox, String)>>
    where
        P: AsRef<std::path::Path>,
    {
        self.classification_bbox(std::fs::read(path)?, bbox)
    }

    /// 目标检测。
    pub fn detection<I>(&self, image: I) -> anyhow::Result<Vec<BBox>>
    where
        I: AsRef<[u8]>,
    {
        let image = load_image(image.as_ref())?.to_rgb8();
        let (h, w) = (image.height(), image.width());
        let r = (MODEL_WIDTH as f32 / h as f32).min(MODEL_WIDTH as f32 / w as f32);
        let new_w = ((w as f32 * r) as u32).max(1);
        let new_h = ((h as f32 * r) as u32).max(1);

        let resized =
            image::imageops::resize(&image, new_w, new_h, image::imageops::FilterType::Triangle);

        let mut padded =
            image::RgbImage::from_pixel(MODEL_WIDTH, MODEL_HEIGHT, image::Rgb([114, 114, 114]));

        image::GenericImage::copy_from(&mut padded, &resized, 0, 0)?;

        let mut input_tensor = ndarray::Array::from_shape_vec(
            (1, 3, MODEL_WIDTH as usize, MODEL_HEIGHT as usize),
            vec![0f32; 3 * MODEL_WIDTH as usize * MODEL_HEIGHT as usize],
        )?;

        for (x, y, p) in padded.enumerate_pixels() {
            let x = x as usize;
            let y = y as usize;

            input_tensor[[0, 0, y, x]] = p[2] as f32;
            input_tensor[[0, 1, y, x]] = p[1] as f32;
            input_tensor[[0, 2, y, x]] = p[0] as f32;
        }

        let output = &self.session.run(ort::inputs![input_tensor]?)?[0];
        let output = output.try_extract_tensor::<f32>()?;
        let expected_boxes: usize = STRIDES
            .iter()
            .map(|stride| ((MODEL_WIDTH / stride) * (MODEL_HEIGHT / stride)) as usize)
            .sum();
        anyhow::ensure!(
            output.shape() == [1, expected_boxes, 6],
            "invalid detection model output shape"
        );
        let mut dets = Vec::new();
        let mut grid_offset = 0;

        for &stride in &STRIDES {
            let h_grid = MODEL_WIDTH / stride;
            let w_grid = MODEL_HEIGHT / stride;

            for gy in 0..h_grid {
                for gx in 0..w_grid {
                    let i = grid_offset + (gy * w_grid + gx) as usize;
                    let obj = output[[0, i, 4]];
                    let cls = output[[0, i, 5]];
                    let score = obj * cls;

                    if !score.is_finite() || score < SCORE_THR {
                        continue;
                    }

                    let cx = output[[0, i, 0]];
                    let cy = output[[0, i, 1]];
                    let w = output[[0, i, 2]];
                    let h = output[[0, i, 3]];
                    let x1 = (cx + gx as f32) * stride as f32;
                    let y1 = (cy + gy as f32) * stride as f32;
                    let bw = w.exp() * stride as f32;
                    let bh = h.exp() * stride as f32;
                    let mut x1 = x1 - bw / 2.0;
                    let mut y1 = y1 - bh / 2.0;
                    let mut x2 = x1 + bw;
                    let mut y2 = y1 + bh;

                    x1 /= r;
                    y1 /= r;
                    x2 /= r;
                    y2 /= r;

                    dets.push((score, [x1, y1, x2, y2]));
                }
            }
            grid_offset += (h_grid * w_grid) as usize;
        }

        let scores: Vec<f32> = dets.iter().map(|d| d.0).collect();
        let mut order: Vec<usize> = (0..scores.len()).collect();

        order.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]));

        let mut keep = Vec::new();

        while !order.is_empty() {
            let i = order[0];

            keep.push(i);

            let (x1, y1, x2, y2) = {
                let b = dets[i].1;
                (b[0], b[1], b[2], b[3])
            };

            let area_i = (x2 - x1 + 1.0) * (y2 - y1 + 1.0);
            let mut new_order = Vec::new();

            for &j in &order[1..] {
                let (x1j, y1j, x2j, y2j) = {
                    let b = dets[j].1;
                    (b[0], b[1], b[2], b[3])
                };

                let xx1 = x1.max(x1j);
                let yy1 = y1.max(y1j);
                let xx2 = x2.min(x2j);
                let yy2 = y2.min(y2j);
                let w = (xx2 - xx1 + 1.0).max(0.0);
                let h = (yy2 - yy1 + 1.0).max(0.0);
                let inter = w * h;
                let area_j = (x2j - x1j + 1.0) * (y2j - y1j + 1.0);
                let ovr = inter / (area_i + area_j - inter);

                if ovr <= NMS_THR {
                    new_order.push(j);
                }
            }
            order = new_order;
        }

        let (ow, oh) = (w as f32, h as f32);
        let mut result = Vec::new();

        for i in keep {
            let (x1, y1, x2, y2) = {
                let b = dets[i].1;
                (b[0], b[1], b[2], b[3])
            };

            let x1 = x1.clamp(0.0, ow) as u32;
            let y1 = y1.clamp(0.0, oh) as u32;
            let x2 = x2.clamp(0.0, ow) as u32;
            let y2 = y2.clamp(0.0, oh) as u32;

            result.push(crate::BBox { x1, y1, x2, y2 });
        }

        Ok(result)
    }

    /// 目标检测。
    pub fn detection_with_path<P>(&self, path: P) -> anyhow::Result<Vec<BBox>>
    where
        P: AsRef<std::path::Path>,
    {
        self.detection(std::fs::read(path)?)
    }
}

// cargo test --no-default-features --features download-binaries
#[cfg(test)]
mod tests {
    use image::Pixel;

    use super::*;

    fn read_image(path: &str) -> Vec<u8> {
        std::fs::read(path).unwrap_or_else(|e| {
            panic!("failed to read image {}: {}", path, e);
        })
    }

    #[test]
    fn classification_probability() {
        let mut ddddocr = ddddocr_classification().unwrap();

        // CharsetRange::LowercaseUppercase 大写字母和小写字母
        ddddocr.set_ranges(3);

        let mut result = ddddocr
            .classification_probability(read_image("image/3.png"))
            .unwrap();

        println!("识别结果: {}", result.get_text());
        println!("识别可信度: {}", result.get_confidence());

        // 哦呀，看来数据有点儿太多了，小心卡死哦！
        // println!("概率: {}", result.json());
    }

    #[test]
    fn classification_filter() {
        let ddddocr = ddddocr_classification().unwrap();

        println!(
            "{}",
            ddddocr
                .classification_with_filter(read_image("image/4.png"), "green")
                .unwrap()
        );

        println!(
            "{}",
            ddddocr
                .classification_with_filter(
                    read_image("image/4.png"),
                    [((40, 50, 50), (80, 255, 255))]
                )
                .unwrap()
        );
    }

    #[test]
    fn classification() {
        let ddddocr = ddddocr_classification().unwrap();

        println!(
            "{}",
            ddddocr.classification(read_image("image/1.png")).unwrap()
        );

        println!(
            "{}",
            ddddocr.classification(read_image("image/2.png")).unwrap()
        );

        println!(
            "{}",
            ddddocr.classification(read_image("image/3.png")).unwrap()
        );

        println!(
            "{}",
            ddddocr.classification(read_image("image/4.png")).unwrap()
        );

        println!(
            "{}",
            ddddocr.classification(read_image("image/su.png")).unwrap()
        );
    }

    #[test]
    fn classification_old() {
        let ddddocr = ddddocr_classification_old().unwrap();

        println!(
            "{}",
            ddddocr.classification(read_image("image/1.png")).unwrap()
        );

        println!(
            "{}",
            ddddocr.classification(read_image("image/2.png")).unwrap()
        );

        println!(
            "{}",
            ddddocr.classification(read_image("image/3.png")).unwrap()
        );

        println!(
            "{}",
            ddddocr.classification(read_image("image/4.png")).unwrap()
        );
    }

    #[test]
    fn classification_bbox() {
        let input = read_image("image/6.jpg");
        let ddddocr = ddddocr_detection().unwrap();
        let result = ddddocr.detection(input.clone()).unwrap();
        let ddddocr = ddddocr_classification().unwrap();
        let result = ddddocr.classification_bbox(input, &result).unwrap();

        println!("{:?}", result);
    }

    #[test]
    fn detection() {
        let ddddocr = ddddocr_detection().unwrap();
        let input = read_image("image/5.jpg");
        let result = ddddocr.detection(input.clone()).unwrap();

        println!("{:?}", result);

        // 绘制红框
        let mut image = load_image(&input).unwrap().to_rgb8();

        for v in result {
            for i in v.x1..=v.x2 {
                image[(i, v.y1)] = *image::Rgb::from_slice(&[237, 28, 36]);
            }

            for i in v.x1..=v.x2 {
                image[(i, v.y2)] = *image::Rgb::from_slice(&[237, 28, 36]);
            }

            for i in v.y1..=v.y2 {
                image[(v.x1, i)] = *image::Rgb::from_slice(&[237, 28, 36]);
            }

            for i in v.y1..=v.y2 {
                image[(v.x2, i)] = *image::Rgb::from_slice(&[237, 28, 36]);
            }
        }

        image.save("./output1.jpg").unwrap();

        let input = read_image("image/6.jpg");
        let result = ddddocr.detection(input.clone()).unwrap();

        println!("{:?}", result);

        // 绘制红框
        let mut image = load_image(&input).unwrap().to_rgb8();

        for v in result {
            for i in v.x1..=v.x2 {
                image[(i, v.y1)] = *image::Rgb::from_slice(&[237, 28, 36]);
            }

            for i in v.x1..=v.x2 {
                image[(i, v.y2)] = *image::Rgb::from_slice(&[237, 28, 36]);
            }

            for i in v.y1..=v.y2 {
                image[(v.x1, i)] = *image::Rgb::from_slice(&[237, 28, 36]);
            }

            for i in v.y1..=v.y2 {
                image[(v.x2, i)] = *image::Rgb::from_slice(&[237, 28, 36]);
            }
        }

        image.save("./output2.jpg").unwrap();
    }

    #[test]
    fn slide_match() {
        let result =
            crate::slide_match(read_image("image/hk.png"), read_image("image/bg.png")).unwrap();

        println!("{:?}", result);

        let result =
            crate::slide_match(read_image("image/a.png"), read_image("image/b.png")).unwrap();

        println!("{:?}", result);

        let result =
            crate::simple_slide_match(read_image("image/a.png"), read_image("image/b.png"))
                .unwrap();

        println!("{:?}", result);
    }

    #[test]
    fn comparison_match() {
        let result =
            crate::slide_comparison(read_image("image/c.jpg"), read_image("image/d.jpg")).unwrap();
        println!("{:?}", result);
    }

    #[test]
    fn info() {
        println!(
            "current_dir: {}",
            std::env::current_dir().unwrap().display()
        );

        #[cfg(not(feature = "inline-model"))]
        println!("not feature inline-model");
    }
}

#[cfg(test)]
mod regression_tests {
    use super::*;

    fn png(image: image::DynamicImage) -> Vec<u8> {
        let mut bytes = std::io::Cursor::new(Vec::new());
        image.write_to(&mut bytes, image::ImageFormat::Png).unwrap();
        bytes.into_inner()
    }

    #[test]
    fn ctc_keeps_repetitions_separated_by_blank() {
        let charset = vec!["".into(), "a".into(), "b".into()];
        assert_eq!(decode_ctc([1, 1, 0, 1, 2, 2, 0, 2], &charset), "aabb");
        assert_eq!(decode_ctc([0, 0], &charset), "");
        assert_eq!(decode_ctc([99], &charset), "");
    }

    #[test]
    fn probability_text_uses_ctc_and_handles_empty_rows() {
        let mut value = CharacterProbability {
            text: None,
            charset: vec!["".into(), "a".into()],
            probability: vec![
                vec![0.0, 1.0],
                vec![0.0, 1.0],
                vec![1.0, 0.0],
                vec![0.0, 1.0],
                vec![],
            ],
            confidence: None,
        };
        assert_eq!(value.get_text(), "aa");
    }

    #[test]
    fn softmax_is_stable_for_large_logits() {
        let probabilities = softmax(&[10000.0, 10001.0, -10000.0]);
        assert!(probabilities.iter().all(|p| p.is_finite()));
        assert!((probabilities.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(probabilities[1] > probabilities[0]);
        let mut logits = vec![-18.0; 8210];
        logits[0] = 0.0;
        let probabilities = softmax(&logits);
        assert!((probabilities.iter().map(|&p| f64::from(p)).sum::<f64>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn alpha_is_composited_before_grayscale() {
        let input = image::DynamicImage::ImageRgba8(image::RgbaImage::from_fn(3, 1, |x, _| {
            image::Rgba([0, 0, 0, [0, 128, 255][x as usize]])
        }));
        let result = png_rgba_black_preprocess(&input).to_luma8();
        assert_eq!(result.into_raw(), [255, 127, 0]);
    }

    #[test]
    fn comparison_accepts_grayscale_and_rgba_and_top_row_changes() {
        let target = png(image::DynamicImage::ImageLuma8(
            image::GrayImage::from_pixel(8, 8, image::Luma([0])),
        ));
        let background = png(image::DynamicImage::ImageRgba8(
            image::RgbaImage::from_pixel(8, 8, image::Rgba([255, 255, 255, 255])),
        ));
        assert_eq!(slide_comparison(target, background).unwrap(), (2, 0));
    }

    #[test]
    fn transparent_slider_is_rejected() {
        let image = png(image::DynamicImage::new_rgba8(8, 8));
        assert!(slide_match(&image, &image)
            .unwrap_err()
            .to_string()
            .contains("transparent"));
    }

    #[test]
    fn decode_rejects_oversized_and_invalid_images() {
        assert!(load_image(&[]).is_err());
        assert!(load_image(&vec![0; MAX_IMAGE_BYTES + 1]).is_err());
        let image = png(image::DynamicImage::new_luma8(MAX_IMAGE_SIDE + 1, 1));
        assert!(load_image(&image).is_err());
    }

    #[test]
    fn template_work_is_bounded() {
        assert!(check_match_work(
            &image::GrayImage::new(300, 300),
            &image::GrayImage::new(600, 600)
        )
        .is_err());
    }

    #[test]
    fn shared_model_is_thread_safe_and_ranges_are_deterministic() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Ddddocr<'static>>();
        let model = std::sync::Arc::new(ddddocr_classification().unwrap());
        assert_eq!(model.calc_ranges("aaba"), ["a", "b", ""]);
        let expected = model
            .classification(include_bytes!("../image/1.png"))
            .unwrap();
        let handles: Vec<_> = (0..8)
            .map(|_| {
                let model = model.clone();
                std::thread::spawn(move || {
                    model
                        .classification(include_bytes!("../image/1.png"))
                        .unwrap()
                })
            })
            .collect();
        for handle in handles {
            assert_eq!(handle.join().unwrap(), expected);
        }
    }

    #[test]
    fn bbox_rejects_invalid_coordinates_and_matches_whole_image() {
        let model = ddddocr_classification().unwrap();
        let bytes = include_bytes!("../image/1.png");
        let image = load_image(bytes).unwrap();
        let bbox = BBox {
            x1: 0,
            y1: 0,
            x2: image.width(),
            y2: image.height(),
        };
        assert_eq!(
            model.classification_bbox(bytes, &vec![bbox]).unwrap()[0].1,
            model.classification(bytes).unwrap()
        );
        assert!(model
            .classification_bbox(
                bytes,
                &vec![BBox {
                    x1: 2,
                    y1: 0,
                    x2: 1,
                    y2: 1
                }]
            )
            .is_err());
    }

    #[test]
    fn rgb_tensor_uses_interleaved_pixels_and_png_fix_works_for_gray_models() {
        let mut model = ddddocr_classification().unwrap();
        let image = png(image::DynamicImage::ImageRgb8(image::RgbImage::from_fn(
            2,
            1,
            |x, _| {
                if x == 0 {
                    image::Rgb([255, 0, 0])
                } else {
                    image::Rgb([0, 0, 255])
                }
            },
        )));
        let charset = model.charset.as_mut().unwrap().to_mut();
        charset.channel = 3;
        charset.image = [2, 1];
        let tensor = model.prepare_tensor(&image, false, None).unwrap();
        assert!((tensor[[0, 0, 0, 0]] - (1.0 - 0.485) / 0.229).abs() < 1e-5);
        assert!((tensor[[0, 0, 0, 1]] - (0.0 - 0.485) / 0.229).abs() < 1e-5);
        assert!((tensor[[0, 2, 0, 1]] - (1.0 - 0.406) / 0.225).abs() < 1e-5);
        let charset = model.charset.as_mut().unwrap().to_mut();
        charset.channel = 1;
        charset.image = [1, 1];
        let transparent = png(image::DynamicImage::new_rgba8(1, 1));
        assert_eq!(
            model.prepare_tensor(&transparent, true, None).unwrap()[[0, 0, 0, 0]],
            1.0
        );
    }
}
