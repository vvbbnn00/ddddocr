use crate::{Args, ARGS};
use base64::prelude::*;
use ddddocr::*;
use salvo::{catcher::Catcher, prelude::*};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    sync::{Arc, OnceLock, RwLock},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

static OCR: [OnceLock<std::result::Result<Ddddocr<'static>, String>>; 2] =
    [OnceLock::new(), OnceLock::new()];
static DET: OnceLock<Ddddocr<'static>> = OnceLock::new();
static RANGES: [RwLock<Option<CharsetRange>>; 2] = [RwLock::new(None), RwLock::new(None)];
static ADMISSION: OnceLock<Arc<Semaphore>> = OnceLock::new();
static STARTED: OnceLock<Instant> = OnceLock::new();

#[derive(Debug)]
struct ApiError(StatusCode, String);
type Result<T> = std::result::Result<T, ApiError>;
impl ApiError {
    fn bad(error: impl std::fmt::Display) -> Self {
        Self(StatusCode::BAD_REQUEST, error.to_string())
    }
    fn internal(error: impl std::fmt::Display) -> Self {
        Self(StatusCode::INTERNAL_SERVER_ERROR, error.to_string())
    }
    fn render(self, res: &mut Response) {
        res.status_code(self.0);
        res.render(Json(json!({"detail": self.1})));
    }
}

pub fn initialize(args: &Args) -> anyhow::Result<()> {
    ADMISSION
        .set(Arc::new(Semaphore::new(args.max_concurrency as usize)))
        .unwrap();
    STARTED.set(Instant::now()).unwrap();
    if let Some(range) = &args.ocr_charset_range {
        let range = charset(&json!(range)).map_err(|e| anyhow::anyhow!(e.1))?;
        for slot in &RANGES {
            *slot.write().unwrap() = Some(range.clone());
        }
    }
    if args.ocr || args.old {
        model(args.ocr).map_err(|e| anyhow::anyhow!(e.1))?;
    }
    if args.det {
        let bytes = model_bytes("common_det.onnx", args.det_path.as_deref())?;
        DET.set(Ddddocr::new_model_with_threads(
            bytes.as_ref(),
            args.inference_threads as usize,
        )?)
        .unwrap();
    }
    Ok(())
}

fn model_bytes(
    name: &str,
    custom: Option<&std::path::Path>,
) -> anyhow::Result<std::borrow::Cow<'static, [u8]>> {
    if let Some(path) = custom {
        return Ok(std::fs::read(path)?.into());
    }
    #[cfg(feature = "inline-model")]
    {
        Ok(std::borrow::Cow::Borrowed(match name {
            "common.onnx" => include_bytes!("../model/common.onnx"),
            "common_old.onnx" => include_bytes!("../model/common_old.onnx"),
            "common_det.onnx" => include_bytes!("../model/common_det.onnx"),
            _ => anyhow::bail!("unknown model"),
        }))
    }
    #[cfg(not(feature = "inline-model"))]
    {
        Ok(std::fs::read(std::path::Path::new("model").join(name))?.into())
    }
}

fn model_index(beta: bool) -> usize {
    let args = ARGS.get().unwrap();
    usize::from(if args.ocr_path.is_some() {
        args.ocr
    } else {
        beta
    })
}

fn model(beta: bool) -> Result<&'static Ddddocr<'static>> {
    let args = ARGS.get().unwrap();
    if !args.ocr && !args.old {
        return Err(ApiError(
            StatusCode::SERVICE_UNAVAILABLE,
            "OCR is disabled".into(),
        ));
    }
    let index = model_index(beta);
    OCR[index]
        .get_or_init(|| {
            let build = || -> anyhow::Result<Ddddocr<'static>> {
                let bytes = model_bytes(
                    if beta {
                        "common.onnx"
                    } else {
                        "common_old.onnx"
                    },
                    args.ocr_path.as_deref(),
                )?;
                let charset: Charset = if let Some(path) = &args.ocr_path {
                    serde_json::from_slice(&std::fs::read(path.with_extension("json"))?)?
                } else {
                    serde_json::from_str(if beta {
                        include_str!("../model/common.json")
                    } else {
                        include_str!("../model/common_old.json")
                    })?
                };
                Ddddocr::new_with_threads(bytes.as_ref(), charset, args.inference_threads as usize)
            };
            build().map_err(|e| e.to_string())
        })
        .as_ref()
        .map_err(ApiError::internal)
}

pub fn service(args: &Args) -> Service {
    // Admission precedes all parsing, including multipart and MCP requests.
    let mut router = Router::new()
        .hoop(admit)
        .push(Router::with_path("health").get(health))
        .push(Router::with_path("status").get(status))
        .push(Router::with_path("config").get(config));
    if !args.only_mcp {
        if args.ocr || args.old {
            for path in ["ocr", "ocr/file", "ocr/binary"] {
                router = router.push(Router::with_path(path).post(ocr));
            }
            router = router.push(Router::with_path("set_charset_range").post(set_charset));
        }
        if args.det {
            for path in ["det", "det/file", "det/binary"] {
                router = router.push(Router::with_path(path).post(det));
            }
        }
        if args.slide {
            for path in ["slide_match", "slide_match/file"] {
                router = router.push(Router::with_path(path).post(slide_match_route));
            }
            for path in ["slide_comparison", "slide_comparison/file"] {
                router = router.push(Router::with_path(path).post(slide_comparison_route));
            }
        }
    }
    if args.mcp || args.only_mcp {
        router = router.push(Router::with_path("mcp").post(mcp));
    }
    router = router
        .push(Router::with_path("docs.json").get(docs))
        .push(SwaggerUi::new("/docs.json").into_router("docs"));
    Service::new(router).catcher(Catcher::default().hoop(catch_error))
}

#[handler]
async fn admit(req: &mut Request, depot: &mut Depot, res: &mut Response, ctrl: &mut FlowCtrl) {
    if req.method() != salvo::http::Method::POST {
        return;
    }
    let args = ARGS.get().unwrap();
    let limit = usize::from(args.max_body_mib) * 1024 * 1024;
    req.set_secure_max_size(limit);
    if req
        .headers()
        .get("content-length")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<u64>().ok())
        .is_some_and(|n| n > limit as u64)
    {
        ApiError(
            StatusCode::PAYLOAD_TOO_LARGE,
            "request body exceeds limit".into(),
        )
        .render(res);
        ctrl.skip_rest();
        return;
    }
    match ADMISSION.get().unwrap().clone().try_acquire_owned() {
        Ok(permit) => {
            depot.insert("permit", Arc::new(permit));
        }
        Err(_) => {
            res.add_header("Retry-After", "1", true).ok();
            ApiError(
                StatusCode::SERVICE_UNAVAILABLE,
                "server busy; retry later".into(),
            )
            .render(res);
            ctrl.skip_rest();
        }
    }
}

#[handler]
fn catch_error(res: &mut Response) {
    // Preserve explicit JSON errors (including 413 and 503).
    if matches!(res.body, salvo::http::ResBody::Error(_))
        || matches!(res.body, salvo::http::ResBody::None)
    {
        let code = res.status_code.unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
        ApiError(
            code,
            code.canonical_reason().unwrap_or("request failed").into(),
        )
        .render(res);
    }
}

async fn body(req: &mut Request) -> Result<bytes::Bytes> {
    tokio::time::timeout(Duration::from_secs(u64::from(ARGS.get().unwrap().upload_timeout_secs)), req.payload()).await
        .map_err(|_| ApiError(StatusCode::REQUEST_TIMEOUT, "upload timed out".into()))?
        .cloned()
        .map_err(|e| {
            if matches!(&e, salvo::http::ParseError::Other(error) if error.is::<http_body_util::LengthLimitError>()) {
                ApiError(StatusCode::PAYLOAD_TOO_LARGE, "request body exceeds limit".into())
            } else { ApiError::bad(e) }
        })
}

#[derive(Default, Deserialize)]
struct Options {
    #[serde(default)]
    png_fix: bool,
    #[serde(default)]
    probability: bool,
    charset_range: Option<Value>,
    color_filter: Option<Value>,
    #[serde(default, alias = "color_filter_colors")]
    colors: Vec<String>,
    custom_color_ranges: Option<std::collections::HashMap<String, [[u8; 3]; 2]>>,
    color_filter_custom_ranges: Option<Vec<[[u8; 3]; 2]>>,
    #[serde(default)]
    simple_target: bool,
    #[serde(default)]
    flag: bool,
}

struct Input {
    images: Vec<bytes::Bytes>,
    options: Options,
    beta: bool,
}

fn bool_value(value: &str) -> Result<bool> {
    match value.to_ascii_lowercase().as_str() {
        "true" | "1" | "yes" | "y" => Ok(true),
        "false" | "0" | "no" | "n" => Ok(false),
        _ => Err(ApiError::bad("invalid boolean parameter")),
    }
}

fn select_model(req: &Request) -> Result<bool> {
    if req
        .query::<String>("use_gpu")
        .map(|v| bool_value(&v))
        .transpose()?
        .unwrap_or(false)
    {
        return Err(ApiError::bad(
            "this HTTP server uses CPU inference; use_gpu=true is unsupported",
        ));
    }
    // Validate upstream configuration flags instead of silently accepting malformed values.
    for key in ["ocr", "det", "old", "show_ad"] {
        if let Some(value) = req.query::<String>(key) {
            bool_value(&value)?;
        }
    }
    if let Some(value) = req.query::<String>("device_id") {
        value.parse::<u32>().map_err(ApiError::bad)?;
    }
    if req
        .query::<String>("old")
        .map(|v| bool_value(&v))
        .transpose()?
        .unwrap_or(false)
    {
        return Ok(false);
    }
    req.query::<String>("beta")
        .map(|value| bool_value(&value))
        .transpose()
        .map(|v| v.unwrap_or(ARGS.get().unwrap().ocr))
}

fn decode_base64(value: &str) -> Result<bytes::Bytes> {
    let value = if value.starts_with("data:") {
        let (prefix, data) = value
            .split_once(',')
            .ok_or_else(|| ApiError::bad("invalid data URI"))?;
        if !prefix.ends_with(";base64") {
            return Err(ApiError::bad("data URI must use base64"));
        }
        data
    } else {
        value
    };
    if value.len() > MAX_IMAGE_BYTES.div_ceil(3) * 4 {
        return Err(ApiError(
            StatusCode::PAYLOAD_TOO_LARGE,
            "image exceeds 8 MiB".into(),
        ));
    }
    let decoded = BASE64_STANDARD.decode(value).map_err(ApiError::bad)?;
    check_image_size(&decoded)?;
    Ok(decoded.into())
}

fn check_image_size(image: &[u8]) -> Result<()> {
    if image.len() > MAX_IMAGE_BYTES {
        return Err(ApiError(
            StatusCode::PAYLOAD_TOO_LARGE,
            "image exceeds 8 MiB".into(),
        ));
    }
    if image.is_empty() {
        return Err(ApiError::bad("empty image"));
    }
    Ok(())
}

fn json_input(value: Value, pair: bool, beta: bool) -> Result<Input> {
    let fields: &[&str] = if pair {
        &["target_image", "background_image"]
    } else {
        &["image"]
    };
    let images = fields
        .iter()
        .map(|field| {
            value
                .get(field)
                .and_then(Value::as_str)
                .ok_or_else(|| ApiError::bad(format!("missing {field}")))
                .and_then(decode_base64)
        })
        .collect::<Result<Vec<_>>>()?;
    let options = serde_json::from_value(value).map_err(ApiError::bad)?;
    Ok(Input {
        images,
        options,
        beta,
    })
}

async fn input(req: &mut Request, pair: bool, permit: Arc<OwnedSemaphorePermit>) -> Result<Input> {
    let beta = select_model(req)?;
    let content_type = req
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_owned();
    let data = body(req).await?;
    let media_type = content_type
        .split(';')
        .next()
        .unwrap_or("")
        .trim()
        .to_ascii_lowercase();
    if media_type == "application/json" {
        return tokio::task::spawn_blocking(move || {
            let _permit = permit;
            json_input(
                serde_json::from_slice(&data).map_err(ApiError::bad)?,
                pair,
                beta,
            )
        })
        .await
        .map_err(ApiError::internal)?;
    }
    if media_type == "multipart/form-data" {
        let boundary = multer::parse_boundary(&content_type).map_err(ApiError::bad)?;
        // Parse a size-limited memory body, not Salvo's unlimited temporary-file parser.
        let stream = futures_util::stream::once(async move { Ok::<_, std::io::Error>(data) });
        let mut multipart = multer::Multipart::new(stream, boundary);
        let mut fields = serde_json::Map::new();
        let mut images = std::collections::HashMap::new();
        let names: &[&str] = if pair {
            &["target_image", "background_image"]
        } else {
            &["file", "image"]
        };
        while let Some(field) = multipart.next_field().await.map_err(ApiError::bad)? {
            if fields.len() + images.len() >= 32 {
                return Err(ApiError::bad("too many multipart fields"));
            }
            let name = field
                .name()
                .ok_or_else(|| ApiError::bad("multipart field needs a name"))?
                .to_owned();
            let value = field.bytes().await.map_err(ApiError::bad)?;
            if names.contains(&name.as_str()) {
                check_image_size(&value)?;
                if images.insert(name, value).is_some() {
                    return Err(ApiError::bad("duplicate image field"));
                }
            } else {
                if value.len() > 16 * 1024 {
                    return Err(ApiError::bad("option exceeds 16 KiB"));
                }
                let text = std::str::from_utf8(&value).map_err(ApiError::bad)?;
                let value = match name.as_str() {
                    "png_fix" | "probability" | "simple_target" | "flag" => {
                        json!(bool_value(text)?)
                    }
                    "charset_range" | "color_filter" => {
                        serde_json::from_str(text).unwrap_or_else(|_| json!(text))
                    }
                    _ => serde_json::from_str(text).map_err(ApiError::bad)?,
                };
                if fields.insert(name, value).is_some() {
                    return Err(ApiError::bad("duplicate option"));
                }
            }
        }
        let images = if pair {
            vec![
                images
                    .remove("target_image")
                    .ok_or_else(|| ApiError::bad("missing target_image"))?,
                images
                    .remove("background_image")
                    .ok_or_else(|| ApiError::bad("missing background_image"))?,
            ]
        } else {
            if images.len() != 1 {
                return Err(ApiError::bad("provide exactly one file or image field"));
            }
            vec![images
                .remove("file")
                .or_else(|| images.remove("image"))
                .ok_or_else(|| ApiError::bad("missing file"))?]
        };
        return Ok(Input {
            images,
            options: serde_json::from_value(Value::Object(fields)).map_err(ApiError::bad)?,
            beta,
        });
    }
    if !pair && (media_type.starts_with("image/") || media_type == "application/octet-stream") {
        check_image_size(&data)?;
        let mut options = Options::default();
        for key in ["png_fix", "probability"] {
            if let Some(value) = req.query::<String>(key) {
                let flag = bool_value(&value)?;
                if key == "png_fix" {
                    options.png_fix = flag;
                } else {
                    options.probability = flag;
                }
            }
        }
        options.charset_range = req
            .query::<String>("charset_range")
            .map(|v| serde_json::from_str(&v).unwrap_or_else(|_| json!(v)));
        options.color_filter = req
            .query::<String>("color_filter")
            .map(|v| serde_json::from_str(&v).unwrap_or_else(|_| json!(v)));
        options.colors = req
            .query::<String>("colors")
            .or_else(|| req.query::<String>("color_filter_colors"))
            .map(|v| serde_json::from_str(&v).map_err(ApiError::bad))
            .transpose()?
            .unwrap_or_default();
        options.custom_color_ranges = req
            .query::<String>("custom_color_ranges")
            .map(|v| serde_json::from_str(&v).map_err(ApiError::bad))
            .transpose()?;
        options.color_filter_custom_ranges = req
            .query::<String>("color_filter_custom_ranges")
            .map(|v| serde_json::from_str(&v).map_err(ApiError::bad))
            .transpose()?;
        return Ok(Input {
            images: vec![data],
            options,
            beta,
        });
    }
    Err(ApiError(
        StatusCode::UNSUPPORTED_MEDIA_TYPE,
        "use application/json, multipart/form-data, or image bytes".into(),
    ))
}

fn charset(value: &Value) -> Result<CharsetRange> {
    match value {
        Value::Number(n) => n
            .as_i64()
            .filter(|v| (0..=7).contains(v))
            .map(|v| CharsetRange::from(v as i32))
            .ok_or_else(|| ApiError::bad("charset_range must be 0..7")),
        Value::String(s) if s.len() <= 16 * 1024 => Ok(
            if matches!(s.as_str(), "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7") {
                CharsetRange::from(s.parse::<i32>().unwrap())
            } else {
                CharsetRange::from(s.as_str())
            },
        ),
        Value::Array(_) => {
            let values: Vec<String> =
                serde_json::from_value(value.clone()).map_err(ApiError::bad)?;
            if values.iter().map(String::len).sum::<usize>() > 16 * 1024 {
                return Err(ApiError::bad("charset_range exceeds 16 KiB"));
            }
            Ok(CharsetRange::Other(values.concat()))
        }
        _ => Err(ApiError::bad("invalid charset_range")),
    }
}

fn filter(options: &Options) -> Result<Option<ColorFilter>> {
    if options.colors.len() > 32
        || options
            .color_filter_custom_ranges
            .as_ref()
            .is_some_and(|v| v.len() > 32)
        || options
            .custom_color_ranges
            .as_ref()
            .is_some_and(|v| v.len() > 32)
    {
        return Err(ApiError::bad("at most 32 color filters allowed"));
    }
    if let Some(value) = &options.color_filter {
        let filter: ColorFilter = serde_json::from_value(value.clone()).map_err(ApiError::bad)?;
        let ranges = filter.clone().into_hsv_ranges();
        validate_ranges(&ranges)?;
        return Ok(Some(filter));
    }
    let mut ranges = Vec::new();
    for name in &options.colors {
        if let Some(custom) = options
            .custom_color_ranges
            .as_ref()
            .and_then(|ranges| ranges.get(name))
        {
            ranges.push((
                (custom[0][0], custom[0][1], custom[0][2]),
                (custom[1][0], custom[1][1], custom[1][2]),
            ));
        } else {
            let color: Color = serde_json::from_value(json!(name)).map_err(ApiError::bad)?;
            ranges.extend(color.into_hsv_ranges());
        }
    }
    if let Some(custom) = &options.color_filter_custom_ranges {
        for range in custom {
            ranges.push((
                (range[0][0], range[0][1], range[0][2]),
                (range[1][0], range[1][1], range[1][2]),
            ));
        }
    }
    validate_ranges(&ranges)?;
    Ok((!ranges.is_empty()).then_some(ColorFilter::HSVRanges(ranges)))
}

fn validate_ranges(ranges: &[HsvRange]) -> Result<()> {
    if ranges.len() > 64
        || ranges
            .iter()
            .any(|(lo, hi)| lo.0 > hi.0 || lo.1 > hi.1 || lo.2 > hi.2 || hi.0 > 180)
    {
        return Err(ApiError::bad("invalid HSV range or too many ranges"));
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum Operation {
    Ocr,
    Det,
    Match,
    Comparison,
}

fn restrict_text(text: String, range: Option<CharsetRange>, model: &Ddddocr<'_>) -> String {
    if let Some(range) = range {
        let allowed: std::collections::HashSet<char> =
            model.calc_ranges(range).concat().chars().collect();
        text.chars().filter(|c| allowed.contains(c)).collect()
    } else {
        text
    }
}

#[derive(Serialize)]
#[serde(untagged)]
enum ResponseData {
    Probability {
        text: String,
        probabilities: Vec<Vec<Vec<f32>>>,
        charset: Vec<String>,
        confidence: f64,
    },
    Other(Value),
}

#[derive(Serialize)]
struct InferenceResponse {
    result: ResponseData,
    processing_time: f64,
}

fn compute(operation: Operation, input: Input) -> Result<InferenceResponse> {
    let started = Instant::now();
    let args = ARGS.get().unwrap();
    let Input {
        images,
        options,
        beta,
    } = input;
    let result = match operation {
        Operation::Ocr => {
            let model = model(beta)?;
            let global = RANGES[model_index(beta)]
                .read()
                .map_err(ApiError::internal)?
                .clone();
            let range = options
                .charset_range
                .as_ref()
                .map(charset)
                .transpose()?
                .or(global);
            let filter = filter(&options)?;
            if options.probability {
                let mut probability = model
                    .classification_probability_with_options(
                        &images[0],
                        options.png_fix,
                        filter,
                        None,
                    )
                    .map_err(ApiError::bad)?;
                let text = probability.get_text().to_owned();
                let text = restrict_text(text, range, model);
                let confidence = probability.get_confidence();
                ResponseData::Probability {
                    text,
                    probabilities: probability
                        .probability
                        .into_iter()
                        .map(|row| vec![row])
                        .collect(),
                    charset: probability.charset,
                    confidence,
                }
            } else {
                let text = model
                    .classification_with_options(&images[0], options.png_fix, filter)
                    .map_err(ApiError::bad)?;
                let text = restrict_text(text, range, model);
                ResponseData::Other(json!(text))
            }
        }
        Operation::Det => {
            let model = DET
                .get()
                .ok_or_else(|| ApiError::bad("detection is disabled"))?;
            let boxes = model.detection(&images[0]).map_err(ApiError::bad)?.to_vec();
            ResponseData::Other(json!(boxes))
        }
        Operation::Match => {
            if !args.slide {
                return Err(ApiError::bad("slide is disabled"));
            }
            if options.flag {
                return Err(ApiError::bad("flag=true is not supported"));
            }
            let bbox = if options.simple_target {
                simple_slide_match(&images[0], &images[1])
            } else {
                slide_match(&images[0], &images[1])
            }
            .map_err(ApiError::bad)?;
            ResponseData::Other(
                json!({"target": [bbox.x1,bbox.y1,bbox.x2,bbox.y2], "target_x": bbox.target_x, "target_y": bbox.target_y}),
            )
        }
        Operation::Comparison => {
            if !args.slide {
                return Err(ApiError::bad("slide is disabled"));
            }
            let (x, y) = slide_comparison(&images[0], &images[1]).map_err(ApiError::bad)?;
            ResponseData::Other(json!({"target":[x,y]}))
        }
    };
    Ok(InferenceResponse {
        result,
        processing_time: started.elapsed().as_secs_f64(),
    })
}

fn send_json(res: &mut Response, bytes: Vec<u8>, depot: &Depot) {
    let permit = depot
        .get::<Arc<OwnedSemaphorePermit>>("permit")
        .unwrap()
        .clone();
    // Retain admission while a slow reader consumes a large probability response.
    // Small chunks let HTTP backpressure stop polling without buffering the whole body downstream.
    let chunks = futures_util::stream::unfold(
        (bytes::Bytes::from(bytes), permit),
        |(mut data, permit)| async move {
            if data.is_empty() {
                None
            } else {
                let chunk = data.split_to(data.len().min(64 * 1024));
                Some((Ok::<_, std::convert::Infallible>(chunk), (data, permit)))
            }
        },
    );
    res.add_header("Content-Type", "application/json", true)
        .ok();
    res.stream(chunks);
}

async fn run(operation: Operation, req: &mut Request, depot: &Depot, res: &mut Response) {
    let result = async {
        let permit = depot
            .get::<Arc<OwnedSemaphorePermit>>("permit")
            .map_err(|_| ApiError::internal("missing admission permit"))?
            .clone();
        if matches!(operation, Operation::Ocr) {
            if req
                .query::<String>("det")
                .map(|v| bool_value(&v))
                .transpose()?
                .unwrap_or(false)
                || !req
                    .query::<String>("ocr")
                    .map(|v| bool_value(&v))
                    .transpose()?
                    .unwrap_or(true)
            {
                return Err(ApiError::bad("request configuration disables OCR"));
            }
        } else if matches!(operation, Operation::Det)
            && !req
                .query::<String>("det")
                .map(|v| bool_value(&v))
                .transpose()?
                .unwrap_or(true)
        {
            return Err(ApiError::bad("request configuration disables detection"));
        }
        let input = input(
            req,
            matches!(operation, Operation::Match | Operation::Comparison),
            permit.clone(),
        )
        .await?;
        tokio::task::spawn_blocking(move || {
            // Keep admission until inference AND JSON serialization finish, even after client disconnect.
            let _permit = permit;
            let value = compute(operation, input)?;
            serde_json::to_vec(&value).map_err(ApiError::internal)
        })
        .await
        .map_err(ApiError::internal)?
    }
    .await;
    match result {
        Ok(bytes) => send_json(res, bytes, depot),
        Err(error) => error.render(res),
    }
}

#[handler]
async fn ocr(req: &mut Request, depot: &mut Depot, res: &mut Response) {
    run(Operation::Ocr, req, depot, res).await;
}
#[handler]
async fn det(req: &mut Request, depot: &mut Depot, res: &mut Response) {
    run(Operation::Det, req, depot, res).await;
}
#[handler]
async fn slide_match_route(req: &mut Request, depot: &mut Depot, res: &mut Response) {
    run(Operation::Match, req, depot, res).await;
}
#[handler]
async fn slide_comparison_route(req: &mut Request, depot: &mut Depot, res: &mut Response) {
    run(Operation::Comparison, req, depot, res).await;
}

#[handler]
fn health(res: &mut Response) {
    res.render(Json(json!({"status":"ok", "timestamp":SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs_f64()})));
}

#[handler]
fn status(res: &mut Response) {
    let args = ARGS.get().unwrap();
    let mut features = Vec::new();
    if args.ocr || args.old {
        features.push("ocr");
    }
    if args.det {
        features.push("det");
    }
    if args.slide {
        features.push("slide");
    }
    if args.mcp || args.only_mcp {
        features.push("mcp");
    }
    let value = json!({"service_status":"running", "enabled_features":features, "uptime":STARTED.get().unwrap().elapsed().as_secs_f64(),
        "max_concurrency":args.max_concurrency, "active_requests":usize::from(args.max_concurrency)-ADMISSION.get().unwrap().available_permits()});
    res.render(Json(value));
}

#[handler]
fn config(res: &mut Response) {
    let args = ARGS.get().unwrap();
    res.render(Json(json!({"default_config":{"ocr":args.ocr || args.old,"det":false,"old":args.old,"beta":args.ocr,"use_gpu":false,"device_id":0,"show_ad":false,
        "import_onnx_path":args.ocr_path.as_ref().map(|p| p.to_string_lossy()).unwrap_or_default(),
        "charsets_path":args.ocr_path.as_ref().map(|p| p.with_extension("json").to_string_lossy().into_owned()).unwrap_or_default()},
        "active_instances":OCR.iter().filter(|cell| cell.get().is_some_and(|v| v.is_ok())).count() + usize::from(DET.get().is_some()),
        "limits":{"max_concurrency":args.max_concurrency,"inference_threads":args.inference_threads,"max_image_bytes":MAX_IMAGE_BYTES,"max_image_side":MAX_IMAGE_SIDE},
        "environment":{"runtime":"rust","version":env!("CARGO_PKG_VERSION")}})));
}

#[handler]
async fn set_charset(req: &mut Request, res: &mut Response) {
    let started = Instant::now();
    let result = async {
        let beta = select_model(req)?;
        let data: Value = serde_json::from_slice(&body(req).await?).map_err(ApiError::bad)?;
        let value = data.get("charset_range").ok_or_else(|| ApiError::bad("missing charset_range"))?;
        let range = charset(value)?;
        *RANGES[model_index(beta)].write().map_err(ApiError::internal)? = if value.as_array().is_some_and(Vec::is_empty) { None } else { Some(range) };
        Ok(json!({"result":"字符范围设置成功","charset_range":value,"processing_time":started.elapsed().as_secs_f64()}))
    }.await;
    match result {
        Ok(value) => res.render(Json(value)),
        Err(error) => ApiError::render(error, res),
    }
}

#[handler]
fn docs(res: &mut Response) {
    res.render(Json(crate::schema::document()));
}

#[handler]
async fn mcp(req: &mut Request, depot: &mut Depot, res: &mut Response) {
    let data = match body(req).await {
        Ok(data) => data,
        Err(error) => {
            error.render(res);
            return;
        }
    };
    let permit = depot
        .get::<Arc<OwnedSemaphorePermit>>("permit")
        .unwrap()
        .clone();
    let value: Value = match tokio::task::spawn_blocking(move || {
        let _permit = permit;
        serde_json::from_slice::<Value>(&data)
    })
    .await
    {
        Ok(Ok(value)) => value,
        Ok(Err(_)) => {
            res.render(Json(
                json!({"jsonrpc":"2.0","id":null,"error":{"code":-32700,"message":"parse error"}}),
            ));
            return;
        }
        Err(error) => {
            ApiError::internal(error).render(res);
            return;
        }
    };
    if value["jsonrpc"] != "2.0"
        || !value["method"].is_string()
        || value
            .get("id")
            .is_some_and(|id| !(id.is_null() || id.is_string() || id.is_i64() || id.is_u64()))
    {
        res.render(Json(
            json!({"jsonrpc":"2.0","id":null,"error":{"code":-32600,"message":"invalid request"}}),
        ));
        return;
    }
    let id = value.get("id").cloned();
    if id.is_none()
        && value
            .get("method")
            .and_then(Value::as_str)
            .is_some_and(|s| s.starts_with("notifications/"))
    {
        res.status_code(StatusCode::ACCEPTED);
        return;
    }
    let id = id.unwrap_or(Value::Null);
    let method = value.get("method").and_then(Value::as_str).unwrap_or("");
    let result = match method {
        "initialize" => {
            Ok(serde_json::from_str::<Value>(include_str!("../initialize.json")).unwrap())
        }
        "ping" => Ok(json!({})),
        "tools/list" => {
            let mut tools = crate::schema::tools().clone();
            let args = ARGS.get().unwrap();
            tools["tools"]
                .as_array_mut()
                .unwrap()
                .retain(|tool| match tool["name"].as_str() {
                    Some("ocr") => args.ocr || args.old,
                    Some("det") => args.det,
                    _ => args.slide,
                });
            Ok(tools)
        }
        "tools/call" => {
            let operation = match value["params"]["name"].as_str() {
                Some("ocr") => Some(Operation::Ocr),
                Some("det") => Some(Operation::Det),
                Some("slide_match") => Some(Operation::Match),
                Some("slide_comparison") => Some(Operation::Comparison),
                _ => None,
            };
            if let Some(operation) = operation {
                let arguments = value["params"]["arguments"].clone();
                let permit = depot
                    .get::<Arc<OwnedSemaphorePermit>>("permit")
                    .unwrap()
                    .clone();
                let output = tokio::task::spawn_blocking(move || -> Result<Vec<u8>> {
                    let _permit = permit;
                    let computed = json_input(arguments, matches!(operation, Operation::Match | Operation::Comparison), ARGS.get().unwrap().ocr)
                        .and_then(|input| compute(operation, input));
                    let result = match computed {
                        Ok(output) => {
                            let structured = serde_json::to_value(output).map_err(ApiError::internal)?;
                            let text = serde_json::to_string(&structured).map_err(ApiError::internal)?;
                            json!({"content":[{"type":"text","text":text}],"structuredContent":structured,"isError":false})
                        }
                        Err(error) => json!({"content":[{"type":"text","text":error.1}],"isError":true}),
                    };
                    serde_json::to_vec(&json!({"jsonrpc":"2.0","id":id,"result":result})).map_err(ApiError::internal)
                }).await.map_err(ApiError::internal).and_then(|value| value);
                match output {
                    Ok(bytes) => send_json(res, bytes, depot),
                    Err(error) => error.render(res),
                }
                return;
            } else {
                Err(json!({"code":-32602,"message":"unknown tool"}))
            }
        }
        _ => Err(json!({"code":-32601,"message":"method not found"})),
    };
    res.render(Json(match result {
        Ok(result) => json!({"jsonrpc":"2.0","id":id,"result":result}),
        Err(error) => json!({"jsonrpc":"2.0","id":id,"error":error}),
    }));
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    use salvo::test::{ResponseExt, TestClient};

    #[test]
    fn base64_data_uri_and_option_validation() {
        assert_eq!(
            decode_base64("data:image/png;base64,YQ==")
                .unwrap()
                .as_ref(),
            b"a"
        );
        assert!(decode_base64("not base64").is_err());
        assert!(decode_base64("data:image/png,a").is_err());
        assert!(charset(&json!(8)).is_err());
        assert!(filter(&serde_json::from_value(json!({"colors":["unknown"]})).unwrap()).is_err());
        assert!(filter(
            &serde_json::from_value(json!({"color_filter":[[[180,0,0],[0,255,255]]]})).unwrap()
        )
        .is_err());
        assert!(filter(
            &serde_json::from_value(
                json!({"colors":["custom"],"custom_color_ranges":{"custom":[[0,1,2],[10,20,30]]}})
            )
            .unwrap()
        )
        .unwrap()
        .is_some());
    }

    #[tokio::test]
    async fn http_contract_limits_and_admission() {
        let args = Args::parse_from([
            "ddddocr",
            "--old",
            "--det",
            "--slide",
            "--mcp",
            "--max-concurrency",
            "2",
            "--max-body-mib",
            "1",
            "--ocr-charset-range",
            "0",
        ]);
        ARGS.set(args.clone()).unwrap();
        initialize(&args).unwrap();
        let service = service(&args);
        let image = include_bytes!("../image/1.png");
        let mut restricted = TestClient::post("http://localhost/ocr")
            .json(&json!({"image":BASE64_STANDARD.encode(image)}))
            .send(&service)
            .await;
        let restricted: Value = restricted.take_json().await.unwrap();
        assert!(restricted["result"]
            .as_str()
            .unwrap()
            .chars()
            .all(|c| c.is_ascii_digit()));
        let reset = TestClient::post("http://localhost/set_charset_range")
            .json(&json!({"charset_range":[]}))
            .send(&service)
            .await;
        assert_eq!(reset.status_code, Some(StatusCode::OK));
        let mut response = TestClient::post("http://localhost/ocr")
            .json(&json!({"image":BASE64_STANDARD.encode(image)}))
            .send(&service)
            .await;
        assert_eq!(response.status_code, Some(StatusCode::OK));
        let baseline: Value = response.take_json().await.unwrap();
        assert_eq!(
            baseline["result"],
            model(false).unwrap().classification(image).unwrap()
        );
        assert_eq!(baseline.as_object().unwrap().len(), 2);
        assert!(baseline["processing_time"].is_number());
        let mut response = TestClient::post("http://localhost/ocr/binary")
            .add_header("content-type", "image/png", true)
            .body(image.as_slice())
            .send(&service)
            .await;
        let binary: Value = response.take_json().await.unwrap();
        assert_eq!(binary["result"], baseline["result"]);
        let mut response = TestClient::post("http://localhost/ocr")
            .json(&json!({"image":BASE64_STANDARD.encode(image),"probability":true}))
            .send(&service)
            .await;
        let probability: Value = response.take_json().await.unwrap();
        assert_eq!(probability["result"]["text"], baseline["result"]);
        assert!(probability["result"]["probabilities"][0][0].is_array());
        assert!(probability["result"]["charset"].is_array());
        for (path, data, code) in [
            ("ocr", json!({"image":"invalid"}), StatusCode::BAD_REQUEST),
            ("ocr", json!({}), StatusCode::BAD_REQUEST),
            (
                "ocr?use_gpu=true",
                json!({"image":"YQ=="}),
                StatusCode::BAD_REQUEST,
            ),
        ] {
            let mut response = TestClient::post(format!("http://localhost/{path}"))
                .json(&data)
                .send(&service)
                .await;
            assert_eq!(response.status_code, Some(code));
            assert!(response.take_json::<Value>().await.unwrap()["detail"].is_string());
        }
        let response = TestClient::post("http://localhost/ocr/binary")
            .add_header("content-type", "image/png", true)
            .body(vec![0; 1024 * 1024 + 1])
            .send(&service)
            .await;
        assert_eq!(response.status_code, Some(StatusCode::PAYLOAD_TOO_LARGE));
        let unread = TestClient::post("http://localhost/ocr/binary")
            .add_header("content-type", "image/png", true)
            .body(image.as_slice())
            .send(&service)
            .await;
        assert_eq!(unread.status_code, Some(StatusCode::OK));
        assert_eq!(ADMISSION.get().unwrap().available_permits(), 1);
        drop(unread);
        assert_eq!(ADMISSION.get().unwrap().available_permits(), 2);
        let permits = ADMISSION
            .get()
            .unwrap()
            .clone()
            .acquire_many_owned(2)
            .await
            .unwrap();
        let response = TestClient::post("http://localhost/ocr")
            .json(&json!({}))
            .send(&service)
            .await;
        assert_eq!(response.status_code, Some(StatusCode::SERVICE_UNAVAILABLE));
        assert_eq!(response.headers().get("retry-after").unwrap(), "1");
        let response = TestClient::get("http://localhost/health")
            .send(&service)
            .await;
        assert_eq!(response.status_code, Some(StatusCode::OK));
        drop(permits);
        let mut response=TestClient::post("http://localhost/mcp").json(&json!({"jsonrpc":"2.0","id":0,"method":"tools/call","params":{"name":"ocr","arguments":{"image":BASE64_STANDARD.encode(image)}}})).send(&service).await;
        let mcp_response: Value = response.take_json().await.unwrap();
        assert_eq!(mcp_response["result"]["isError"], false); // --old previously disabled MCP OCR.
        let tool_output: Value = serde_json::from_str(
            mcp_response["result"]["content"][0]["text"]
                .as_str()
                .unwrap(),
        )
        .unwrap();
        assert_eq!(tool_output["result"], baseline["result"]);
        assert!(tool_output["processing_time"].is_number());
        assert_eq!(baseline.as_object().unwrap().len(), 2);
        assert_eq!(mcp_response["result"]["structuredContent"], tool_output);
        let mut malformed = TestClient::post("http://localhost/mcp")
            .json(&json!({"jsonrpc":"2.0","id":true,"method":"ping"}))
            .send(&service)
            .await;
        assert_eq!(
            malformed.take_json::<Value>().await.unwrap()["error"]["code"],
            -32600
        );
        let mut invalid_json = TestClient::post("http://localhost/mcp")
            .add_header("content-type", "application/json", true)
            .body("{")
            .send(&service)
            .await;
        assert_eq!(
            invalid_json.take_json::<Value>().await.unwrap()["error"]["code"],
            -32700
        );
        let response = TestClient::post("http://localhost/mcp")
            .json(&json!({"jsonrpc":"2.0","method":"notifications/initialized"}))
            .send(&service)
            .await;
        assert_eq!(response.status_code, Some(StatusCode::ACCEPTED));
    }
}
