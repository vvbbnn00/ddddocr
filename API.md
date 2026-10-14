# HTTP API 与 Linux 部署

识别接口统一返回 `result` 和 `processing_time`，请求字段参考[上游 ddddocr](https://github.com/sml2h3/ddddocr/tree/master/ddddocr) 的 `ddddocr.api.app`。服务始终关闭广告，`show_ad` 查询参数仅用于兼容，不会输出广告。

## Docker 一键启动

在安装了 Docker Engine 和 Compose 插件的 Linux 主机上，在仓库目录运行：

```sh
docker compose up -d --build --wait
```

默认监听端口 `8000`。镜像使用 Debian、非 root 用户和内嵌模型，运行时不下载模型。Compose 将文件系统设为只读并限制日志文件大小。

```sh
curl --fail http://127.0.0.1:8000/health
docker compose logs --tail=50
docker compose down
```

修改端口和并发数：

```sh
DDDDOCR_PORT=8001 DDDDOCR_MAX_CONCURRENCY=4 DDDDOCR_INFERENCE_THREADS=1 \
  docker compose up -d --build --wait
```

Dockerfile 支持 Linux amd64 / arm64 原生构建。构建从 crates.io 获取锁定版本的 Rust 依赖，并从 ONNX Runtime 二进制分发站下载运行库，不克隆其他 Git 仓库；仍需要网络访问软件包源。仓库提供双架构 Linux CI，验证构建、单元测试和容器 HTTP 接口。

## 推荐：直接发送图片字节

OCR 和检测均接受原始 PNG/JPEG 等图片文件，无需 Base64。不要先把图片解码为像素数据；直接发送压缩图片文件。

```sh
curl --fail http://127.0.0.1:8000/ocr/binary \
  -H 'Content-Type: image/png' --data-binary @image/1.png

curl --fail 'http://127.0.0.1:8000/ocr/binary?png_fix=true&charset_range=0' \
  -H 'Content-Type: application/octet-stream' --data-binary @image/1.png

curl --fail http://127.0.0.1:8000/det/binary \
  -H 'Content-Type: image/jpeg' --data-binary @image/5.jpg
```

返回格式：

```json
{"result":"识别文字","processing_time":0.012}
```

其中 `processing_time` 的单位是秒。上面的文字和耗时仅为格式示例。

Base64 的图片部分长度为 `4 × ceil(原始字节数 / 3)`。直接发送文件通常比 Base64 少约 25% 的图片传输量，并省去 Base64 编解码。此比较不包含 HTTP 头；原图为 PNG/JPEG 时，再压缩通常收益有限。

## 文件上传与 Base64 请求

| 功能 | JSON 路径 | 文件路径及字段 | 二进制路径 |
| --- | --- | --- | --- |
| OCR | `/ocr`，`image` | `/ocr/file`，`file` | `/ocr/binary` |
| 目标检测 | `/det`，`image` | `/det/file`，`file` | `/det/binary` |
| 滑块匹配 | `/slide_match`，`target_image`、`background_image` | `/slide_match/file`，同名两个文件 | 使用文件上传 |
| 图片比较 | `/slide_comparison`，同上 | `/slide_comparison/file`，同上 | 使用文件上传 |

`/ocr`、`/det` 路径也接受文件或二进制；滑块路径也接受 multipart。

```sh
curl --fail http://127.0.0.1:8000/ocr/file \
  -F 'file=@image/1.png' -F 'probability=false' -F 'colors=[]'

curl --fail http://127.0.0.1:8000/slide_match/file \
  -F 'target_image=@image/hk.png' -F 'background_image=@image/bg.png' \
  -F 'simple_target=false'
```

JSON 请求仍支持 `image`、`target_image`、`background_image` 的 Base64 字符串，也接受 `data:image/png;base64,...` 格式。

OCR 支持 `probability`、`png_fix`、整数或字符串 `charset_range`、`colors`、`custom_color_ranges`，以及 `color_filter` 和 `color_filter_colors` / `color_filter_custom_ranges` 字段。multipart 的数组和对象字段使用 JSON 字符串；二进制请求通过查询参数传这些选项。

`probability=true` 时，`result` 为对象，包含 `text`、`probabilities`、`charset`、`confidence`。`probabilities` 形状为 `[时间步][1][字符]`。

检测的 `result` 是坐标数组。滑块匹配的 `result` 包含 `target`、`target_x`、`target_y`；图片比较的 `result` 包含 `target: [x, y]`。

## 模型与兼容边界

- 默认加载 `common_old.onnx`。`beta=true` 使用 `common.onnx`，首次使用时加载，随后复用。`old=true` 优先选择旧模型。
- 启动参数 `--ocr` 选择新模型作为默认值，`--old` 选择旧模型；`--det` 和 `--slide` 控制功能。未指定功能时启用 OCR、检测和滑块。
- `--ocr-path` 读取指定模型和同名 `.json` 字符集，`--det-path` 读取指定检测模型，内嵌模型构建也遵守这些参数。
- `POST /set_charset_range` 接受 `{"charset_range":["0","1"]}`；空数组清除设置。设置按模型隔离。并发客户端优先使用每次请求自己的 `charset_range`，避免改变其他客户端的默认值。
- 此 HTTP 服务使用 CPU。`use_gpu=true`、`flag=true` 会明确返回 `400`，不会悄悄忽略。自定义模型的概率输出暂不支持。
- 兼容目标是上游 app 的 HTTP 协议。Rust 与 Pillow/OpenCV 的图像算法存在差别，识别文字、概率数值和滑块坐标不保证逐值相同。
- 不提供另一套 API Server 的 `/initialize`、`/switch-model`、`/toggle-feature` 等管理接口。

MCP 通过 `--mcp` 启用，`--only-mcp` 仅开放 MCP 和状态/文档接口。工具调用结果也包含 `result` 和 `processing_time`。

错误响应使用对应 HTTP 状态码和 `{"detail":"错误说明"}`，无需切换返回格式。状态、健康检查和配置接口直接返回各自的信息对象。

## 并发与限制

模型由请求共享，每个模型最多加载一份；首次请求另一个 OCR 模型时才加载它。CPU 推理在线程池中执行，默认每个 ONNX session 使用 1 个线程，避免多请求叠加模型内部线程导致抢占。

| 参数或限制 | 默认值 | 行为 |
| --- | --- | --- |
| `--max-concurrency` | 本机可用 CPU 数，上限 8；Compose 为 4 | 限制上传、解析、推理和响应发送中的请求数量，满载直接返回 `503` 和 `Retry-After: 1` |
| `--inference-threads` | 1 | 每个 ONNX session 的算子线程数 |
| `--max-body-mib` | 24 MiB | 包括 JSON/Base64 或 multipart 开销，超限 `413`，也检查无 Content-Length 的请求 |
| `--upload-timeout-secs` | 15 秒 | 上传超时返回 `408` |
| 单张图片 | 8 MiB、最长边 4096 像素 | 解码前限制输入和分配量 |
| OCR 缩放后宽度 | 8192 像素 | 避免极端宽高比生成巨型张量 |
| 概率输出缩放后宽度 | 1024 像素 | 限制概率表大小 |
| 滑块模板匹配 | 最多 1 亿次像素比较 | 超限返回 `400`，防止大图长期占用 CPU |

非法图片和选项返回 `400`，不支持的 Content-Type 返回 `415`。上游部分参数验证使用 `422`，本服务统一使用 `400`。`/health`、`/status`、`/config` 不占推理名额，`/docs` 提供接口文档。

客户端应限制自身并发，并在收到 `503` 后按 `Retry-After` 退避重试。提高服务并发数前，用目标机器和真实图片测量吞吐量、延迟和内存，过高的并发不会自动提高吞吐量。

## 验证

```sh
cargo test --locked
cargo clippy --locked --all-targets -- -D warnings
python3 test_api.py --url http://127.0.0.1:8000
python3 tests/benchmark.py --url http://127.0.0.1:8000 --requests 200 --concurrency 4
```

`test_api.py` 需要 OCR、检测和滑块功能，检查三种上传方式、错误响应、突发并发、慢上传占满时的拒绝策略及健康检查。基准脚本分别报告成功吞吐量、延迟和 `503` 数量。
