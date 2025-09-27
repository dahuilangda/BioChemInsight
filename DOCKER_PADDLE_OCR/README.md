# DOCKER_PADDLE_OCR 服务配置指南

基于最新 `PaddleOCR` (PPStructureV3) 的独立 OCR 推理服务。服务以 HTTP 接口方式提供 Markdown 结果，方便主程序或其它项目调用。

## 1. 环境依赖

1. **NVIDIA Docker**（需要 GPU 加速环境）
2. **Docker Compose v2**
3. 可选：若希望持久化模型缓存，请预先创建 `cache/paddlex` 目录

> **提示**：首次启动会自动下载官方模型，可能需要较长时间并占用约 8~10 GB 的磁盘空间。

## 2. 构建与启动

```bash
# 在仓库根目录执行
cd DOCKER_PADDLE_OCR

# 构建镜像（首次运行或更新后执行）
docker compose build

# 启动服务
docker compose up -d
```

启动后默认监听 `8010` 端口，容器名为 `paddle-ocr-server`。

查看日志：

```bash
docker compose logs -f
```

停止服务：

```bash
docker compose down
```

## 3. 接口说明

服务提供以下两个核心接口：

### 接口 1：PDF 到 Markdown

- **URL**: `POST http://<host>:8010/v1/pdf-to-markdown`
- **描述**: 将 PDF 文档的指定页面范围转换为 Markdown 格式。
- **Content-Type**: `multipart/form-data`
- **参数**：
  - `file` *(必填)*：PDF 文件。
  - `page_start` *(选填)*：起始页（默认 `1`，基于 1 的索引）。
  - `page_end` *(选填)*：结束页（默认 `-1`，表示处理到文档末尾）。
  - `return_raw` *(选填)*：是否返回原始 JSON 结构，默认 `false`。

**响应示例 (PDF)**：

```json
{
  "markdown": "# Page 1\n...\n\n-#-#-#-#-\n\n# Page 2\n...",
  "page_numbers": [1, 2],
  "page_count": 2
}
```

### 接口 2：图片到 Markdown

- **URL**: `POST http://<host>:8010/v1/image-to-markdown`
- **描述**: 将单个图片文件（如 PNG, JPG）转换为 Markdown 格式。
- **Content-Type**: `multipart/form-data`
- **参数**：
  - `file` *(必填)*：图片文件，支持 `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff` 等格式。
  - `return_raw` *(选填)*：是否返回原始 JSON 结构，默认 `false`。

**响应示例 (图片)**：

```json
{
  "markdown": "...",
  "page_numbers": [1],
  "page_count": 1
}
```

## 4. Python 调用示例

### 示例 1：处理 PDF

```python
import requests

url = "http://localhost:8010/v1/pdf-to-markdown"
files = {"file": ("demo.pdf", open("data/demo.pdf", "rb"), "application/pdf")}
form = {"page_start": "1", "page_end": "3"}

resp = requests.post(url, files=files, data=form, timeout=600)
resp.raise_for_status()
markdown = resp.json()["markdown"]
print("--- PDF OCR Result ---")
print(markdown[:500])
```

### 示例 2：处理图片

```python
import requests

# 准备一张图片文件用于测试
# with open("my_image.png", "wb") as f:
#     f.write(...)

url = "http://localhost:8010/v1/image-to-markdown"
files = {"file": ("test.png", open("path/to/your/image.png", "rb"), "image/png")}
form = {"return_raw": "false"}

resp = requests.post(url, files=files, data=form, timeout=300)
resp.raise_for_status()
markdown = resp.json()["markdown"]
print("\n--- Image OCR Result ---")
print(markdown[:500])
```

## 5. 与主项目联动

1. 在 `constants.py` 中新增或设置：
   ```python
   PADDLEOCR_SERVER_URL = "http://localhost:8010"
   ```
2. 运行命令行或前端时，通过参数 `--ocr-engine paddleocr` 即可调用该容器。

## 6. 常见问题

- **模型重复下载**：
  - 确保 `docker-compose.yml` 中的 `./cache/paddlex` 挂载目录存在，以复用缓存。
- **显存占用较高**：
  - 默认使用 x2 放大渲染以提升识别率，可在 `app/server.py` 中调整矩阵参数。
- **接口超时**：
  - 大文件请适当调大客户端的 `timeout`，或分批提交页码范围。
