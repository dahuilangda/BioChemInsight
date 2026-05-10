## 快速开始

### 1. 启动本地异步任务服务

前端的结构解析、活性解析和完整自动流程都是异步任务。若不使用 `docker compose`，本地开发时需要先启动 Redis、Queue dispatcher 和 Celery worker；否则页面可以打开，但提交任务后不会真正执行。

**终端 A：启动 Redis**

```bash
# Ubuntu/Debian
sudo systemctl start redis-server

# 或以前台开发模式启动
redis-server
```

如 Redis 地址不是默认值，请在所有后端相关终端中设置：

```bash
export REDIS_URL=redis://localhost:6379/0
export CELERY_BROKER_URL=$REDIS_URL
export CELERY_RESULT_BACKEND=$REDIS_URL
```

**终端 B：启动 Queue dispatcher**

```bash
python -m frontend.backend.queue_dispatcher
```

**终端 C：启动 Celery worker**

```bash
celery -A frontend.backend.celery_app.celery_app worker \
  -Q compute \
  --pool threads \
  --concurrency 2 \
  --prefetch-multiplier 1 \
  --loglevel INFO
```

### 2. 启动后端

```bash
uvicorn frontend.backend.main:app --host 0.0.0.0 --port 8000
```

默认监听 `http://127.0.0.1:8000`，对外暴露以下关键接口：

- `POST /api/pdfs`：上传 PDF，返回 `pdf_id` 与页数。
- `GET /api/pdfs/{pdf_id}/pages/{page}`：返回指定页的 PNG base64 预览。
- `POST /api/tasks/structures`：提交结构解析任务（异步执行）。
- `GET /api/tasks/{task_id}`：查询任务状态与进度。
- `GET /api/tasks/{task_id}/structures`：获取解析结果。
- `PUT /api/tasks/{task_id}/structures`：保存前端修改后的结构数据。
- `GET /api/tasks/{task_id}/download`：下载当前结果的 CSV。
- `GET /api/artifacts?path=...`：安全读取任务生成的局部图片。

可直接用 `curl` 提交异步任务。推荐使用完整自动流程：

```bash
API=http://localhost:8000/api

# 1) 上传 PDF，并自动提取 pdf_id。
PDF_ID=$(
  curl -s -X POST "$API/pdfs" \
    -F "file=@data/sample.pdf" \
  | python -c 'import json,sys; print(json.load(sys.stdin)["pdf_id"])'
)

# 2) 提交完整自动流程，并自动提取 task_id。
TASK_ID=$(
  curl -s -X POST "$API/tasks/full-pipeline" \
    -H "Content-Type: application/json" \
    -d "{\"pdf_id\":\"$PDF_ID\",\"structure_filter_strictness\":\"strict\",\"lang\":\"en\"}" \
  | python -c 'import json,sys; print(json.load(sys.stdin)["task_id"])'
)

# 3) 查询状态并下载结果。
curl -s "$API/tasks/$TASK_ID" | python -m json.tool
curl -L "$API/tasks/$TASK_ID/download" -o result.csv
```

### 3. 启动前端

```bash
cd frontend/ui
npm install
npm run dev
```

开发服务器默认在 `http://127.0.0.1:5173`，并通过 Vite 代理将 `/api` 请求转发到 `http://127.0.0.1:8000`。若后端地址不同，可在启动前设置 `VITE_API_BASE` 环境变量。

### 4. 工作流说明

1. 上传 PDF，系统会返回页数并建立临时缓存。
2. 页面展示为卡片式缩略图，可在顶部切换「结构页」或「活性页」模式并直接点选、取消选中；文本框仍支持手动输入 `1,3,5-8` 等格式，两种方式自动联动。
3. 「结构提取」和「活性数据提取」分离：结构提取用于化合物解析并支持在线编辑，活性提取需填写测定名称列表（如 `IC50`），分别启动异步任务，互不影响。
4. 任务完成后展示对应表格：结构表格支持就地编辑、保存与恢复；活性表格以只读形式列出各测定数据。
5. 两类任务均可实时查看进度信息，并通过「下载 CSV」导出最新结果。
6. 结构表格右侧按钮可查看结构图片或原文片段，便于核对。

## 目录结构

```
frontend/
├── backend/
│   ├── __init__.py
│   ├── main.py            # FastAPI 入口
│   ├── pdf_manager.py     # PDF 元数据与存储
│   ├── schemas.py         # Pydantic 数据模型
│   └── task_manager.py    # 内存任务调度与状态管理
└── ui/
    ├── index.html
    ├── package.json
    ├── src/
    │   ├── App.tsx        # 核心界面逻辑
    │   ├── api/client.ts  # Axios 封装
    │   ├── main.tsx
    │   ├── styles.css
    │   └── types.ts
    ├── tsconfig.json
    ├── tsconfig.node.json
    └── vite.config.ts
```
