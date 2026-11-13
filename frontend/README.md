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

后端在 8000 端口对外提供服务（上述命令绑定全部网卡），关键接口：

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

开发服务器默认在 `http://127.0.0.1:5173`，并通过 Vite 代理将 `/api` 请求转发到 `http://127.0.0.1:8000`。若后端地址不同，可在启动前设置 `VITE_API_BASE`（Axios base）与 Vite 代理目标 `VITE_PROXY_API` 环境变量。

### 4. 工作流说明

1. 上传或拖入 PDF，也可通过专利号导入；首页只预取第一页。
2. 点击 Start 启动完整流程，或进入 Structures / Bioactivity 分别提取。缩略图支持点击、Enter / Space 和 Shift 连选；手选关闭 Auto，重新开启 Auto 会清空手选。调整选择不会提交任务，点击 Extract 后才执行。
3. 桌面结果页左侧为 PDF，右侧为卡片或表格。点击记录、原图或结构图，定位到原文页与标注区域；Edit 单独打开结构编辑器。
4. 拖动分隔条调整两侧宽度，方向键微调，双击恢复 55%；比例自动保存。PDF 支持页码跳转、缩放、适合宽度与原文件打开。专注模式扩大工作区，Escape 退出。
5. 窄屏按需切换数据与 PDF；点击记录进入原文，Escape 返回。筛选面板提供分页大小、密度、列和原页缩略图选项，默认隐藏重复内容。
6. 编辑自动保存，失败保留修改并可重试；导出、切换任务与替换文件前等待保存。Export 下载包含当前数据的 ZIP。
7. Jobs 显示后台进度，可筛选、打开或取消任务；Escape 或关闭按钮收起。后台标签页暂停轮询，返回后继续。

生产构建会生成 Brotli / gzip 静态文件。`docker/static-server.py` 直接发送预压缩资源，散列资源使用长期缓存，HTML 重新验证；API 代理流式传输并保留 PDF Range 响应。PDF 阅读器按需请求字节区间、只绘制当前页，并限制画布缓存。实际传输量依 PDF 的对象和交叉引用布局而异。

## 目录结构

```
frontend/
├── backend/
│   ├── __init__.py
│   ├── celery_app.py      # Celery 实例
│   ├── celery_tasks.py    # 任务入队与子进程生命周期
│   ├── constants.py       # 环境相关常量（PROJECT_ROOT 等）
│   ├── main.py            # FastAPI 入口
│   ├── pdf_manager.py     # PDF 元数据与存储
│   ├── queue_dispatcher.py# 队列分发与僵死任务恢复
│   ├── schemas.py         # Pydantic 数据模型
│   ├── task_child_runner.py # 隔离子进程执行任务
│   ├── task_manager.py    # 任务状态管理
│   ├── work_queue.py      # Redis 队列操作
│   └── data/              # 上传 PDF、任务产物、标注
└── ui/
    ├── index.html
    ├── package.json
    ├── src/
    │   ├── App.tsx        # 核心界面逻辑
    │   ├── api/client.ts  # Axios 封装
    │   ├── components/    # PDF 预览、工作区、结构编辑器等组件
    │   ├── utils/         # 前端工具（化学渲染等）
    │   ├── main.tsx
    │   ├── styles.css
    │   └── types.ts
    ├── tsconfig.json
    ├── tsconfig.node.json
    └── vite.config.ts
```

## 调用现有 harness API

`utils/api_harness.py` 直接请求独立的结构服务，不在客户端加载模型。默认地址为
`http://127.0.0.1:8011`，可用 `HARNESS_API_URL` 或 `--base-url` 指定。

```bash
python -m utils.api_harness health
python -m utils.api_harness skills
python -m utils.api_harness detect page.png --query conf=0.25
python -m utils.api_harness recognize crop.png
python -m utils.api_harness --output page-result.json parse-page page.png --query max_structures=4
python -m utils.api_harness resolve-figure crop.png --field document_id=patent-1
python -m utils.api_harness verify-structure crop.png --field 'smiles=CCO'
python -m utils.api_harness invoke variant_sites --json variant.json
```

`variant.json` 示例：`{"hgvs":["p.Val600Glu"]}`。
`--field` 对应 API 的 multipart 表单字段，`--query` 对应查询参数，可重复传入。
全局选项（地址、超时、输出文件）放在子命令前。请求默认超时 600 秒；HTTP 错误、
连接失败和响应中的能力失败均返回非零退出码。模型推理请求不自动重试，避免重复计算。
若机器设置了代理，本机联调时设置 `NO_PROXY=127.0.0.1,localhost`。

### 本地回归检查

```bash
python -m pytest -q frontend/tests/test_api_harness.py
cd frontend/ui
npm run build
```

结果页浏览器检查使用独立 API 测试数据，验证 320、375、768、1280、1920 像素宽度、
卡片活性值展示和第 20 张卡片滚动后的图片加载，不代表模型结果准确率或完整流水线验证。
启动 Vite 后，在仓库根目录执行：

```bash
npm install --prefix /tmp/biochem-ui-qa playwright@1.49.1
NODE_PATH=/tmp/biochem-ui-qa/node_modules node frontend/tests/workbench.browser.cjs
```

若没有 Chromium，先运行 `/tmp/biochem-ui-qa/node_modules/.bin/playwright install chromium`。
可用 `WORKBENCH_URL` 指定页面地址，`QA_OUTPUT_DIR` 保存各分辨率截图。

### 完整 PDF 流程与断点续查

harness 也可直接驱动工作台 API：上传 PDF、提交完整任务、轮询、读取结构/活性结果、校验并下载 ZIP。

```bash
python -m utils.api_harness workflow patent.pdf \
  --workbench-url http://127.0.0.1:8000/api \
  --report run.json --download results.zip --wait-seconds 900

# 观察超时或终端中断后，继续查询同一个任务，不重新上传或提交。
python -m utils.api_harness workflow --resume TASK_ID \
  --report run.json --download results.zip --wait-seconds 900
```

`run.json` 在任务提交后立即保存任务 ID，之后原子更新状态及结果。
观察超时不会取消远程任务；任务失败会保留服务端错误并返回非零退出码。
`--workbench-url` 的默认值为 `WORKBENCH_API_URL` 或 `http://127.0.0.1:8000/api`。
`--assay IC50` 可重复指定测定名称。报告、PDF 和下载必须使用不同文件路径。

编辑器回归也包含在 `workbench.browser.cjs` 中：加载真实 JSME 静态资源，检查六种视口中的
弹窗与保存按钮，并检查保存的 Molfile 保留标准三行标题。API 数据仍为测试数据。

保存与取消的浏览器回归（使用异步 API 测试数据）：

```bash
NODE_PATH=/tmp/biochem-ui-qa/node_modules node frontend/tests/workbench.persistence.browser.cjs
```

覆盖结构任务与完整流程任务的失败重试、连续编辑请求排序、下载前保存、重新打开任务前保存、
删除最后一条记录后刷新与导出，旧任务延迟响应隔离、结果加载前禁止保存、重跑时绑定新任务，以及取消失败后的继续轮询和重试。
后端安装了项目运行依赖后，可运行 `python -m pytest -q frontend/tests/test_workbench_persistence.py`，
验证实际 FastAPI 保存和 ZIP 导出契约；测试替换任务存储，不需要启动 Redis 或模型。
