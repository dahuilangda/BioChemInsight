## 快速开始

### 1. 启动后端

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

### 2. 启动前端

```bash
cd frontend/ui
npm install
npm run dev
```

开发服务器默认在 `http://127.0.0.1:5173`，并通过 Vite 代理将 `/api` 请求转发到 `http://127.0.0.1:8000`。若后端地址不同，可在启动前设置 `VITE_API_BASE` 环境变量。

### 3. 工作流说明

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