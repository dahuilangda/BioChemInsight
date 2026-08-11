# BioChemInsight 🧪

**BioChemInsight** 是一个强大的平台，可以自动从科学文献中提取化学结构及其对应的生物活性数据。通过利用深度学习进行图像识别和OCR，它简化了为化学信息学、机器学习和药物发现研究创建高质量结构化数据集的流程。

![logo](images/BioChemInsight.jpeg)

## 功能特性 🎉

  * **自动化数据提取** 🔍: 自动从 PDF 文档中识别并提取化合物结构和生物活性数据（例如 IC50, EC50, Ki）。
  * **先进识别核心** 🧠: 采用顶尖的 DECIMER Segmentation 模型进行图像分析，并使用 PaddleOCR 进行文本识别。
  * **推荐视觉模型**: 视觉模型推荐使用 **GLM-V4.5** 或 **MiniCPM-V-4**，效果最佳。
  * **结构识别** ⚙️: 结合 DECIMER 分割与 MolNexTR，将化学图谱转换为 SMILES 字符串。内置 Markush 微调模型，专门优化专利中的骨架/片段识别。
  * **自动文档规划** 📄: 自动识别结构页面、活性页面和实验名称；也可以用页面范围约束处理范围。
  * **结构化数据输出** 🛠️: 将非结构化的文本和图像转换为可直接用于分析的格式，如 CSV 和 Excel。
  * **现代化 Web UI** 🌐: 基于 React 的前端界面配合 FastAPI 后端，提供直观的 PDF 处理、实时进度跟踪和交互式结果可视化。
  * **智能数据合并** 🔗: 基于化合物 ID 自动合并结构和生物活性数据，提供无缝的集成结果。


## 应用场景 🌟

  * **AI/ML 模型训练**: 为化学信息学和生物信息学领域的预测模型生成高质量的训练数据集。
  * **药物发现**: 加速构效关系（SAR）研究和先导化合物的优化过程。
  * **自动化文献挖掘**: 大幅减少从科学文章中手动整理数据所需的人力和时间。


## 工作流程 🚀

BioChemInsight 采用多阶段流水线将原始 PDF 转换为结构化数据：

1.  **PDF 预处理**: 将输入的 PDF 拆分为单个页面，然后将这些页面转换为高分辨率图像以供分析。
2.  **结构检测**: **DECIMER Segmentation** 扫描图像，以定位和分离化学结构图。
3.  **SMILES 转换**: MolNexTR 将分离出的图谱转换为机器可读的 SMILES 字符串。
4.  **标识符识别**: 视觉模型（推荐：**GLM-V4.5**）识别与每个结构相关的化合物标识符（例如，“化合物 **1**”、“**2a**”）。
5.  **生物活性提取**: **PaddleOCR** 从识别出的活性页面提取文本，大型语言模型则辅助解析和标准化生物活性结果。
6.  **数据整合**: 所有提取的信息——化合物ID、SMILES 字符串和生物活性数据——被合并到结构化文件（CSV/Excel）中，以便下载和进行下游分析。


## 安装 🔧

#### 步骤 1: 克隆代码仓库

```bash
git clone https://github.com/dahuilangda/BioChemInsight
cd BioChemInsight
```

#### 步骤 2: 配置常量

项目需要一个 `constants.py` 文件来设置环境变量和路径。我们提供了一个模板文件。

```bash
# 重命名示例文件
mv constants_example.py constants.py
```

然后，编辑 `constants.py` 文件，设置您的 API 密钥、模型路径和其他必要配置。

#### 步骤 3: 创建并激活 Conda 环境

```bash
conda install -c conda-forge mamba
mamba create -n chem_ocr python=3.10
conda activate chem_ocr
```

#### 步骤 4: 安装依赖

首先，安装支持 CUDA 的 PyTorch。

```bash
# 安装 CUDA 工具和 PyTorch
mamba install -c nvidia -c conda-forge cudatoolkit=11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

接下来，安装其余的 Python 包。

```bash
# 安装核心库 (使用镜像源以加快下载速度)
pip install SmilesPE opencv-python-headless -i https://pypi.tuna.tsinghua.edu.cn/simple
mamba install -c conda-forge jupyter pytesseract transformers
pip install PyMuPDF PyPDF2 openai Levenshtein mdutils tabulate python-multipart -i https://pypi.tuna.tsinghua.edu.cn/simple

# 安装 Web 服务依赖
pip install fastapi uvicorn celery redis -i https://pypi.tuna.tsinghua.edu.cn/simple

# 安装 Redis 服务端以及 Node.js/npm（用于异步 Web UI）
# 在 Ubuntu/Debian 上:
sudo apt-get install -y redis-server
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs

# 在 macOS 上 (使用 homebrew):
# brew install redis node
```


## 使用方法 📖

BioChemInsight 可以通过交互式网页界面或直接从命令行操作。

> **重要提示：** 在启动流程之前，请确保 PaddleOCR 微服务正在运行，并在 `constants.py` 中配置 `PADDLEOCR_SERVER_URL`。

### 网页界面 🌐

基于 React 的现代化网页界面提供了直观的文档处理平台，具备实时进度跟踪功能。

#### 本地启动 Web 服务

Web UI 使用异步任务队列。本地开发模式下需要同时启动 **5 个进程/服务**：

1. Redis：保存任务状态、队列状态，并作为 Celery broker/result backend。
2. FastAPI 后端。
3. Queue dispatcher：把 BioChemInsight 队列中的任务派发给 Celery。
4. Celery worker：真正执行结构解析、活性解析和完整流程任务。
5. Vite 前端开发服务器。

如果不想手动管理这些进程，推荐直接使用下方的 Docker Compose 部署。

**步骤 1: 启动 Redis**

Ubuntu/Debian：

```bash
sudo systemctl start redis-server
# 或以前台开发模式启动：
redis-server
```

macOS Homebrew：

```bash
brew services start redis
# 或：
redis-server
```

默认 Redis 地址为 `redis://localhost:6379/0`。如需修改：

```bash
export REDIS_URL=redis://localhost:6379/0
export CELERY_BROKER_URL=$REDIS_URL
export CELERY_RESULT_BACKEND=$REDIS_URL
```

**步骤 2: 启动后端 API 服务器**

在项目根目录下运行：

```bash
uvicorn frontend.backend.main:app --host 0.0.0.0 --port 8000 --reload
```

**步骤 3: 启动 Queue dispatcher**

在新的终端中，从项目根目录运行：

```bash
python -m frontend.backend.queue_dispatcher
```

**步骤 4: 启动 Celery worker**

在另一个终端中，从项目根目录运行：

```bash
celery -A frontend.backend.celery_app.celery_app worker \
  -Q compute \
  --pool threads \
  --concurrency 2 \
  --prefetch-multiplier 1 \
  --loglevel INFO
```

可以用环境变量调整本地并发：

```bash
export MAX_CONCURRENT_TASKS=2
export DISPATCHER_MAX_CONCURRENT_TASKS=2
export CELERY_WORKER_CONCURRENCY=2
export STRUCTURE_TASK_CONCURRENCY=2
```

**步骤 5: 启动前端开发服务器**

在新的终端中运行：

```bash
cd frontend/ui
npm install
NODE_OPTIONS="--max-old-space-size=8196" npm run dev
```

**步骤 6: 访问界面**

在网页浏览器中打开 `http://localhost:5173` 访问界面。后端 API 地址为 `http://localhost:8000`。


#### 网页界面功能

1.  **PDF 上传**: 通过直观界面上传和管理 PDF 文件。
2.  **自动提取规划**: 自动识别结构页面、活性页面和实验名称；页面缩略图和范围输入仍可用于约束处理范围。
3.  **分步处理流程**: 
    - **步骤 1**: 上传 PDF 并预览页面
    - **步骤 2**: 实时进度显示的化学结构提取
    - **步骤 3**: 基于结构约束化合物匹配的生物活性数据提取
    - **步骤 4**: 查看和下载合并结果
4.  **实时进度跟踪**: 详细状态更新的提取进度监控。
5.  **交互式结果**: 查看、编辑和下载结构化数据，集成化合物-活性匹配。
6.  **自动数据合并**: 基于化合物 ID 无缝结合结构和生物活性数据。


### 命令行界面 (CLI)

对于批处理和自动化任务，推荐使用命令行界面。

#### 自动提取（推荐）

不提供页面和实验名称参数时，BioChemInsight 会自动规划结构和活性提取。

```bash
python pipeline.py data/sample.pdf \
    --output output
```

**约束处理范围示例:**

  * **从指定页面中提取结构:**
    ```bash
    python pipeline.py data/sample.pdf --structure-pages "242-250,255,260-267" --output output
    ```
  * **提取指定活性页面和实验:**
    ```bash
    python pipeline.py data/sample.pdf --structure-pages "242-267" --assay-pages "30,35,270-272" --assay-names "IC50,FRET EC50" --output output
    ```


## 输出 📂

平台将在指定的输出目录中生成以下结构化数据文件：

  * `structures.csv`: 包含检测到的化合物标识符及其对应的 SMILES 表示。
  * `assay_data.json`: 存储提取出的原始生物活性数据。
  * `merged.csv`: 一个合并文件，将化学结构与其相关的生物活性数据整合在一起。


## Docker 部署 🐳

在容器化环境中部署 BioChemInsight 以确保一致性和可移植性。

#### 推荐方式：Docker Compose

Docker Compose 部署包含：
- `web`：FastAPI + React UI。
- `redis`：保存任务卡片和队列状态。
- `dispatcher`：任务派发服务。
- `worker`：Celery 执行器。

启动服务：

```bash
docker compose up --build -d
```

容器入口会自动检测宿主机 bind mount 目录的 owner，并用对应 UID/GID 运行应用。通常不需要设置 `APP_UID` 或 `APP_GID`；只有需要覆盖自动检测结果时才设置：

```bash
APP_UID=1000
APP_GID=1000
ZENODO_HOST=188.185.48.75
```

`ZENODO_HOST` 是可选项，只在镜像构建阶段下载 Zenodo 上的 DECIMER 分子分割权重时使用。Dockerfile 使用 `curl --resolve` 指定 `zenodo.org` 的解析 IP，不会写 `/etc/hosts`，因此兼容 BuildKit。如果当前网络可以正常解析 `zenodo.org`，可以不设置它。

启动容器前，如果宿主机 bind mount 目录不存在，先创建它们。`web` 容器入口会先以 root 短暂运行，把 `output` 和 `frontend/backend/data` 修正为自动检测到的运行 UID/GID 可写，然后再降权启动后端和前端：

```bash
mkdir -p data output frontend/backend/data
```

可以在 `docker-compose.yml` 或 Compose `.env` 文件里调节并发：

```bash
MAX_CONCURRENT_TASKS=3
DISPATCHER_MAX_CONCURRENT_TASKS=3
CELERY_WORKER_CONCURRENCY=3
STRUCTURE_TASK_CONCURRENCY=2
```

Compose 网络使用 `172.200.0.0/16`，不是 Docker 常见的 `172.17.*` bridge 网段。

Docker 镜像在安装运行时数据处理和 RDKit 依赖后会固定 `numpy==1.26.4`，并在复制项目文件前检查实际安装的 numpy 版本，避免依赖求解时悄悄升级 numpy。

启动后，访问：
- 前端界面：`http://localhost:3000`
- 后端 API：`http://localhost:8000`

查看服务和日志：

```bash
docker compose ps
docker compose logs --tail 100 web worker dispatcher redis
```

Redis 可能输出宿主机内核参数 `vm.overcommit_memory` 的 warning。服务可以在 warning 存在时运行，但长期部署建议在宿主机启用：

```bash
sudo sysctl vm.overcommit_memory=1
echo 'vm.overcommit_memory=1' | sudo tee /etc/sysctl.d/99-redis-overcommit.conf
sudo sysctl --system
docker compose restart redis
```

#### 使用 `curl` 提交任务

Docker Compose 会暴露与 Web UI 相同的 FastAPI 后端，因此可以直接用 `curl` 以 API 方式上传 PDF、提交任务、轮询状态和下载结果。

推荐方式：使用 **完整自动流程**，自动检测结构页、活性页和实验名称。

```bash
API=http://localhost:8000/api

# 1) 上传 PDF，并自动提取返回的 pdf_id。
PDF_ID=$(
  curl -s -X POST "$API/pdfs" \
    -F "file=@data/sample.pdf" \
  | python -c 'import json,sys; print(json.load(sys.stdin)["pdf_id"])'
)
echo "$PDF_ID"

# 2) 提交推荐的完整自动流程，并自动提取 task_id。
TASK_ID=$(
  curl -s -X POST "$API/tasks/full-pipeline" \
    -H "Content-Type: application/json" \
    -d "{\"pdf_id\":\"$PDF_ID\",\"structure_filter_strictness\":\"strict\",\"lang\":\"en\"}" \
  | python -c 'import json,sys; print(json.load(sys.stdin)["task_id"])'
)
echo "$TASK_ID"

# 3) 查询任务状态，直到 "status" 变成 "completed"。
curl -s "$API/tasks/$TASK_ID" | python -m json.tool

# 4) 下载结果 CSV。
curl -L "$API/tasks/$TASK_ID/download" -o result.csv
```

高级用法：也可以分开提交结构解析和活性解析任务。

```bash
API=http://localhost:8000/api

# 结构解析：指定页面。
# 若不传 pages 或设置 auto_detect_pages=true，则自动检测结构页。
STRUCTURE_TASK_ID=$(
  curl -s -X POST "$API/tasks/structures" \
    -H "Content-Type: application/json" \
    -d "{\"pdf_id\":\"$PDF_ID\",\"pages\":\"1,3,5-7\",\"structure_filter_strictness\":\"strict\"}" \
  | python -c 'import json,sys; print(json.load(sys.stdin)["task_id"])'
)

# 活性解析：可传入已完成的结构任务 ID，用结构中的化合物列表约束匹配。
ASSAY_TASK_ID=$(
  curl -s -X POST "$API/tasks/assays" \
    -H "Content-Type: application/json" \
    -d "{\"pdf_id\":\"$PDF_ID\",\"pages\":\"10-12\",\"assay_names\":[\"IC50\",\"EC50\"],\"structure_task_id\":\"$STRUCTURE_TASK_ID\",\"lang\":\"en\"}" \
  | python -c 'import json,sys; print(json.load(sys.stdin)["task_id"])'
)

# 如需取消排队中或运行中的任务。
curl -s -X POST "$API/tasks/$ASSAY_TASK_ID/cancel" | python -m json.tool
```

#### 在 Docker 中运行命令行流程

如果您需要使用命令行执行批处理任务，请在 `docker run` 命令后指定 `python pipeline.py` 和相关参数来覆盖默认入口点。

```bash
docker run --rm --gpus all \
    -e http_proxy="" \
    -e https_proxy="" \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/output:/app/output \
    --entrypoint python \
    biocheminsight \
    pipeline.py data/sample.pdf \
    --output output
```

#### 进入容器进行交互式会话

如果需要在容器内进行调试或手动运行命令：

```bash
docker run --gpus all -it --rm \
  --entrypoint /bin/bash \
  -e http_proxy="" \
  -e https_proxy="" \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/output:/app/output \
  --name biocheminsight_container \
  biocheminsight
```


## 微调 MolNexTR Markush 结构识别 🔬

BioChemInsight 包含一套 MolNexTR 微调流程，目标是提升对 Markush 结构（骨架、片段、取代基和连接原子）的识别能力。微调后的模型可直接替换基础 MolNexTR 权重，但必须先通过当前 Markush 训练门控。

### 为什么需要微调？

基础 MolNexTR 模型针对普通化学结构训练，在 Markush 特有元素上仍需要专门的数据和门控。

### 当前策略

当前维护的训练目标是从视觉模型根源解决 BioChemInsight 的 Markush + fragment
组合问题，不是给某个 PDF 打补丁。MolNexTR 必须直接学会骨架标签和片段
attachment atom 的视觉证据。禁止用推理侧 fallback、输出改写、硬过滤或真实任务专用
配置名来修复。

使用干净的默认配置 `training/molnextr_markush/configs/markush.json`。当前维护的
数据策略以真实完整分子池为主，优先使用 PubChem canonical SMILES 以及可获得的专利/公开
分子池。完整分子会作为普通训练锚点，同时从这些完整分子上切出 attachment fragment。
普通分子可以使用 canonical SMILES，但受控 attachment target 必须同时覆盖 leading、
internal、terminal 三类 `*` 序列化。RGReco 风格 fragment 常见 `*N...`、`*O...`
这类 leading-star target；如果排除这种顺序，原版 MolNexTR 的首 token 先验不会被真正修正。
默认构建禁用手工 attachment 模板；如果启用，构建门禁会直接失败。训练集必须覆盖 wavy mark、可见 `*`、
acyl/table/document-style fragment，以及 R、X、Y、Z、Ar、Het、
Hal 等通用 Markush 标签。长训练前必须通过重复审计和图片 review。外部普通分子数据和评估
门禁必须保留，用来保持并提升 MolNexTR 的 pose 和 validity 能力。

当前泛化根治路线只放在训练侧。生产 checkpoint 必须从原版 MolNexTR 开始训练，
不能从已经微调过的 Markush checkpoint 继续堆补丁。维护中的默认配置组合了普通结构 replay、
ordinary-only 覆盖尾部的 teacher consistency、encoder+decoder L2-SP 正则、
`pad_tail_no_object_loss`、Markush label-set coverage、窄范围 Markush label unlikelihood，
以及 scoped attachment `*` 监督。target 表达本身是主修复，loss 是 greedy decoding 和过生成的护栏。
attachment 目标使用 target `*` 位置的 argmax-margin 监督，并对 leading-star 行使用首步
free-run 监督，让 dummy atom 在 greedy decoding 时真正胜过普通 terminal chemistry。
这个 unlikelihood 只允许压制 `R`/prime 这类 Markush-only 假阳性；不能压制 SMILES 数字、
方括号或常见原子字符，否则会破坏原始 OCR。
不要用 fallback、硬过滤、decoder 插值或 PDF 专用输出改写替代这条路线。上线门禁必须先保
Markush scaffold 正确且 `*` 数精确，再保原始普通结构能力不倒退，最后看真实任务 fragment 的 `*`
精确率。

### 快速开始

微调流程在 Docker 内运行，训练数据约需 30 GB 磁盘空间。

```bash
# 1. 构建训练镜像
docker build -f training/molnextr_markush/Dockerfile -t molnextr-markush-train:dev .

# 2. 下载数据集（约 27 GB）
docker run --rm -v $(pwd):/workspace -w /workspace \
  molnextr-markush-train:dev \
  python training/molnextr_markush/scripts/download_data.py

# 3. 构建训练数据集
docker run --rm -v $(pwd):/workspace -w /workspace \
  molnextr-markush-train:dev \
  python training/molnextr_markush/scripts/build_dataset.py

# 4. 训练（2× RTX 4070 约 33 小时）
docker run --rm --gpus all --shm-size=16g \
  -v $(pwd):/workspace -w /workspace \
  molnextr-markush-train:dev \
  python training/molnextr_markush/scripts/train.py

# 5. 评估（与基础模型对比）
docker run --rm --gpus all -v $(pwd):/workspace -w /workspace \
  molnextr-markush-train:dev \
  python training/molnextr_markush/scripts/evaluate.py
```

默认训练配置是 `training/molnextr_markush/configs/markush.json`。它从
`/workspace/models/molnextr_best.pth` 开始训练，读取
`training/molnextr_markush/data/dataset/train_pose_markush.csv`，输出到
`training/molnextr_markush/runs/markush/molnextr_markush.pth`。

构建数据集使用正式数据集 ID `molnextr_moe_production_v1`，目录在
`training/molnextr_markush/data/generated/pose_factory/`。自合成 attachment 使用 controlled 规则。默认从完整分子切分得到不同
attachment fragment，每个 fragment 只渲染 1 个 variant，数量来自不同化学片段而不是
同一个小 fragment 集反复换样式。宽覆盖的 `synthetic_attachment_fragment` 同时包含
visible `*` 和 wavy endpoint；`synthetic_wavy_fragment`、
`synthetic_attachment_fragment_real_style`、
`synthetic_attachment_fragment_real_style_acyl`、
`synthetic_attachment_fragment_table_style` 和
`synthetic_attachment_fragment_document_style` 也覆盖 visible `*` 和 wavy mark。
straight-open/free-valence endpoint 不再进入训练，因为它不能可靠标明 attachment atom，
会损害泛化。RGReco cut crop 只保留作评估，不进入训练。手写 anchor 模板默认关闭。
训练前会排除重复过多的相同片段、语义标错和堆叠/不清晰样本。请先看分布审计和 review 图，
再开始长训练。
wavy fragment 必须满足生产级短连接线合同：垂直 wavy 和连接线使用与分子键一致的线宽，
连接线可以穿过 wavy 中心，也可以只接触一侧，但异常长的连接线会被 schema 和
MolNexTR 输入质量门禁拒绝，不能靠裁剪掩盖。

自生成化学骨架采用 RDKit-first 渲染，使用黑白 MolDraw2D 风格族来覆盖常见
ChemDraw、Marvin、ACS 和专利输出，并保留低比例 aromatic-circle 风格。自定义绘制只用于
垂直 wavy cut 等 attachment overlay。

`build_dataset.py` 每次都会先清掉上一轮生成产物，再重新构建：
`data/generated/`、`data/dataset/`、`data/literature_eval/`、
`data/rgreco_fragment_eval/`、`data/original_eval/` 和 `runs/markush/`。
这是故意的，防止旧的 attachment crop 在策略改动后继续被误用。

生成训练图必须保持分子 pose，不能做非等比拉伸。`markush_label_boost`、
external ordinary molecules 和所有 attachment bucket 都必须设置
`preserve_aspect_ratio: true`；数据集验证会拒绝没有记录该标记的自生成行。
MG2 reverse-graph 增强在生产训练中禁用。
生成骨架优先使用 RDKit MolDraw2D 黑白绘图；自定义绘制只用于垂直 wavy cut 等
attachment overlay。验证还会拒绝彩色像素、过满裁剪、边缘粘连、长横/竖规则线、
重复或挤在一起的原子坐标、过短/异常长键和交叉键。每次构建都会在
`training/molnextr_markush/runs/markush/visual_review/` 输出按来源分层的样张；
长训练前必须检查这些图，发现结构很假、拥挤、重叠或文档线条主导时要重建数据。

### 部署微调模型

将训练好的权重导出到 Docker runtime 模型目录并重建 web/worker：

```bash
python training/molnextr_markush/scripts/export_checkpoint.py
docker compose up -d --force-recreate web worker
```

或在 `constants.py` 中显式指定路径：

```python
MOLNEXTR_MODEL_PATH = '/app/runtime_models/molnextr_markush/molnextr_markush.pth'
```

完整文档（包括课程学习策略、评估门控、分桶指标和清理说明）请参阅 [`training/molnextr_markush/README.md`](training/molnextr_markush/README.md)。

## Markush/Fragment 输出准确性保障 🛡️

生产中最危险的失败模式是"**组装看起来成功，实际得到错误化合物**"：fragment/骨架的
附加点契约（dummy 存在、数量、位置一致）全部通过，但骨架化学本身是错的（芳香环被
识别为单键、ghost carbon、缺原子），RDKit 组装仍然成功并进入最终输出。以下三层
机制将"混入错误分子"降为零（**宁可排除，不可混入**）：

### 1. 置信度门控（A1）— `constants.py`

MolNexTR MoE 输出校准置信度 `MOLNEXTR_CONFIDENCE`（E[Tanimoto]，见
`utils/MolNexTR/moe_confidence.py`），它是骨架正确性的代理。组装规划
（`utils/markush_assembly.py`）现在对 scaffold 与每个 fragment 施加置信度下限：

- `MARKUSH_ASSEMBLY_MIN_SCAFFOLD_CONFIDENCE`（默认 0.55）
- `MARKUSH_ASSEMBLY_MIN_FRAGMENT_CONFIDENCE`（默认 0.0，禁用 — 实测该头在 fragment 上无分离能力，fragment 精度由 A2 视觉校验承担）

低于下限的组装关系标记为 `low_confidence_scaffold/fragment_backbone` 并 **blocked**
（排除），不再产出组装记录。置信度缺失的行（base 模型/无置信度头）保持旧行为。
阈值由 `evaluation/calibrate_confidence_gate.py` 在真实 hard 集上按
"零错误混入"原则标定。

### 2. 组装后视觉校验（A2）— `pipeline.review_assembled_structures`

RDKit 组装成功后，将组装结果渲染为 2D 图，与 scaffold + fragment 红框图拼成三联图，
交给视觉模型做最终一致性审查（`review_assembled_structure`）：
骨架区、取代基区、附加位置三层全部一致才放行；任何可见矛盾（芳香环变单键、错位连接、
残留 R 标签）→ 该组装被 **blocked** 并记录 `assembled_visual_review_rejected`。
模型调用失败时保持组装不变并写入 `unavailable` 审计（此时 A1 置信度门控仍保护该行）。
每组最大审查数由 `MARKUSH_ASSEMBLY_VISUAL_REVIEW_MAX_ASSEMBLIES` 控制。

### 3. more_dummies 根治（B5，默认关闭）— 解码器图编辑

解码器过度发射 `*` 占 MoE markush 失败 27%。已实现 cardinality head 图编辑（解码 dummy 数超过 head 预测时从图中剪除多余 dummy），但 1195 行real_markushgrapher A/B 实测**回归**：exact_graph_normalized 0.507→0.472（解码器的 dummy 数比 cardinality head 更接近 gold，剪除会删掉正确的附加点）。该编辑保留在 `MOLNEXTR_DUMMY_PRUNE_ENABLED=1` 开关后，供未来 head 改进后启用。

## 置信度路由

MolNexTR 校准备信度低于放行阈值（0.75）的结构会进入 harness 视觉二次确认。
低于审查阈值（0.40）的直接排除，不调用视觉模型。视觉确认失败也会排除
（不 fail-open）。由 `constants.py` 的 `STRUCTURE_CONFIDENCE_*` 常量控制。
