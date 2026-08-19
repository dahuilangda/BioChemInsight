# BioChemInsight

从科技文献 PDF（专利与期刊论文）中提取化学结构与生物活性数据，合并输出结构化的 CSV/Excel 数据集。

![logo](images/BioChemInsight.jpeg)

## 功能

- 自动检测结构页、活性数据页与实验名称；也支持指定页码范围。
- 化学结构图转 SMILES（DECIMER 分割 + MolNexTR 混合专家，含 Markush/片段旁路专家与置信度视觉复核）；Markush 骨架与分离片段自动组装为完整分子。
- 视觉模型识别每个结构的化合物编号（推荐 **GLM-4.5V**；任何 OpenAI 兼容模型可在 `constants.py` 中通过 `VISUAL_MODEL_NAME` / `VISUAL_MODEL_URL` / `VISUAL_MODEL_KEY` 配置）。
- PaddleOCR + 语言模型提取活性数值（IC50、EC50、Ki 等）。只保留实验测量值，排除 docking 打分与计算预测值。
- 支持文献系列模式（结构只画一次、活性按成员列出）：系列编号展开到成员级，成员结构经 PubChem/OPSIN 从名称解析。
- 结构与活性数据按化合物编号合并。
- 提供 Web 界面（React + FastAPI，实时进度）与命令行批量处理。

## 安装

### Docker（推荐）

需要支持 GPU 的 Docker（NVIDIA Container Toolkit）；构建时自动下载约 2 GB 模型权重。

```bash
git clone https://github.com/dahuilangda/BioChemInsight
cd BioChemInsight
mv constants_example.py constants.py   # 编辑 API key 与模型端点
mkdir -p data output frontend/backend/data
docker compose up --build -d
```

- 界面：`http://localhost:3000` — API：`http://localhost:8000`
- PaddleOCR 为独立微服务：在 `DOCKER_PADDLE_OCR` 中构建，并在 `constants.py` 设置 `PADDLEOCR_SERVER_URL`。
- 可选构建参数：`ZENODO_HOST`
- 如需运行与镜像内置不同的 MolNexTR 权重，挂载后设置 `MOLNEXTR_MODEL_PATH` 为完整文件路径。（zenodo.org 不可达时代理下载 DECIMER 权重）。`APP_UID`/`APP_GID` 是入口脚本的运行时环境变量，不是构建参数。

### 手动安装

```bash
conda create -n chem_ocr python=3.12
conda activate chem_ocr

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129
pip install SmilesPE opencv-python-headless PyMuPDF PyPDF2 openai Levenshtein \
    mdutils tabulate python-multipart fastapi uvicorn celery redis huggingface_hub py2opsin
mamba install -c conda-forge jupyter pytesseract transformers
sudo apt-get install -y redis-server nodejs   # macOS: brew install redis node
```

权重托管在 HuggingFace 数据集 [dahuilangda/BioChemInsight](https://huggingface.co/datasets/dahuilangda/BioChemInsight)（GitHub 仓库不含权重）：

```bash
huggingface-cli download dahuilangda/BioChemInsight --repo-type dataset \
    --local-dir /tmp/bci_weights --local-dir-use-symlinks False
mkdir -p models experiments/moe/production
mv /tmp/bci_weights/molnextr_best.pth models/
mv /tmp/bci_weights/moe/* experiments/moe/production/
```

# 国内镜像：下载前 export HF_ENDPOINT=https://hf-mirror.com


| 文件 | 大小 | 目标路径 |
|------|------|----------|
| `molnextr_best.pth` | 1.1 GB | `models/molnextr_best.pth` |
| `moe/moe_encoder.pth` | 322 MB | `experiments/moe/production/` |
| `moe/moe_expert1.pth` | 32 MB | `experiments/moe/production/` |
| `moe/moe_expert2.pth` | 32 MB | `experiments/moe/production/` |
| `moe/moe_router.pt` | 38 MB | `experiments/moe/production/` |
| `moe/moe_confidence.pt` | 1.3 MB | `experiments/moe/production/` |
| `moe/moe_config.json` | < 1 MB | `experiments/moe/production/` |

## 使用

### Web 界面

Docker Compose 已包含全部服务。本地开发需分别启动五个进程：

```bash
export REDIS_URL=redis://localhost:6379/0                         # Docker 外运行必须设置（默认主机名为 redis）
redis-server                                                    # 1
uvicorn frontend.backend.main:app --host 0.0.0.0 --port 8000    # 2
python -m frontend.backend.queue_dispatcher                     # 3
celery -A frontend.backend.celery_app.celery_app worker -Q compute \
  --pool threads --concurrency 2 --loglevel INFO                # 4
cd frontend/ui && npm install && npm run dev                    # 5 -> http://localhost:5173
```

上传 PDF，确认自动检测的页面（或手动指定范围），运行流水线，下载合并结果。

### 命令行

```bash
# 全自动：自动检测结构页、活性页与实验名称
python pipeline.py data/sample.pdf --output output

# 指定范围
python pipeline.py data/sample.pdf --structure-pages "242-250,255" --output output
python pipeline.py data/sample.pdf --structure-pages "242-267" \
    --assay-pages "30,35,270-272" --assay-names "IC50,FRET EC50" --output output

# Docker 内运行
docker run --rm --gpus all -e http_proxy= -e https_proxy= \
    -v $(pwd)/data:/app/data -v $(pwd)/output:/app/output \
    --entrypoint python biocheminsight \
    pipeline.py data/sample.pdf --output output
```

### REST API

```bash
API=http://localhost:8000/api

PDF_ID=$(curl -s -X POST "$API/pdfs" -F "file=@data/sample.pdf" \
  | python -c 'import json,sys; print(json.load(sys.stdin)["pdf_id"])')

TASK_ID=$(curl -s -X POST "$API/tasks/full-pipeline" \
    -H "Content-Type: application/json" \
    -d "{\"pdf_id\":\"$PDF_ID\",\"structure_filter_strictness\":\"strict\",\"lang\":\"en\"}" \
  | python -c 'import json,sys; print(json.load(sys.stdin)["task_id"])')

curl -s "$API/tasks/$TASK_ID" | python -m json.tool          # 轮询直到 completed
curl -L "$API/tasks/$TASK_ID/download" -o results.zip   # ZIP：含合并 CSV、结构、活性、审计
```

结构与活性也可分别提交（`POST /api/tasks/structures`、`POST /api/tasks/assays`），支持取消（`POST /api/tasks/{id}/cancel`）。

## 输出

- `structures.csv` — 化合物编号、SMILES、molblock 与证据列。
- `{assay}_assay_data.json`（CLI）/ `assays.csv`（Web 后端）— 各实验的活性数值。
- `merged.csv` — 结构与活性按化合物编号合并。

## 微调 MolNexTR（Markush MoE）

`experiments/moe/production/` 下的权重开箱即用。重新训练：

```bash
docker build -f training/molnextr_markush/Dockerfile -t molnextr-markush-train:dev .
docker run --rm -v $(pwd):/workspace -w /workspace molnextr-markush-train:dev \
  python training/molnextr_markush/scripts/download_data.py          # 约 27 GB
docker run --rm --gpus all --shm-size=16g -v $(pwd):/workspace -w /workspace \
  molnextr-markush-train:dev \
  bash training/molnextr_markush/scripts/run_moe_production.sh --stage all
```

阶段：`generate -> qc -> build-data -> train -> eval`。产物 `moe_*` 复制到 `experiments/moe/production/` 后重启服务；`constants.py` 中设 `MOLNEXTR_MOE_CONFIG_PATH = ''` 可停用旁路专家。完整参考：[`training/molnextr_markush/README.md`](training/molnextr_markush/README.md)。
