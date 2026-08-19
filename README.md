# BioChemInsight

BioChemInsight extracts chemical structures and bioactivity data from scientific literature PDFs and merges them into structured datasets for cheminformatics and drug discovery.

![logo](images/BioChemInsight.png)

## Features

- Structure and bioactivity extraction (IC50, EC50, Ki, ...) from PDF documents.
- DECIMER Segmentation locates structure diagrams; PaddleOCR reads text.
- A Mixture-of-Experts extension of MolNexTR converts diagrams to SMILES, with sidecar experts for Markush scaffolds, fragments, and attachment atoms.
- Visual model recognizes compound identifiers per structure. Recommended: **GLM-4.5V**; any OpenAI-compatible vision model works via `VISUAL_MODEL_NAME` / `VISUAL_MODEL_URL` / `VISUAL_MODEL_KEY` in `constants.py` (`constants_example.py` ships a `gpt-4o` example).
- Automatic document planning: structure pages, bioactivity pages, and assay names are detected automatically; explicit page ranges are supported.
- Structure and bioactivity results are merged on compound IDs into CSV/Excel.
- React + FastAPI web UI with progress tracking and result visualization.

## Workflow

1. **Preprocessing**: PDF pages are rendered to high-resolution images.
2. **Structure detection**: DECIMER Segmentation isolates chemical diagrams.
3. **SMILES conversion**: the MolNexTR MoE (attention-pooled router over complete / Markush / fragment experts) decodes diagrams; a calibrated confidence head triggers visual re-verification of low-confidence predictions. Markush scaffolds with detached fragments are assembled via dummy-atom attachment and visually confirmed.
4. **Identifier recognition**: the visual model reads compound identifiers attached to each structure.
5. **Bioactivity extraction**: PaddleOCR plus language models parse and standardize assay values; only experimentally measured values are kept.
6. **Integration**: compound IDs, SMILES, and assay values merge into CSV/Excel.

## Installation

#### Step 1: Clone the Repository

```bash
git clone https://github.com/dahuilangda/BioChemInsight
cd BioChemInsight
```

#### Step 2: Configure Constants

The project requires a `constants.py` file for environment variables and paths. A template is provided.

```bash
# Rename the example file
mv constants_example.py constants.py
```

Then, edit `constants.py` to set your API keys, model paths, and other necessary configurations.

#### Step 3: Download Model Weights

Download the pre-trained model weights from [HuggingFace](https://huggingface.co/datasets/dahuilangda/BioChemInsight):

```bash
# Option A: Using huggingface-cli (recommended)
pip install huggingface_hub
huggingface-cli download dahuilangda/BioChemInsight \
    --repo-type dataset \
    --local-dir . \
    --local-dir-use-symlinks False
```

This downloads all weights into the correct directory structure:

| File | Size | Target Path | Description |
|------|------|-------------|-------------|
| `molnextr_best.pth` | 1.1 GB | `models/molnextr_best.pth` | MolNexTR base model |
| `moe/moe_encoder.pth` | 322 MB | `experiments/moe/production/moe_encoder.pth` | MoE shared encoder |
| `moe/moe_expert1.pth` | 32 MB | `experiments/moe/production/moe_expert1.pth` | Markush sidecar expert |
| `moe/moe_expert2.pth` | 32 MB | `experiments/moe/production/moe_expert2.pth` | Fragment sidecar expert |
| `moe/moe_router.pt` | 38 MB | `experiments/moe/production/moe_router.pt` | MoE attention router |
| `moe/moe_confidence.pt` | 1.3 MB | `experiments/moe/production/moe_confidence.pt` | Confidence head |
| `moe/moe_config.json` | < 1 MB | `experiments/moe/production/moe_config.json` | MoE deployment config |

For users in China, set the HF mirror before downloading:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

> **Docker users**: The Dockerfile downloads all weights automatically during `docker build` — skip this step.

##### DECIMER Segmentation weights (manual install only)

Structure detection also needs the DECIMER Mask R-CNN weights
(`models/mask_rcnn_molecule.pth`, ~244 MB). These are **not** on HuggingFace —
they are fetched from [Zenodo](https://zenodo.org/records/10663579) as a
Keras `.h5` and converted to a PyTorch `.pth`:

```bash
# 1. Download the h5 from Zenodo (set the proxy if Zenodo is unreachable)
curl -L -o /tmp/mask_rcnn_molecule.h5 \
    "https://zenodo.org/records/10663579/files/mask_rcnn_molecule.h5?download=1"

# 2. Convert h5 -> pth
python -c "from utils.convert_decimer_weights import convert_weights; \
           convert_weights('/tmp/mask_rcnn_molecule.h5','models/mask_rcnn_molecule.pth')"
```

Docker builds do this automatically (see the `DECIMER_WEIGHTS_URL` / optional
`ZENODO_HOST` handling in the Dockerfile).

#### Step 4: Create and Activate the Conda Environment

> The supported reference environment is the Docker image (Python 3.12, CUDA 12.9.1). The manual install below mirrors it.

```bash
conda install -c conda-forge mamba
mamba create -n chem_ocr python=3.12
conda activate chem_ocr
```

#### Step 5: Install Dependencies

First, install PyTorch with CUDA support (cu129 wheels, matching the Docker image).

```bash
# Install PyTorch (CUDA 12.9 build)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

Next, install the remaining Python packages.

```bash
# Install core libraries (using a mirror for faster downloads)
pip install SmilesPE opencv-python-headless -i https://pypi.tuna.tsinghua.edu.cn/simple
mamba install -c conda-forge jupyter pytesseract transformers
pip install PyMuPDF PyPDF2 openai Levenshtein mdutils tabulate python-multipart -i https://pypi.tuna.tsinghua.edu.cn/simple

# Install web service dependencies
pip install fastapi uvicorn celery redis -i https://pypi.tuna.tsinghua.edu.cn/simple

# Install Redis server plus Node.js/npm (for the async web UI)
# On Ubuntu/Debian:
sudo apt-get install -y redis-server
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs

# On macOS (using homebrew):
# brew install redis node
```

## Usage

BioChemInsight can be operated via an interactive web interface or directly from the command line.

> **Important:** Start the PaddleOCR microservice before launching the pipeline. Use `DOCKER_PADDLE_OCR` and set `PADDLEOCR_SERVER_URL` in `constants.py`.

### Web Interface

#### Launch the Web Service Locally

The web UI uses asynchronous jobs. A local development deployment therefore needs **five** running processes:

1. Redis, used by the task registry, queue state, Celery broker, and result backend.
2. FastAPI backend.
3. Queue dispatcher, which moves queued BioChemInsight jobs into Celery.
4. Celery worker, which executes extraction jobs.
5. Vite frontend development server.

If you do not want to manage these processes manually, use the Docker Compose deployment below.

**Step 1: Start Redis**

On Ubuntu/Debian:

```bash
sudo systemctl start redis-server
# or, for a foreground development process:
redis-server
```

On macOS with Homebrew:

```bash
brew services start redis
# or:
redis-server
```

BioChemInsight defaults to `redis://localhost:6379/0`. Override it if needed:

```bash
export REDIS_URL=redis://localhost:6379/0
export CELERY_BROKER_URL=$REDIS_URL
export CELERY_RESULT_BACKEND=$REDIS_URL
```

**Step 2: Start the Backend API Server**

From the project root directory, run:

```bash
uvicorn frontend.backend.main:app --host 0.0.0.0 --port 8000 --reload
```

**Step 3: Start the Queue Dispatcher**

In a new terminal from the project root:

```bash
python -m frontend.backend.queue_dispatcher
```

**Step 4: Start the Celery Worker**

In another terminal from the project root:

```bash
celery -A frontend.backend.celery_app.celery_app worker \
  -Q compute \
  --pool threads \
  --concurrency 2 \
  --prefetch-multiplier 1 \
  --loglevel INFO
```

You can tune local concurrency with:

```bash
export MAX_CONCURRENT_TASKS=2
export DISPATCHER_MAX_CONCURRENT_TASKS=2
export CELERY_WORKER_CONCURRENCY=2
export STRUCTURE_TASK_CONCURRENCY=2
```

**Step 5: Start the Frontend Development Server**

In a new terminal, run:

```bash
cd frontend/ui
npm install
NODE_OPTIONS="--max-old-space-size=8196" npm run dev
```

**Step 6: Access the Interface**

Open `http://localhost:5173` in your web browser to access the interface. The backend API will be available at `http://localhost:8000`.

#### Web Interface Features

1. PDF upload with page preview.
2. Automatic structure-page, assay-page, and assay-name detection; thumbnails and page-range inputs for constrained runs.
3. Step-by-step processing: upload → structures → bioactivity → merged results.
4. Live progress tracking; view, edit, and download results; structure and bioactivity data merged on compound IDs.

### Command-Line Interface (CLI)

For batch processing and automation, the CLI is recommended.

#### Automatic Extraction (Recommended)

Run the pipeline without page or assay-name arguments to let BioChemInsight plan structure and bioactivity extraction automatically.

```bash
python pipeline.py data/sample.pdf \
    --output output
```

**Constrained Run Examples:**

  * **Extract structures from selected pages:**
    ```bash
    python pipeline.py data/sample.pdf --structure-pages "242-250,255,260-267" --output output
    ```
  * **Extract selected bioactivity pages and assays:**
    ```bash
    python pipeline.py data/sample.pdf --structure-pages "242-267" --assay-pages "30,35,270-272" --assay-names "IC50,FRET EC50" --output output
    ```

## Output

Files written to the output directory:

- `structures.csv` — compound identifiers and SMILES.
- `assays.csv`, `*_assay_data.json` — extracted bioactivity values per assay.
- `merged.csv` — structures joined with bioactivity on compound IDs.

## Docker Deployment

Deploy BioChemInsight in a containerized environment for consistency and portability.

#### Recommended: Docker Compose

The Docker Compose deployment includes:
- `web`: FastAPI + React UI.
- `redis`: durable task registry and queue state.
- `dispatcher`: queue dispatcher.
- `worker`: Celery executor.

Start the stack:

```bash
docker compose up --build -d
```

The container entrypoint detects the owner of the bind-mounted host directories and runs the application as that UID/GID. You normally do not need to set `APP_UID` or `APP_GID`; set them only if you need to override the detected user:

```bash
APP_UID=1000
APP_GID=1000
ZENODO_HOST=188.185.48.75
```

`ZENODO_HOST` is optional; it is used only during image build to download the DECIMER molecule segmentation weights from Zenodo. Leave it unset if `zenodo.org` resolves normally in your network.

Before starting containers, create the host bind-mount directories if they do not exist. The `web` container starts as root only long enough to make `output` and `frontend/backend/data` writable by the detected runtime UID/GID, then drops privileges before running the backend and frontend:

```bash
mkdir -p data output frontend/backend/data
```

Tune concurrency in `docker-compose.yml` or a Compose `.env` file:

```bash
MAX_CONCURRENT_TASKS=3
DISPATCHER_MAX_CONCURRENT_TASKS=3
CELERY_WORKER_CONCURRENCY=3
STRUCTURE_TASK_CONCURRENCY=2
```

The Compose network uses `172.200.0.0/16`, not Docker's usual `172.17.*` bridge range.

The Docker image pins `numpy==1.26.4` after installing the runtime data-science and RDKit dependencies.

After launching, access the UI by visiting:
- Frontend: `http://localhost:3000`
- Backend API: `http://localhost:8000`

Check services and logs:

```bash
docker compose ps
docker compose logs --tail 100 web worker dispatcher redis
```

Redis may print a host-kernel warning about `vm.overcommit_memory`. The stack can run with this warning, but long-running deployments should enable it on the host:

```bash
sudo sysctl vm.overcommit_memory=1
echo 'vm.overcommit_memory=1' | sudo tee /etc/sysctl.d/99-redis-overcommit.conf
sudo sysctl --system
docker compose restart redis
```

#### Submit Jobs with `curl`

The Docker Compose deployment exposes the same FastAPI backend used by the web UI, so jobs can also be submitted programmatically with `curl`.

Recommended path: use the **full automatic pipeline**. It detects structure pages, bioactivity pages, and assay names automatically.

```bash
API=http://localhost:8000/api

# 1) Upload a PDF and capture the returned pdf_id.
PDF_ID=$(
  curl -s -X POST "$API/pdfs" \
    -F "file=@data/sample.pdf" \
  | python -c 'import json,sys; print(json.load(sys.stdin)["pdf_id"])'
)
echo "$PDF_ID"

# 2) Submit the recommended full automatic pipeline and capture task_id.
TASK_ID=$(
  curl -s -X POST "$API/tasks/full-pipeline" \
    -H "Content-Type: application/json" \
    -d "{\"pdf_id\":\"$PDF_ID\",\"structure_filter_strictness\":\"strict\",\"lang\":\"en\"}" \
  | python -c 'import json,sys; print(json.load(sys.stdin)["task_id"])'
)
echo "$TASK_ID"

# 3) Poll task status until "status" is "completed".
curl -s "$API/tasks/$TASK_ID" | python -m json.tool

# 4) Download the result CSV.
curl -L "$API/tasks/$TASK_ID/download" -o result.csv
```

Optional advanced usage: run structure and bioactivity extraction separately.

```bash
API=http://localhost:8000/api

# Structure extraction from explicit pages.
# Omit pages or set auto_detect_pages=true for automatic structure-page detection.
STRUCTURE_TASK_ID=$(
  curl -s -X POST "$API/tasks/structures" \
    -H "Content-Type: application/json" \
    -d "{\"pdf_id\":\"$PDF_ID\",\"pages\":\"1,3,5-7\",\"structure_filter_strictness\":\"strict\"}" \
  | python -c 'import json,sys; print(json.load(sys.stdin)["task_id"])'
)

# Bioactivity extraction constrained by a completed structure task.
ASSAY_TASK_ID=$(
  curl -s -X POST "$API/tasks/assays" \
    -H "Content-Type: application/json" \
    -d "{\"pdf_id\":\"$PDF_ID\",\"pages\":\"10-12\",\"assay_names\":[\"IC50\",\"EC50\"],\"structure_task_id\":\"$STRUCTURE_TASK_ID\",\"lang\":\"en\"}" \
  | python -c 'import json,sys; print(json.load(sys.stdin)["task_id"])'
)

# Cancel a queued/running task if needed.
curl -s -X POST "$API/tasks/$ASSAY_TASK_ID/cancel" | python -m json.tool
```

#### Command-Line Pipeline in Docker

If you need to execute a batch job using the CLI, override the default entrypoint by specifying `python pipeline.py` and its arguments after the `docker run` command.

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

#### Interactive Container Session

To debug or run commands manually inside the container:

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

## Fine-Tuning MolNexTR for Markush Structures

BioChemInsight ships a Mixture-of-Experts (MoE) extension of MolNexTR for
Markush structures, fragments, substituents, and attachment atoms. The MoE
**augments** (not replaces) the base MolNexTR checkpoint: a frozen base decoder
serves as the *complete-molecule* expert (expert 0), while two trained sidecar
experts specialize in *Markush scaffolds* (expert 1) and *fragments* (expert 2).
An attention-pooled router blends the experts per depiction, and a calibrated
confidence head estimates expected graph-level Tanimoto similarity.

The MoE is already wired into production via
`MOLNEXTR_MOE_CONFIG_PATH = 'experiments/moe/production/moe_config.json'`
in `constants.py`. The weights under `experiments/moe/production/` are the
shipped, pre-trained production checkpoints — no fine-tuning is required to use
BioChemInsight. The steps below are only for retraining from scratch.

### Quick Start

Training is orchestrated by `run_moe_production.sh` and runs inside Docker.
It requires ~30 GB of disk for generated training data.

```bash
# 1. Build the training image
docker build -f training/molnextr_markush/Dockerfile -t molnextr-markush-train:dev .

# 2. Download source datasets (~27 GB)
docker run --rm -v $(pwd):/workspace -w /workspace \
  molnextr-markush-train:dev \
  python training/molnextr_markush/scripts/download_data.py

# 3. Run the production MoE pipeline stage-by-stage:
#    generate -> qc -> build-data -> train -> eval
docker run --rm --gpus all --shm-size=16g \
  -v $(pwd):/workspace -w /workspace \
  molnextr-markush-train:dev \
  bash training/molnextr_markush/scripts/run_moe_production.sh --stage all
```

Individual stages can be run on their own (useful for iteration):

```bash
bash training/molnextr_markush/scripts/run_moe_production.sh --stage generate   # render training data
bash training/molnextr_markush/scripts/run_moe_production.sh --stage qc         # visual review sheets / audits
bash training/molnextr_markush/scripts/run_moe_production.sh --stage build-data # aggregate into train DataFrame
bash training/molnextr_markush/scripts/run_moe_production.sh --stage train --ddp-gpus 2   # ~33h on 2× RTX 4070
bash training/molnextr_markush/scripts/run_moe_production.sh --stage eval       # evaluate vs base model
```

Training uses the production dataset id `molnextr_moe_production_v1`. The
aggregated training DataFrame is cached at
`experiments/moe/molnextr_moe_production_v1_train_df.parquet`. Generated
chemistry is RDKit-first (ChemDraw/Marvin/ACS/patent drawing styles, pose
preserved); the QC stage writes stratified visual review sheets under
`training/molnextr_markush/runs/` — inspect them before a long run. See
[`training/molnextr_markush/README.md`](training/molnextr_markush/README.md)
for the full stage reference, hyperparameters, and per-bucket metrics.

### Deploy the Fine-Tuned MoE

Training writes the MoE artifact set directly into the run directory:

```
moe_encoder.pth      # shared encoder (fine-tuned)
moe_expert1.pth      # Markush sidecar expert
moe_expert2.pth      # fragment sidecar expert
moe_router.pt        # attention-pooled router
moe_confidence.pt    # calibrated E[Tanimoto] confidence head
moe_config.json      # deployment config (expert layout, routing, thresholds)
```

To deploy, copy these six files into `experiments/moe/production/` (the path
`MOLNEXTR_MOE_CONFIG_PATH` already points there by default in `constants.py`)
and restart the services:

```bash
cp training/molnextr_markush/runs/<run>/moe_*.pth experiments/moe/production/
cp training/molnextr_markush/runs/<run>/moe_*.pt  experiments/moe/production/
cp training/molnextr_markush/runs/<run>/moe_config.json experiments/moe/production/
docker compose up -d --force-recreate web worker
```

The base `models/molnextr_best.pth` must remain in place — expert 0 (the
complete-molecule expert) reuses it. To fall back to base-only MolNexTR (no
Markush/fragment sidecars), set `MOLNEXTR_MOE_CONFIG_PATH = ''` in
`constants.py`.