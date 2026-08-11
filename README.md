# BioChemInsight 🧪

**BioChemInsight** is a powerful platform that automates the extraction of chemical structures and their corresponding bioactivity data from scientific literature. By leveraging deep learning for image recognition and OCR, it streamlines the creation of high-quality, structured datasets for cheminformatics, machine learning, and drug discovery research.

![logo](images/BioChemInsight.png)

## Features 🎉

  * **Automated Data Extraction** 🔍: Automatically identifies and extracts compound structures and biological activity data (e.g., IC50, EC50, Ki) from PDF documents.
  * **Advanced Recognition Core** 🧠: Utilizes state-of-the-art DECIMER Segmentation models for image analysis and PaddleOCR for text recognition.
  * **Recommended Visual Model**: For the visual model, it is recommended to use **GLM-V4.5** for optimal results.
  * **Structure Recognition** ⚙️: Uses DECIMER Segmentation plus MolNexTR to convert chemical diagrams into SMILES strings. Includes a fine-tuned Markush model for patent-specific scaffold/fragment recognition.
  * **Automatic Document Planning** 📄: Detects structure pages, bioactivity pages, and assay names automatically, with optional page ranges for constrained runs.
  * **Structured Data Output** 🛠️: Converts unstructured text and images into analysis-ready formats like CSV and Excel.
  * **Modern Web UI** 🌐: A React-based frontend with FastAPI backend for intuitive PDF processing, real-time progress tracking, and interactive result visualization.
  * **Intelligent Data Merging** 🔗: Automatically merges structure and bioactivity data based on compound IDs, providing seamless integrated results.


## Applications 🌟

  * **AI/ML Model Training**: Generate high-quality datasets for training predictive models in cheminformatics and bioinformatics.
  * **Drug Discovery**: Accelerate structure-activity relationship (SAR) studies and lead compound optimization.
  * **Automated Literature Mining**: Drastically reduce the manual effort and time required for curating data from scientific articles.


## Workflow 🚀

BioChemInsight employs a multi-stage pipeline to convert raw PDFs into structured data:

1.  **PDF Preprocessing**: The input PDF is split into individual pages, which are then converted into high-resolution images for analysis.
2.  **Structure Detection**: **DECIMER Segmentation** scans the images to locate and isolate chemical structure diagrams.
3.  **SMILES Conversion**: MolNexTR converts the isolated diagrams into machine-readable SMILES strings.
4.  **Identifier Recognition**: A visual model (recommended: **GLM-4.5V**) recognizes the compound identifiers (e.g., "Compound **1**", "**2a**") associated with each structure.
5.  **Bioactivity Extraction**: **PaddleOCR** extracts text from detected bioactivity pages, and large language models help parse and standardize the bioactivity results.
6.  **Data Integration**: All extracted information—compound IDs, SMILES strings, and bioactivity data—is merged into structured files (CSV/Excel) for download and downstream analysis.


## Installation 🔧

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

#### Step 3: Create and Activate the Conda Environment

```bash
conda install -c conda-forge mamba
mamba create -n chem_ocr python=3.10
conda activate chem_ocr
```

#### Step 4: Install Dependencies

First, install PyTorch with CUDA support.

```bash
# Install CUDA Tools and PyTorch
mamba install -c nvidia -c conda-forge cudatoolkit=11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 -i https://pypi.tuna.tsinghua.edu.cn/simple
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

## Usage 📖

BioChemInsight can be operated via an interactive web interface or directly from the command line.

> **Important:** Start the PaddleOCR microservice before launching the pipeline. Use `DOCKER_PADDLE_OCR` and set `PADDLEOCR_SERVER_URL` in `constants.py`.

### Web Interface 🌐

The modern React-based web interface provides an intuitive platform for processing documents with real-time progress tracking.

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

1.  **PDF Upload**: Upload and manage PDF files through the intuitive interface.
2.  **Automatic Extraction Planning**: Structure pages, bioactivity pages, and assay names are detected automatically; page thumbnails and range inputs remain available for constrained runs.
3.  **Step-by-Step Processing**: 
    - **Step 1**: Upload PDF and preview pages
    - **Step 2**: Extract chemical structures with real-time progress
    - **Step 3**: Extract bioactivity data with structure-constrained compound matching
    - **Step 4**: Review and download merged results
4.  **Real-time Progress Tracking**: Monitor extraction progress with detailed status updates.
5.  **Interactive Results**: View, edit, and download structured data with integrated compound-activity matching.
6.  **Automatic Data Merging**: Seamlessly combines structure and bioactivity data based on compound IDs.

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


## Output 📂

The platform generates the following structured data files in the specified output directory:

  * `structures.csv`: Contains the detected compound identifiers and their corresponding SMILES representations.
  * `assay_data.json`: Stores the raw extracted bioactivity data for each assay.
  * `merged.csv`: A combined file that merges chemical structures with their associated bioactivity data.


## Docker Deployment 🐳

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

`ZENODO_HOST` is optional. It is used only during image build to download the DECIMER molecule segmentation weights from Zenodo with `curl --resolve`, which avoids editing `/etc/hosts` and works with BuildKit. Leave it unset if `zenodo.org` resolves normally in your network.

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

The Docker image pins `numpy==1.26.4` after installing the runtime data-science and RDKit dependencies, and the build checks the installed numpy version before copying project files. This keeps dependency changes from silently upgrading numpy.

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


## Fine-Tuning MolNexTR for Markush Structures 🔬

BioChemInsight includes a fine-tuning pipeline for MolNexTR that targets Markush structures, fragments, substituents, and attachment atoms. The fine-tuned model replaces the base MolNexTR checkpoint at runtime with no code changes, but it must be trained under the current Markush README gates before use.

### Why Fine-Tune?

The base MolNexTR model is trained on ordinary chemical structures and still
needs Markush-specific training data and gates for attachment atoms and label
sets.

### Current Strategy

The maintained training target is BioChemInsight Markush + fragment assembly,
not a PDF-specific patch. MolNexTR must learn the visual evidence for scaffold
labels and fragment attachment atoms directly. Do not fix this with inference
fallback, output rewriting, hard-coded filters, or real-task-specific config
names.

Use the clean default config `training/molnextr_markush/configs/markush.json`.
The maintained dataset strategy uses real complete-molecule pools, currently
PubChem canonical SMILES plus available patent/public molecule pools, as the
main chemistry source. Complete molecules are rendered as ordinary training
anchors, and attachment fragments are derived by cutting those complete
molecules. Ordinary molecules may use canonical SMILES, but controlled
attachment targets must deliberately cover leading, internal, and terminal `*`
serialization. RGReco-style fragments are often leading-star targets such as
`*N...` or `*O...`; excluding that order leaves the original MolNexTR
first-token prior untouched. Hand-built attachment templates are disabled in the
default build, and the build gate fails if they are used.
Training data must cover wavy marks, visible `*`,
acyl/table/document-style fragments, and general
Markush labels such as R, X, Y, Z, Ar, Het, and Hal. The repeat audit and image
review are mandatory before long training runs. External ordinary-molecule data
and eval gates remain active to preserve and improve MolNexTR pose and validity
behavior.

The current root generalization fix is training-side only. Production
checkpoints must start from the original MolNexTR checkpoint, not from an
already fine-tuned Markush checkpoint. The maintained config combines ordinary
structure replay, ordinary-only tail-aware teacher consistency, encoder+decoder
L2-SP regularization, `pad_tail_no_object_loss`, Markush label-set coverage,
narrow Markush-label unlikelihood, and scoped attachment `*` supervision. The
target representation is the primary fix; the losses are guardrails around
greedy decoding and over-generation. The attachment objectives use argmax-margin
supervision at target `*` positions, plus first-step free-run supervision for
leading-star rows, so dummy atoms become the greedy decoding choice instead of
being completed as ordinary terminal chemistry. The unlikelihood is
intentionally limited to Markush-only false positives such as `R`/prime tokens;
do not suppress SMILES digits, brackets, or common atom characters because that
damages original OCR. Do not replace this with
fallback, hard filters, decoder interpolation, or PDF-specific output rewriting.

### Quick Start

The fine-tuning pipeline runs inside Docker and requires ~30 GB of disk space for training data.

```bash
# 1. Build the training image
docker build -f training/molnextr_markush/Dockerfile -t molnextr-markush-train:dev .

# 2. Download datasets (~27 GB)
docker run --rm -v $(pwd):/workspace -w /workspace \
  molnextr-markush-train:dev \
  python training/molnextr_markush/scripts/download_data.py

# 3. Build training dataset
docker run --rm -v $(pwd):/workspace -w /workspace \
  molnextr-markush-train:dev \
  python training/molnextr_markush/scripts/build_dataset.py

# 4. Train (~33 hours on 2× RTX 4070)
docker run --rm --gpus all --shm-size=16g \
  -v $(pwd):/workspace -w /workspace \
  molnextr-markush-train:dev \
  python training/molnextr_markush/scripts/train.py

# 5. Evaluate against the base model
docker run --rm --gpus all -v $(pwd):/workspace -w /workspace \
  molnextr-markush-train:dev \
  python training/molnextr_markush/scripts/evaluate.py
```

The default training config is `training/molnextr_markush/configs/markush.json`.
It trains from `/workspace/models/molnextr_best.pth`, reads
`training/molnextr_markush/data/dataset/train_pose_markush.csv`, and writes
`training/molnextr_markush/runs/markush/molnextr_markush.pth`.

Dataset building uses the production dataset id
`molnextr_moe_production_v1` under
`training/molnextr_markush/data/generated/pose_factory/`. Dataset building uses
controlled self-generated attachment splits. Attachment
rows are derived from distinct fragments cut from complete molecules by default,
with one rendered variant per fragment. The broad
`synthetic_attachment_fragment` bucket covers visible `*` and wavy endpoint
examples. The `synthetic_wavy_fragment`,
`synthetic_attachment_fragment_real_style`,
`synthetic_attachment_fragment_real_style_acyl`,
`synthetic_attachment_fragment_table_style`, and
`synthetic_attachment_fragment_document_style` buckets also cover visible `*`
and wavy marks. Straight-open/free-valence endpoints are excluded because they
do not reliably identify the attachment atom. RGReco cut crops are reserved for
evaluation only. Hand-written anchor templates are disabled by default.
Repeated near-identical fragments, semantically mislabeled samples, and unclear
stacked strokes are excluded before training. Check the attachment-bucket
recalls and the distribution audit before starting a long run.
Wavy fragment rows must follow the production short patent connector contract:
the perpendicular wavy mark and connector use the same stroke width as native
bonds, the connector may cross the wavy center or touch one side, and long
connector stubs are rejected by schema and MolNexTR input-quality gates instead
of being hidden by crop.

Generated chemical skeletons are RDKit-first, using black/white MolDraw2D style
families approximating common ChemDraw, Marvin, ACS, and patent output, plus a
low-frequency aromatic-circle style. Custom drawing is limited to attachment
overlays such as perpendicular wavy cuts. Dataset validation rejects generated
rows with non-preserved aspect ratio, colored pixels, overfilled crops, border
ink, long table/rule lines, duplicate or collapsed atom coordinates, very short
or extreme long bonds, and crossing bonds. Each build also writes stratified
visual review sheets under `training/molnextr_markush/runs/markush/visual_review/`;
inspect them before starting a long training run.

The build step clears previous generated outputs before rebuilding:
`data/generated/`, `data/dataset/`, `data/literature_eval/`,
`data/rgreco_fragment_eval/`, `data/original_eval/`, and `runs/markush/`.
This is deliberate so stale attachment crops do not survive strategy changes.

Generated training images must preserve molecular pose without non-uniform
stretching. `markush_label_boost`, external ordinary molecules, and all
attachment buckets set `preserve_aspect_ratio: true`; dataset validation rejects
generated rows that do not record this flag. MG2 reverse-graph augmentation is
disabled for production training.

### Deploy the Fine-Tuned Model

Export the checkpoint to the Docker runtime model directory and restart:

```bash
python training/molnextr_markush/scripts/export_checkpoint.py
docker compose up -d --force-recreate web worker
```

Or set the path explicitly in `constants.py`:

```python
MOLNEXTR_MODEL_PATH = '/app/runtime_models/molnextr_markush/molnextr_markush.pth'
```

For full documentation including curriculum learning, evaluation gates, per-bucket metrics, and cleanup instructions, see [`training/molnextr_markush/README.md`](training/molnextr_markush/README.md).


### Confidence-based routing

Structures with calibrated MolNexTR confidence below the pass threshold
(0.75) enter visual re-verification through the model harness. Below the
review threshold (0.40) they are excluded without a vision call. Visual
verification failure also excludes the structure (no fail-open).
Controlled by `STRUCTURE_CONFIDENCE_*` constants in `constants.py`.
## Markush/Fragment Output Accuracy Guarantees 🛡️

The most dangerous production failure is "**assembly looks successful but the
compound is wrong**": the fragment/scaffold attachment contract (dummy present,
count, position) passes every check while the backbone chemistry itself is
wrong (aromatic rings read as single bonds, ghost carbons, missing atoms), and
RDKit assembly still succeeds into the final output. Three layers drive the
wrong-inclusion rate to zero (**exclude, never include**):

1. **Confidence gate (A1)** — `utils/markush_assembly.py` blocks assembly when
   the calibrated MoE confidence (`MOLNEXTR_CONFIDENCE`, E[Tanimoto]) of the
   scaffold or any fragment falls below
   `MARKUSH_ASSEMBLY_MIN_SCAFFOLD_CONFIDENCE` (0.40) /
   `MARKUSH_ASSEMBLY_MIN_FRAGMENT_CONFIDENCE` (0.0 — disabled). Thresholds calibrated
   on real patent data (markushgrapher 1195: correct-vs-wrong mean confidence
   0.610 vs 0.485; fragments: conservative garbage filter — the head's fragment
   signal is weak, so the A2 review carries fragment precision).
2. **Post-assembly visual review (A2)** — `pipeline.review_assembled_structures`
   renders the assembled molecule (RDKit 2D) next to the scaffold + fragment
   red-box crops in a three-panel image and asks the vision model to confirm
   scaffold region, fragment region, and attachment position; any visible
   contradiction blocks the assembly (`assembled_visual_review_rejected`).
3. **more_dummies root fix (B5, default OFF)** — the decoder over-emits `*` (27% of MoE markush failures); a cardinality-head graph edit prunes excess dummies, but the 1195-row real_markushgrapher A/B measured a REGRESSION when it fires (exact_graph_normalized 0.507 -> 0.472: the decoder's dummy count is closer to gold than the cardinality head's). The edit stays behind `MOLNEXTR_DUMMY_PRUNE_ENABLED=1` for future head improvements.

**Model training**: `training/molnextr_markush/scripts/run_bond_finetune.sh`
fine-tunes the MoE decoder with aromatic/multiple edge loss weighting.
`training/molnextr_markush/tools/train_confidence_head.py` trains the
calibrated confidence head on a frozen MoE. Detector pseudo-label
analysis is in `build_detector_pseudo_labels.py`.