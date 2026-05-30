# --- Language Model Configuration ---
LLM_OPENAI_COMPATIBLE_MODEL_NAME = 'gpt-4o-mini'
LLM_OPENAI_COMPATIBLE_MODEL_URL = 'https://api.openai.com/v1'
LLM_OPENAI_COMPATIBLE_MODEL_KEY = 'sk-YOUR_OFFICIAL_OPENAI_API_KEY_HERE'


# --- Visual Model Configuration ---
VISUAL_MODEL_NAME = 'gpt-4o'
VISUAL_MODEL_URL = 'https://api.openai.com/v1'
VISUAL_MODEL_KEY = 'sk-YOUR_OFFICIAL_OPENAI_API_KEY_HERE'


# --- Proxies & Other ---
HTTP_PROXY = ''
HTTPS_PROXY = ''
# Zenodo IP override: if DNS cannot resolve zenodo.org (common in China),
# set this to an IP like '188.185.48.75'. Leave empty to use normal DNS.
ZENODO_HOST = ''
# Timeout (seconds) for a single vision model call (structure classification,
# compound-ID recognition, structure auto-detection contact sheets).
VISION_MODEL_TIMEOUT_SECONDS = 90
# Max simultaneous visual-model image requests across structure detection,
# structure filtering, and structure ID extraction.
VISION_MODEL_CONCURRENCY = 4
# Keep visual retries low for long patents. A slow visual server can otherwise
# multiply per-page latency by many minutes.
VISION_MODEL_MAX_RETRIES = 1
VISION_MODEL_OUTER_TIMEOUT_PADDING_SECONDS = 10
# Timeout (seconds) for a single text/LLM model call (content_to_dict,
# compound-ID resolution, assay extraction).
LLM_MODEL_TIMEOUT_SECONDS = 180
# Hard outer guard added around text model calls because some OpenAI-compatible
# servers ignore SDK timeouts after accepting the request.
LLM_MODEL_OUTER_TIMEOUT_PADDING_SECONDS = 10
# Assay extraction LLM calls can hit slow table chunks. Keep retries low so one
# problematic page cannot block a full pipeline for tens of minutes.
ASSAY_EXTRACTION_LLM_MAX_RETRIES = 1
ASSAY_EXTRACTION_LLM_TIMEOUT_SECONDS = 120
# Use model calls scoped to one assay on one OCR page. This keeps prompts small
# and avoids losing multiple assays when a large shared chunk times out.
ASSAY_EXTRACTION_MODE = 'per_assay_page'
ASSAY_EXTRACTION_MAX_PAGE_CANDIDATE_IDS = 96
# Prefer one complete OCR page per model call. Only pages above this budget are
# split by the table/paragraph-aware chunker.
ASSAY_EXTRACTION_MAX_MODEL_CONTENT_CHARS = 12000
# Keep large page-level allowlists out of the first extraction prompt; canonical
# ID enforcement still happens in the compound-ID verifier.
ASSAY_EXTRACTION_PROMPT_MAX_COMPOUND_IDS = 128
# Large tables are split only on complete rows, with table headers repeated.
ASSAY_EXTRACTION_MAX_TABLE_ROWS_PER_CHUNK = 12
ASSAY_EXTRACTION_CHUNK_HEADER_LINES = 8


# --- OCR Engine Configuration ---
# Standalone PaddleOCR queue service endpoint.
# Point this at the host exposing DOCKER_PADDLE_OCR, even if it runs on another machine.
PADDLEOCR_SERVER_URL = 'http://your_paddleocr_server:8010'
# Default OCR language hint. 'auto' lets PaddleOCR use its default multilingual
# behavior; use explicit PaddleOCR language codes only when you need to force one.
PADDLEOCR_LANG = 'auto'

# --- Docker GPU Allocation ---
# These defaults document the intended split for a 2-GPU host:
# - PaddleOCR runs on GPU 0
# - BioChemInsight web/worker runs on GPU 1
# Docker GPU visibility is decided before Python starts, so export these values
# or put them in a Compose .env file before running docker compose.
PADDLEOCR_GPU = '0'
BIOCHEMINSIGHT_GPU = '0'

# Cache shared OCR page text in-memory so multiple assay names reuse one OCR pass.
ASSAY_PAGE_TEXT_CACHE_ENABLED = True
# Keep the cache small to control memory while still reusing repeated page ranges.
ASSAY_PAGE_TEXT_CACHE_MAX_ENTRIES = 4
# Only cache ranges up to this many pages.
ASSAY_PAGE_TEXT_CACHE_MAX_PAGES = 64
# Visually re-read only suspicious OCR assay cells after text extraction.
# This is gated by full OCR table context (assay/table header + same-column
# value domain), so normal numeric assay tables do not call the vision model.
ASSAY_VISUAL_VALUE_REVIEW_ENABLED = True
ASSAY_VISUAL_VALUE_REVIEW_MAX_PAGES = 3
ASSAY_VISUAL_VALUE_REVIEW_RENDER_SCALE = 2.0
ASSAY_VISUAL_VALUE_REVIEW_MAX_WIDTH = 1400
# Split large visual review plans into bounded vision prompts. This keeps the
# model-owned visual reread path while avoiding one huge image+JSON prompt.
ASSAY_VISUAL_VALUE_REVIEW_MAX_ITEMS_PER_CALL = 20
# Assay-page auto-detection uses PaddleOCR markdown plus the configured text
# model and a skill prompt.
# OCR requests are page-range batches. By default each batch is first written
# to a small temporary PDF, so concurrent requests upload only their own pages
# instead of re-uploading the full patent PDF.
ASSAY_AUTO_DETECT_OCR_SPLIT_PDF = True
ASSAY_AUTO_DETECT_OCR_BATCH_SIZE = 3
ASSAY_AUTO_DETECT_OCR_CONCURRENCY = 1
ASSAY_AUTO_DETECT_OCR_TIMEOUT_SECONDS = 360
# If a batched OCR request times out, split it into smaller page groups and retry.
ASSAY_AUTO_DETECT_OCR_SPLIT_RETRY_ENABLED = True
ASSAY_AUTO_DETECT_LLM_BATCH_SIZE = 6
ASSAY_AUTO_DETECT_LLM_MAX_PAGE_CHARS = 5000
ASSAY_AUTO_DETECT_LLM_MAX_RETRIES = 1
ASSAY_AUTO_DETECT_LLM_TIMEOUT_SECONDS = 120

# --- Structure Candidate Filtering ---
# When True, a visual model first classifies each segmented image and only complete compounds
# continue to downstream compound-ID recognition and assay matching.
STRUCTURE_FILTER_ENABLED = True
# When True, filtered-out Markush / fragment / noise / uncertain segments are exported too.
SAVE_FILTERED_STRUCTURES = True
# Filtering strictness:
# - 'strict': safest default; suspicious border-touching candidates are withheld
# - 'balanced': use model + review checks, but skip the final unconditional border holdback
# - 'permissive': rely mostly on the first-pass model judgment
STRUCTURE_FILTER_STRICTNESS = 'strict'
# Optional structure runtime limits. Leave as 0 to auto-tune from available memory.
STRUCTURE_PAGE_WORKERS = 0
STRUCTURE_PAGE_MAX_INFLIGHT = 0
STRUCTURE_ID_BATCH_SIZE = 0
# Optional upper bound for concurrent structure-ID requests. 0 means auto (about 2x batch size).
STRUCTURE_ID_MAX_INFLIGHT = 0
# Run a second visual pass after structure Compound ID recognition to verify
# row/label attachment and reject partial or borrowed IDs.
STRUCTURE_ID_VERIFIER_ENABLED = True
# Structure extraction safeguards. If a page or model call stalls, skip it
# quickly instead of blocking an entire patent for 10+ minutes per page.
STRUCTURE_MODEL_TIMEOUT_SECONDS = 180
STRUCTURE_MOLECULE_PROCESSING_TIMEOUT_SECONDS = 60
STRUCTURE_PAGE_PROCESSING_TIMEOUT_SECONDS = 240
# MolNexTR graph-to-SMILES post-processing workers. Keep at 1 for web tasks:
# per-segment multiprocessing can stall threaded API workers and accumulate
# zombie child processes.
MOLNEXTR_POSTPROCESS_WORKERS = 1
# Upscale small structure crops to this long edge before MolNexTR's fixed
# 384x384 inference transform. Set to 0 to disable this pre-upscaling.
MOLNEXTR_PREPROCESS_LONG_EDGE = 512
# Long-running task concurrency for the Redis/Celery deployment. The same
# settings can be overridden from docker-compose.yml or Compose environment.
MAX_CONCURRENT_TASKS = 3
DISPATCHER_MAX_CONCURRENT_TASKS = 3
CELERY_WORKER_CONCURRENCY = 3
QUEUE_DISPATCHER_POLL_SECONDS = 1
# While blocking OCR/model/extraction steps run in worker helper threads, the
# Celery task thread checks cancellation and refreshes updated_at on this cadence.
TASK_STEP_HEARTBEAT_SECONDS = 10
# Optional outer timeout for one blocking task step. 0 disables the guard; use
# a large value such as 3600 if you want stuck steps to fail and free a slot.
TASK_STEP_TIMEOUT_SECONDS = 0
# Celery keeps the lightweight threads pool, but each heavy BioChemInsight job
# runs in its own child process. Canceling a running task terminates only that
# child process instead of restarting the whole worker.
TASK_CHILD_POLL_SECONDS = 2
TASK_CHILD_TERMINATE_GRACE_SECONDS = 20
# Optional hard wall-clock timeout for one queued task child process. 0 disables
# it; set a large value such as 7200 to fail and kill runaway jobs.
TASK_CHILD_TIMEOUT_SECONDS = 0
# Recovery guard for Docker restarts: if Redis says a task is inflight but
# Celery has no active/reserved copy for this many seconds, requeue it.
# Set to 0 to disable.
QUEUE_DISPATCHER_STALE_RUNNING_SECONDS = 300
# Prevent duplicate Celery deliveries of the same BioChemInsight task id from
# running concurrently after Docker/Redis recovery. Dispatcher clears stale
# locks when it has verified that Celery no longer has the task active.
QUEUE_TASK_EXECUTION_LOCK_SECONDS = 86400
# Structure extraction often uses the GPU heavily. Keep this aligned with
# available GPU memory; docker-compose.yml starts with 2 by default.
STRUCTURE_TASK_CONCURRENCY = 2
# PyTorch CUDA allocator tuning for long patent runs.
PYTORCH_CUDA_ALLOC_CONF = 'max_split_size_mb:64,garbage_collection_threshold:0.8'
# Structure-page auto-detection uses the configured visual model on low-memory
# contact sheets of page thumbnails.
STRUCTURE_AUTO_DETECT_VISION_BATCH_SIZE = 12
STRUCTURE_AUTO_DETECT_VISION_COLUMNS = 3
STRUCTURE_AUTO_DETECT_VISION_RENDER_SCALE = 0.7
STRUCTURE_AUTO_DETECT_VISION_THUMB_WIDTH = 420
STRUCTURE_AUTO_DETECT_VISION_MAX_RETRIES = 1
# Optional strict visual second pass. The first pass keeps recall on compact
# contact sheets; this review pass rerenders only candidate pages larger and
# rejects text-only patent pages/tables using the visual model itself.
STRUCTURE_AUTO_DETECT_VISION_REVIEW_ENABLED = True
STRUCTURE_AUTO_DETECT_VISION_REVIEW_BATCH_SIZE = 4
STRUCTURE_AUTO_DETECT_VISION_REVIEW_COLUMNS = 2
STRUCTURE_AUTO_DETECT_VISION_REVIEW_RENDER_SCALE = 1.0
STRUCTURE_AUTO_DETECT_VISION_REVIEW_THUMB_WIDTH = 700
# Cache document-level auto-detect results for repeated runs on the same PDF.
DOCUMENT_AUTO_DETECT_CACHE_ENABLED = True
# Leave empty to use /tmp/biocheminsight_auto_detect_cache
DOCUMENT_AUTO_DETECT_CACHE_DIR = ''

# --- Thread & BLAS Limits ---
# These prevent BLAS/MKL/OpenMP thread oversubscription when PyTorch, TensorFlow,
# and numpy run in the same process. Set via environment variables (docker-compose
# or Dockerfile ENV). The values below document the defaults baked into the
# Dockerfile; override them in docker-compose.yml if needed.
# OMP_NUM_THREADS = 4
# MKL_NUM_THREADS = 4
# OPENBLAS_NUM_THREADS = 4
# NUMEXPR_NUM_THREADS = 4
# TOKENIZERS_PARALLELISM = false
# TF_NUM_INTRAOP_THREADS = 2
# TF_NUM_INTEROP_THREADS = 2
