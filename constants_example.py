# --- General API Keys ---
# If you don't set GEMINI, you must configure language and visual models below
GEMINI_API_KEY = 'YOUR_GEMINI_API_KEY_HERE'
GEMINI_MODEL_NAME = 'gemini-2.0-flash'


# --- Language Model Configuration ---
# # If not configured, use GEMINI as the language model
# LLM_OPENAI_COMPATIBLE_MODEL_NAME = 'gpt-4o-mini'
# LLM_OPENAI_COMPATIBLE_MODEL_URL = 'https://api.openai.com/v1'
# LLM_OPENAI_COMPATIBLE_MODEL_KEY = 'sk-YOUR_OFFICIAL_OPENAI_API_KEY_HERE'


# --- Visual Model Configuration ---
# # If not configured, use GEMINI for visual tasks
# VISUAL_MODEL_TYPE = 'openai'
# VISUAL_MODEL_NAME = 'gpt-4o'
# VISUAL_MODEL_URL = 'https://api.openai.com/v1'
# VISUAL_MODEL_KEY = 'sk-YOUR_OFFICIAL_OPENAI_API_KEY_HERE'


# --- Proxies & Other ---
HTTP_PROXY = ''
HTTPS_PROXY = ''
MOLVEC = '/path/to/your/BioChemInsight/bin/molvec-0.9.9-SNAPSHOT-jar-with-dependencies.jar'

# Timeout (seconds) for a single vision model call (structure classification,
# compound-ID recognition, structure auto-detection contact sheets).
VISION_MODEL_TIMEOUT_SECONDS = 90
# Max simultaneous visual-model image requests across structure detection,
# structure filtering, and structure ID extraction.
VISION_MODEL_CONCURRENCY = 2
# Keep visual retries low for long patents. A slow visual server can otherwise
# multiply per-page latency by many minutes.
VISION_MODEL_MAX_RETRIES = 1
VISION_MODEL_OUTER_TIMEOUT_PADDING_SECONDS = 10
# Timeout (seconds) for a single text/LLM model call (content_to_dict,
# compound-ID resolution, assay extraction).
LLM_MODEL_TIMEOUT_SECONDS = 180


# --- OCR Engine Configuration ---
# PADDLEOCR_SERVER_URL = 'http://your_paddleocr_server:8010'
# Cache shared OCR page text in-memory so multiple assay names reuse one OCR pass.
ASSAY_PAGE_TEXT_CACHE_ENABLED = True
# Keep the cache small to control memory while still reusing repeated page ranges.
ASSAY_PAGE_TEXT_CACHE_MAX_ENTRIES = 4
# Only cache ranges up to this many pages.
ASSAY_PAGE_TEXT_CACHE_MAX_PAGES = 64
# Assay-page auto-detection uses PaddleOCR markdown plus the configured text
# model and a skill prompt.
# OCR requests are page-range batches. By default each batch is first written
# to a small temporary PDF, so concurrent requests upload only their own pages
# instead of re-uploading the full patent PDF.
ASSAY_AUTO_DETECT_OCR_SPLIT_PDF = True
ASSAY_AUTO_DETECT_OCR_BATCH_SIZE = 6
ASSAY_AUTO_DETECT_OCR_CONCURRENCY = 2
ASSAY_AUTO_DETECT_OCR_TIMEOUT_SECONDS = 180
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
# Structure extraction safeguards. If a page or model call stalls, skip it
# quickly instead of blocking an entire patent for 10+ minutes per page.
STRUCTURE_MODEL_TIMEOUT_SECONDS = 180
STRUCTURE_MOLECULE_PROCESSING_TIMEOUT_SECONDS = 60
STRUCTURE_PAGE_PROCESSING_TIMEOUT_SECONDS = 240
# MolNexTR graph-to-SMILES post-processing workers. Keep at 1 for web tasks:
# per-segment multiprocessing can stall threaded API workers and accumulate
# zombie child processes.
MOLNEXTR_POSTPROCESS_WORKERS = 1
# Long-running task concurrency. Keep structure extraction serialized on one GPU
# to avoid CUDA OOM when multiple PDFs/pages are running.
MAX_CONCURRENT_TASKS = 4
STRUCTURE_TASK_CONCURRENCY = 1
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
