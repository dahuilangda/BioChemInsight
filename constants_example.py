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


# --- OCR Engine Configuration ---
# Set to 'paddleocr' or 'dots_ocr'. Leave unset to default to 'paddleocr'.
DEFAULT_OCR_ENGINE = 'dots_ocr'
# PADDLEOCR_SERVER_URL = 'http://your_paddleocr_server:8010'       # Required when using paddleocr


# --- DotsOCR Server Configuration ---
DOTSOCR_SERVER_URL = 'http://your_dotsocr_server:8001'
DOTSOCR_PROMPT_MODE = 'prompt_layout_all_en'
