
"""Runtime defaults for bounded CPU execution."""

import os

os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
