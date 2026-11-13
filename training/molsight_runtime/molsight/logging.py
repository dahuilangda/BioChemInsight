import json
import logging
import math
import time
from pathlib import Path

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s -%(module)s:  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S ",
    level=logging.INFO,
)


def get_logger(name: str | None = None) -> logging.Logger:
    if name is None:
        name = __name__
    return logging.getLogger(name)


class ScalarWriter:
    def __init__(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.stream = (directory / "scalars.jsonl").open("a", buffering=1)

    def add_scalar(self, tag, value, step):
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"Nonfinite training metric: {tag}")
        self.stream.write(
            json.dumps(
                {"time": time.time(), "step": int(step), "tag": tag, "value": value},
                separators=(",", ":"),
            )
            + "\n"
        )

    def flush(self):
        self.stream.flush()

    def close(self):
        self.stream.close()
