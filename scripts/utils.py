"""Shared configuration loader for SAM server and clients."""

import json
from pathlib import Path
from typing import Any, Dict

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"

def load_config(config_path: str | Path | None = None) -> Dict[str, Any]:
    """Load a JSON config file if it exists, otherwise return an empty dict."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)



import builtins
from datetime import datetime

class ColorPrint:
    COLORS = {
        "reset": "\033[0m",
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
        "white": "\033[97m",
        "orange": "\033[38;5;208m",
    }

    def __init__(self, worker_name="WORKER", default_color="cyan"):
        self.worker_name = worker_name
        self.default_color = default_color
        self._orig_print = builtins.print

    def __call__(self, *args, color=None, **kwargs):
        ts = datetime.now().strftime("%H:%M:%S")
        c = self.COLORS.get(color or self.default_color, "")
        r = self.COLORS["reset"]

        prefix = f"[{ts}][{self.worker_name}]"

        self._orig_print(f"{c}{prefix}", *args, f"{r}", **kwargs)
