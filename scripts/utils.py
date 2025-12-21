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
        "purple": "\033[38;5;135m",
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
