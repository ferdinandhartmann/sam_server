"""Simple client utility to test the reconstruction API."""

from __future__ import annotations

import argparse
from pathlib import Path

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit an image for SAM reconstruction")
    parser.add_argument("image", type=Path, help="Path to an input image")
    parser.add_argument("--host", default="127.0.0.1", help="Server host or IP")
    parser.add_argument("--port", default=8000, type=int, help="Server port")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    url = f"http://{args.host}:{args.port}/reconstruct"
    with args.image.open("rb") as f:
        response = requests.post(url, files={"image": f})
    response.raise_for_status()
    print(response.json())


if __name__ == "__main__":
    main()
