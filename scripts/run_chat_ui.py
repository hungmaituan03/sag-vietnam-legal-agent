#!/usr/bin/env python3
"""Launch the SAG Legal chat UI (FastAPI + static frontend).

Usage (repo root, venv on):
  pip install -e ".[web,dev]"
  python scripts/run_chat_ui.py

Then open http://127.0.0.1:8000
Requires VOYAGE_API_KEY, QWEN_API_KEY, and data/raw/uts_vlc_processed.json.
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="SAG Legal chat UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Auto-reload on code changes (dev).",
    )
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit(
            'Install web extras first: pip install -e ".[web,dev]"'
        ) from exc

    uvicorn.run(
        "sag_legal.chat.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
