from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    """Entrypoint for running UI/CLI locally without memorizing long commands."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help", "help"}:
        print(
            "Usage:\n"
            "  python run_local.py ui [-- <extra streamlit args>]\n"
            "  python run_local.py cli <metrics|attack|mixfrac> ...\n"
        )
        return 0

    mode = argv[0]
    root = Path(__file__).resolve().parent

    if mode == "ui":
        extra = argv[1:]
        cmd = [sys.executable, "-m", "streamlit", "run", str(root / "app.py"), *extra]
        return int(subprocess.call(cmd))

    if mode == "cli":
        from src.cli import main as cli_main

        return int(cli_main(argv[1:]))

    print(f"Unknown mode: {mode}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
