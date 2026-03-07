"""Bootstrap project virtual environment with robust cross-platform fallbacks.

This script centralizes environment setup so Windows batch wrappers stay small and
portable while setup logic remains testable and easier to maintain.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

# Project root and default setup paths.
ROOT = Path(__file__).resolve().parents[1]
VENV_DIR = ROOT / ".venv"
REQ_FILE = ROOT / "requirements.txt"


def _run(cmd: list[str], *, check: bool = True) -> int:
    """Run a command in project root and stream output to current console."""
    print("[CMD]", " ".join(str(x) for x in cmd), flush=True)
    proc = subprocess.run(cmd, cwd=ROOT)
    if check and proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    return proc.returncode


def _venv_python_candidates() -> list[Path]:
    """Return interpreter candidates inside virtualenv for mixed shell setups.

    Some environments run Windows Python from Unix-like shells (or vice versa),
    so we probe both conventional layouts while still preferring native OS order.
    """
    if os.name == "nt":
        return [VENV_DIR / "Scripts" / "python.exe", VENV_DIR / "bin" / "python"]
    return [VENV_DIR / "bin" / "python", VENV_DIR / "Scripts" / "python.exe"]


def _find_existing_venv_python() -> Path | None:
    """Return first existing venv interpreter path, if any."""
    for candidate in _venv_python_candidates():
        if candidate.exists():
            return candidate
    return None


def _have_module(mod: str) -> bool:
    """Check whether bootstrap interpreter can import a module."""
    return subprocess.run(
        [sys.executable, "-c", f"import {mod}"], cwd=ROOT
    ).returncode == 0


def _have_pip() -> bool:
    """Check whether bootstrap interpreter has pip available."""
    return subprocess.run([sys.executable, "-m", "pip", "--version"], cwd=ROOT).returncode == 0


def _try_stdlib_venv() -> bool:
    """Attempt creating virtualenv via stdlib venv."""
    print("[STEP] Trying stdlib venv...", flush=True)
    return _run([sys.executable, "-m", "venv", str(VENV_DIR)], check=False) == 0


def _try_ensurepip() -> bool:
    """Attempt restoring pip using ensurepip, if available."""
    print("[STEP] Trying ensurepip...", flush=True)
    return _run([sys.executable, "-m", "ensurepip", "--upgrade"], check=False) == 0


def _try_virtualenv() -> bool:
    """Attempt creating virtualenv using third-party virtualenv fallback."""
    print("[STEP] Trying virtualenv fallback...", flush=True)
    if not _have_pip():
        print("[WARN] pip is not available in bootstrap interpreter.", flush=True)
        return False

    # Best effort upgrade to improve compatibility with old pip versions.
    _run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], check=False)

    if not _have_module("virtualenv"):
        if _run([sys.executable, "-m", "pip", "install", "virtualenv"], check=False) != 0:
            return False

    return _run([sys.executable, "-m", "virtualenv", str(VENV_DIR)], check=False) == 0


def ensure_venv() -> Path:
    """Create or reuse project virtualenv and return its Python path."""
    py = _find_existing_venv_python()
    if py is not None:
        print(f"[INFO] Reusing existing virtual environment: {py}", flush=True)
        return py

    if VENV_DIR.exists():
        # Stale directory from interrupted setup can block interpreter creation.
        print(f"[WARN] Removing incomplete environment: {VENV_DIR}", flush=True)
        shutil.rmtree(VENV_DIR, ignore_errors=True)

    print(f"[INFO] Bootstrap interpreter: {sys.executable}", flush=True)
    print(f"[INFO] Python version: {sys.version.split()[0]}", flush=True)

    if _try_stdlib_venv() and (py := _find_existing_venv_python()) is not None:
        return py

    print("[WARN] stdlib venv failed.", flush=True)
    _try_ensurepip()
    if _try_stdlib_venv() and (py := _find_existing_venv_python()) is not None:
        return py

    if _try_virtualenv() and (py := _find_existing_venv_python()) is not None:
        return py

    raise SystemExit(
        "[ERROR] Failed to create virtual environment.\n"
        "[HINT] On Windows this usually means one of these:\n"
        "  1) Store/alias python3 was picked instead of real Python\n"
        "  2) Python was installed without pip/venv\n"
        "  3) App execution aliases for python.exe/python3.exe are interfering\n"
        "[FIX] Install official Python from python.org with pip + venv enabled, "
        "or disable App Execution Aliases for python/python3."
    )


def install_requirements(py: Path) -> None:
    """Upgrade packaging tooling and install project requirements if present."""
    print("[STEP] Upgrading pip/setuptools/wheel...", flush=True)
    _run([str(py), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])

    if REQ_FILE.exists():
        print("[STEP] Installing requirements...", flush=True)
        _run([str(py), "-m", "pip", "install", "-r", str(REQ_FILE)])
    else:
        print("[WARN] requirements.txt not found, skipping install.", flush=True)


def main() -> int:
    """Entry-point for environment bootstrap flow."""
    py = ensure_venv()
    install_requirements(py)

    print("\n[OK] Environment is ready.", flush=True)
    print(f"[INFO] Venv Python: {py}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
