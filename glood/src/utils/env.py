from __future__ import annotations
import os
from pathlib import Path

def cwd() -> Path:
    """Root of the project"""
    return Path(os.environ.get("SUBMIT_DIR", "."))

def out() -> Path:
    """Directory where this run writes outputs"""
    p = Path(os.environ.get("RUN_DIR", "."))
    p.mkdir(parents=True, exist_ok=True)
    return p

def attach_to_cwd(path: Path) -> Path:
    return cwd() / path

def attach_to_out(path: Path) -> Path:
    return out() / path