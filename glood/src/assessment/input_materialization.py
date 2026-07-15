from __future__ import annotations

import getpass
import os
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


TAR_SUFFIXES = {".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz"}


def _is_tar_path(path: Path) -> bool:
    return "".join(path.suffixes) in TAR_SUFFIXES


def _has_inference_data(path: Path) -> bool:
    return any(path.glob("example_*/epoch_*/data/pred.npy"))


def _resolve_inference_root(path: Path) -> Path:
    infer_path = path / "infer"
    if infer_path.is_dir() and _has_inference_data(infer_path):
        return infer_path
    if _has_inference_data(path):
        return path
    candidate_roots = [path, *sorted(child for child in path.rglob("*") if child.is_dir())]
    for root in candidate_roots:
        infer_root = root / "infer"
        if infer_root.is_dir() and _has_inference_data(infer_root):
            return infer_root
        if _has_inference_data(root):
            return root
    raise ValueError(f"Could not find example_*/epoch_*/data/pred.npy under {path}")


def _configured_staging_root(cfg: object) -> Path | None:
    for section_name in ("assessment", "analysis"):
        section = getattr(cfg, section_name, None)
        if section is None:
            continue
        value = getattr(section, "staging_root", None)
        if value is None:
            continue
        text = str(value).strip()
        if text and text.lower() != "auto":
            return Path(text)
    return None


def _auto_staging_root() -> Path:
    username = os.environ.get("USER") or getpass.getuser()
    fastscratch = Path("/mnt/fastscratch/users") / username
    if os.environ.get("SLURM_JOB_ID") and fastscratch.is_dir():
        return fastscratch / "glood_assessment_staging"
    return Path("/tmp") / f"glood_assessment_staging_{username}"


def _staging_root(cfg: object) -> Path:
    return _configured_staging_root(cfg) or _auto_staging_root()


@contextmanager
def materialized_inference_input(input_path: Path, *, label: str, cfg: object) -> Iterator[Path]:
    """Yield the directory containing example_* inference outputs.

    Directory inputs may be either the inference container itself or the parent
    folder containing an ``infer`` subdirectory. Tar inputs are extracted
    selectively: only pred.npy, true.npy, and matnum.npy members are staged.
    """

    if input_path.is_dir():
        yield _resolve_inference_root(input_path)
        return

    if not input_path.is_file():
        raise FileNotFoundError(f"Input path does not exist for {label}: {input_path}")
    if not _is_tar_path(input_path):
        raise ValueError(f"Unsupported inference input for {label}: {input_path}")

    staging_root = _staging_root(cfg)
    staging_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"{label}-", dir=staging_root) as tmpdir:
        staging_dir = Path(tmpdir)
        subprocess.run(
            [
                "tar",
                "-xf",
                str(input_path),
                "-C",
                str(staging_dir),
                "--wildcards",
                "--no-anchored",
                "pred.npy",
                "true.npy",
                "matnum.npy",
            ],
            check=True,
        )
        yield _resolve_inference_root(staging_dir)
