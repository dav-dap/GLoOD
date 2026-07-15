from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from glood.assess_profiles import (
    BARKLA_FASTSCRATCH_ROOT,
    _ensure_series_axis,
    _is_tar_path,
    _iter_example_dirs,
    _normalize_matnum_series,
    _resolve_extracted_root,
)


@dataclass(frozen=True)
class ProblematicSnapshot:
    dataset_label: str
    example_id: int
    epoch: int
    timestep: int
    frame: np.ndarray
    matnum: np.ndarray | None
    source_name: str


def _infer_dataset_label(input_path: Path) -> str:
    name = input_path.name
    for suffix in (".tar.gz", ".tar.bz2", ".tar.xz", ".tgz", ".tbz2", ".txz", ".tar"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return input_path.stem if input_path.is_file() else input_path.name


def _resolve_staging_root(configured: str | Path) -> Path:
    text = str(configured).strip()
    if text and text.lower() != "auto":
        return Path(text)
    if os.environ.get("SLURM_JOB_ID"):
        return BARKLA_FASTSCRATCH_ROOT / "glood_assess_problematic_examples_staging"
    return Path("/tmp")


def _label_from_shard_dir(path: str | Path) -> str:
    candidate = Path(path)
    if candidate.name == "dataset":
        candidate = candidate.parent
    if candidate.name.isdigit():
        candidate = candidate.parent
    return candidate.name


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


class InferenceNpySource:
    """Read inference artifacts with example_*/epoch_*/data/*.npy layout."""

    def __init__(
        self,
        input_dir: str | Path,
        *,
        dataset_label: str = "",
        source_name: str = "true",
        staging_root: str | Path = "auto",
    ) -> None:
        self.input_path = Path(input_dir)
        self.dataset_label = str(dataset_label).strip() or _infer_dataset_label(self.input_path)
        self.source_name = str(source_name).strip().lower()
        if self.source_name not in {"true", "pred"}:
            raise ValueError("InferenceNpySource source_name must be 'true' or 'pred'.")
        self.staging_root = staging_root
        self._tmpdir: tempfile.TemporaryDirectory[str] | None = None
        self._root: Path | None = None

    def __enter__(self) -> InferenceNpySource:
        if self.input_path.is_dir():
            self._root = self.input_path
            return self
        if not self.input_path.is_file():
            raise FileNotFoundError(f"Input path does not exist: {self.input_path}")
        if not _is_tar_path(self.input_path):
            raise ValueError(f"Unsupported input path for inference audit: {self.input_path}")

        staging_root = _resolve_staging_root(self.staging_root)
        staging_root.mkdir(parents=True, exist_ok=True)
        self._tmpdir = tempfile.TemporaryDirectory(
            prefix=f"problematic-{self.dataset_label}-",
            dir=staging_root,
        )
        staging_dir = Path(self._tmpdir.name)
        cmd = [
            "tar",
            "-xf",
            str(self.input_path),
            "-C",
            str(staging_dir),
            "--wildcards",
            "--no-anchored",
            f"{self.source_name}.npy",
            "matnum.npy",
        ]
        subprocess.run(cmd, check=True)
        self._root = _resolve_extracted_root(staging_dir)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._root = None
        if self._tmpdir is not None:
            self._tmpdir.cleanup()
            self._tmpdir = None

    def iter_snapshots(self) -> Iterator[ProblematicSnapshot]:
        if self._root is None:
            with self:
                yield from self.iter_snapshots()
            return

        for example_id, epoch, data_dir in _iter_example_dirs(self._root):
            source = _ensure_series_axis(
                np.load(Path(data_dir) / f"{self.source_name}.npy", allow_pickle=False, mmap_mode="r"),
                name=self.source_name,
            )
            matnum = _normalize_matnum_series(
                np.load(Path(data_dir) / "matnum.npy", allow_pickle=False, mmap_mode="r"),
                series_len=source.shape[0],
            )
            for timestep in range(source.shape[0]):
                yield ProblematicSnapshot(
                    dataset_label=self.dataset_label,
                    example_id=int(example_id),
                    epoch=int(epoch),
                    timestep=int(timestep),
                    frame=np.asarray(source[timestep], dtype=float),
                    matnum=np.asarray(matnum[timestep]),
                    source_name=self.source_name,
                )


class ShardedDatasetSource:
    """Read samples through the same sharded dataset objects used by train/infer."""

    def __init__(
        self,
        dataset,
        *,
        dataset_label: str = "",
        sample_source: str = "targets",
        source_name: str = "target",
        split: str = "all",
        val_fraction: float | None = None,
        split_shuffle: bool = False,
        max_examples: int | None = None,
        dataset_config_path: str | None = None,
        parsing_target: str | None = None,
    ) -> None:
        self.dataset = dataset
        self.dataset_label = str(dataset_label).strip() or self._infer_dataset_label()
        self.sample_source = str(sample_source)
        self.source_name = str(source_name)
        self.split = str(split).strip().lower()
        self.val_fraction = val_fraction
        self.split_shuffle = bool(split_shuffle)
        self.max_examples = None if max_examples is None else int(max_examples)
        self.dataset_config_path = dataset_config_path
        self.parsing_target = parsing_target

    def _infer_dataset_label(self) -> str:
        for attr_name in ("shard_dir", "root", "root_dir", "data_dir"):
            value = getattr(self.dataset, attr_name, None)
            if value is not None:
                return _label_from_shard_dir(value)
        return self.dataset.__class__.__name__

    def __enter__(self) -> ShardedDatasetSource:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def _selected_dataset(self):
        if self.split == "all":
            return self.dataset
        if self.val_fraction is None:
            raise ValueError("ShardedDatasetSource split requires val_fraction unless split='all'.")
        train_dataset, val_dataset = self.dataset.split(
            float(self.val_fraction),
            shuffle=self.split_shuffle,
        )
        if self.split == "train":
            return train_dataset
        if self.split in {"val", "validation", "test"}:
            return val_dataset
        raise ValueError("ShardedDatasetSource split must be one of: all, train, val, validation, test.")

    def _extract_sample_array(self, sample) -> tuple[np.ndarray, np.ndarray | None]:
        matnum = None
        if isinstance(sample, dict):
            if self.sample_source in {"inputs", "input", "x"}:
                array = sample["inputs"]
            elif self.sample_source in {"targets", "target", "y", "true"}:
                array = sample.get("targets", sample["inputs"])
            else:
                raise ValueError(f"Unsupported sample_source for dict batch: {self.sample_source}")
            matnum = sample.get("matnum")
        elif isinstance(sample, (list, tuple)):
            if len(sample) != 2:
                raise ValueError("Tuple/list samples must have exactly two entries: inputs, targets.")
            if self.sample_source in {"inputs", "input", "x"}:
                array = sample[0]
            elif self.sample_source in {"targets", "target", "y", "true"}:
                array = sample[1]
            else:
                raise ValueError(f"Unsupported sample_source for tuple sample: {self.sample_source}")
        else:
            array = sample

        return _to_numpy(array), None if matnum is None else _to_numpy(matnum)

    def iter_snapshots(self) -> Iterator[ProblematicSnapshot]:
        dataset = self._selected_dataset()
        for example_id, sample in enumerate(dataset):
            if self.max_examples is not None and example_id >= self.max_examples:
                break
            array, matnum = self._extract_sample_array(sample)
            series = _ensure_series_axis(array, name=self.sample_source)
            matnum_series = None
            if matnum is not None:
                matnum_series = _normalize_matnum_series(matnum, series_len=series.shape[0])
            for timestep in range(series.shape[0]):
                yield ProblematicSnapshot(
                    dataset_label=self.dataset_label,
                    example_id=int(example_id),
                    epoch=0,
                    timestep=int(timestep),
                    frame=np.asarray(series[timestep], dtype=float),
                    matnum=None if matnum_series is None else np.asarray(matnum_series[timestep]),
                    source_name=self.source_name,
                )
