from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real
from typing import Any

from torch import nn
from torch.utils.data import DataLoader

from glood.src.callbacks.training_callback import TrainingCallback


class NaNGuardCallback(TrainingCallback):
    """
    Raise an error as soon as any recorded loss or metric becomes non-finite.

    The ModelWrapper aggregates statistics per epoch before invoking callbacks, so this guard
    aborts training right after the first affected epoch finishes.
    """

    def __call__(
        self,
        epoch: int,
        model: nn.Module,
        train_loader: DataLoader | None,
        val_loader: DataLoader | None,
        data: dict,
    ) -> None:
        if not isinstance(data, Mapping):
            raise TypeError("NaNGuardCallback expects 'data' to be a mapping.")

        epoch_display = epoch + 1
        self._check_split("train", data.get("train"), epoch_display)
        self._check_split("validation", data.get("validation"), epoch_display)

    def _check_split(
        self,
        split_name: str,
        split_stats: Mapping[str, Any] | None,
        epoch_display: int,
    ) -> None:
        if split_stats is None:
            return
        if not isinstance(split_stats, Mapping):
            raise TypeError(f"NaNGuardCallback expects '{split_name}' stats to be a mapping.")

        self._validate_value(split_stats.get("loss"), f"{split_name}.loss", epoch_display)

        metrics = split_stats.get("metrics")
        if metrics is None:
            return
        if not isinstance(metrics, Mapping):
            raise TypeError(f"NaNGuardCallback expects '{split_name}.metrics' to be a mapping.")

        for metric_name, metric_value in metrics.items():
            key = f"{split_name}.metrics[{metric_name}]"
            self._validate_value(metric_value, key, epoch_display)

    def _validate_value(self, value: Any, key: str, epoch_display: int) -> None:
        numeric = self._as_float(value)
        if numeric is None:
            return
        if not math.isfinite(numeric):
            raise RuntimeError(
                f"NaNGuardCallback detected a non-finite value for {key} at epoch {epoch_display}: {numeric!r}"
            )

    @staticmethod
    def _as_float(value: Any) -> float | None:
        if value is None:
            return None
        if isinstance(value, Real):
            return float(value)
        if hasattr(value, "item"):
            try:
                return float(value.item())
            except (TypeError, ValueError):
                return None
        return None
