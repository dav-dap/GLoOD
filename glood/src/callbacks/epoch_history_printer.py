from __future__ import annotations

from collections.abc import Mapping
from itertools import cycle
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader

import glood.src.utils.env as env
from glood.src.callbacks.training_callback import TrainingCallback


class EpochHistoryPrinterCallback(TrainingCallback):
    """
    Track and persist the per-epoch training history while printing a concise summary.

    Parameters
    ----------
    digits : int
        Number of decimal places used when rendering numeric values.
    output_path : str | Path
        Location where the history table is written after every epoch.
    """

    def __init__(
        self,
        digits: int = 4,
    ) -> None:
        self.digits = digits
        self.output_path = env.out() / 'history.csv'
        self._records: List[Dict[str, Any]] = []

    def __call__(
        self,
        epoch: int,
        model: nn.Module,
        train_loader: DataLoader | None,
        val_loader: DataLoader | None,
        data: dict
    ) -> None:
        if not isinstance(epoch, int):
            raise TypeError("EpochHistoryPrinterCallback expects 'epoch' to be provided as an int.")
        if not isinstance(data, Mapping):
            raise TypeError("EpochHistoryPrinterCallback expects 'data' to be a mapping.")

        train_info = data.get("train")
        if not isinstance(train_info, Mapping):
            raise TypeError("EpochHistoryPrinterCallback expects data['train'] to be a mapping.")
        val_info = data.get("validation") or {}
        if not isinstance(val_info, Mapping):
            raise TypeError("EpochHistoryPrinterCallback expects data['validation'] to be a mapping when provided.")

        train_loss = train_info.get("loss")
        raw_train_metrics = train_info.get("metrics") or {}
        val_loss = val_info.get("loss")
        raw_val_metrics = val_info.get("metrics") or {}

        if train_loss is None:
            raise ValueError("EpochHistoryPrinterCallback requires a 'train' entry with a 'loss' value.")
        if not isinstance(train_loss, (int, float)):
            raise TypeError("EpochHistoryPrinterCallback requires 'train.loss' to be numeric.")
        if val_loss is not None and not isinstance(val_loss, (int, float)):
            raise TypeError("EpochHistoryPrinterCallback expects 'validation.loss' to be numeric or None.")
        if not isinstance(raw_train_metrics, Mapping):
            raise TypeError("EpochHistoryPrinterCallback expects 'train.metrics' to be a mapping.")
        if not isinstance(raw_val_metrics, Mapping):
            raise TypeError("EpochHistoryPrinterCallback expects 'validation.metrics' to be a mapping.")

        train_metrics = dict(raw_train_metrics)
        val_metrics = dict(raw_val_metrics)

        epoch_display = epoch + 1  # Present epochs in 1-based numbering for readability.
        digits = self.digits

        segments = [
            f"train_loss={train_loss:.{digits}f}",
        ]
        record: Dict[str, Any] = {
            "epoch": epoch_display,
            "train_loss": float(train_loss),
        }

        if val_loss is not None:
            segments.append(f"validation_loss={val_loss:.{digits}f}")
            record["validation_loss"] = float(val_loss)
        else:
            record["validation_loss"] = None

        for name in sorted(train_metrics.keys()):
            value = train_metrics[name]
            if isinstance(value, (int, float)):
                segments.append(f"train_{name}={value:.{digits}f}")
                record[f"train_{name}"] = float(value)
            else:
                segments.append(f"train_{name}={value}")
                record[f"train_{name}"] = value

        for name in sorted(val_metrics.keys()):
            value = val_metrics[name]
            key = f"validation_{name}"
            if isinstance(value, (int, float)):
                segments.append(f"{key}={value:.{digits}f}")
                record[key] = float(value)
            else:
                segments.append(f"{key}={value}")
                record[key] = value

        self._records.append(record)
        dataframe = self._write_history()
        if dataframe is not None:
            self._plot_history(dataframe)

        message = f"[Epoch {epoch_display:03d}] " + " | ".join(segments)
        print(message)

    def _write_history(self) -> pd.DataFrame | None:
        if not self._records:
            return None

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        field_names = self._determine_field_order()
        dataframe = self._build_dataframe(field_names)
        self._write_with_pandas(dataframe)
        return dataframe

    def _determine_field_order(self) -> List[str]:
        keys = set().union(*(record.keys() for record in self._records))
        # Ensure deterministic ordering: epoch first, then losses, then metrics alphabetically.
        ordered = ["epoch"]
        if "train_loss" in keys:
            ordered.append("train_loss")
            keys.discard("train_loss")
        if "validation_loss" in keys:
            ordered.append("validation_loss")
            keys.discard("validation_loss")
        keys.discard("epoch")
        ordered.extend(sorted(keys))
        return ordered

    def _build_dataframe(self, field_names: List[str]) -> pd.DataFrame:
        return pd.DataFrame(self._records, columns=field_names)

    def _write_with_pandas(self, dataframe: pd.DataFrame) -> None:
        dataframe.to_csv(self.output_path, index=False)

    def _plot_history(self, dataframe: pd.DataFrame) -> None:
        if dataframe.empty:
            return

        epochs = dataframe["epoch"]
        color_cycle = cycle(plt.rcParams["axes.prop_cycle"].by_key().get("color", ["#1f77b4"]))

        def sanitize_for_log(series: pd.Series) -> pd.Series:
            numeric = pd.to_numeric(series, errors="coerce")
            if numeric.isna().all():
                return numeric

            sanitized = numeric.copy()
            zero_mask = sanitized == 0
            sanitized = sanitized.where(sanitized > 0)
            if zero_mask.any():
                sanitized.loc[zero_mask] = 1e-12
            return sanitized

        # Loss curves.
        loss_fig, loss_ax = plt.subplots(figsize=(8, 6))
        plotted_loss = False
        loss_color = next(color_cycle)

        if "train_loss" in dataframe:
            train_series = sanitize_for_log(dataframe["train_loss"])
            if train_series.notna().any():
                loss_ax.plot(epochs, train_series, label="train_loss", color=loss_color, linestyle="-")
                plotted_loss = True

        if "validation_loss" in dataframe and dataframe["validation_loss"].notna().any():
            val_series = sanitize_for_log(dataframe["validation_loss"])
            if val_series.notna().any():
                loss_ax.plot(epochs, val_series, label="validation_loss", color=loss_color, linestyle=":")
                plotted_loss = True

        loss_ax.set_xlabel("Epoch")
        loss_ax.set_ylabel("Loss")
        loss_ax.set_title("Training and Validation Loss")
        loss_ax.grid(True)
        loss_ax.legend(loc="best")
        if plotted_loss:
            try:
                loss_ax.set_yscale("log")
            except ValueError:
                pass
        loss_fig.tight_layout()
        loss_fig.savefig(self.output_path.parent / "loss_plot.png")
        plt.close(loss_fig)

        metric_columns = [
            column
            for column in dataframe.columns
            if column not in {"epoch", "train_loss", "validation_loss"}
        ]

        metrics_to_plot: dict[str, dict[str, pd.Series]] = {}
        for column in metric_columns:
            series = pd.to_numeric(dataframe[column], errors="coerce")
            if series.notna().any():
                if column.startswith("train_"):
                    base_name = column[len("train_"):]
                    metrics_to_plot.setdefault(base_name, {})["train"] = series
                elif column.startswith("validation_"):
                    base_name = column[len("validation_"):]
                    metrics_to_plot.setdefault(base_name, {})["validation"] = series
                else:
                    metrics_to_plot.setdefault(column, {})["value"] = series

        if not metrics_to_plot:
            return

        metrics_fig, metrics_ax = plt.subplots(figsize=(8, 6))
        plotted_metric = False
        color_map: Dict[str, str] = {}

        for base_name in sorted(metrics_to_plot.keys()):
            if base_name not in color_map:
                color_map[base_name] = next(color_cycle)
            color = color_map[base_name]

            series_dict = metrics_to_plot[base_name]
            for split in sorted(series_dict.keys()):
                raw_series = series_dict[split]
                sanitized = sanitize_for_log(raw_series)
                if not sanitized.notna().any():
                    continue

                label = f"{split}_{base_name}" if split in {"train", "validation"} else base_name
                linestyle = "-" if split == "train" else ":" if split == "validation" else "-"
                metrics_ax.plot(epochs, sanitized, label=label, color=color, linestyle=linestyle)
                plotted_metric = True

        metrics_ax.set_xlabel("Epoch")
        metrics_ax.set_ylabel("Metric")
        metrics_ax.set_title("Training and Validation Metrics")
        metrics_ax.grid(True)
        metrics_ax.legend(loc="best")
        if plotted_metric:
            try:
                metrics_ax.set_yscale("log")
            except ValueError:
                pass
        metrics_fig.tight_layout()
        metrics_fig.savefig(self.output_path.parent / "metrics_plot.png")
        plt.close(metrics_fig)
