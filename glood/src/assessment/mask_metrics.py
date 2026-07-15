from __future__ import annotations

import numpy as np

METRIC_COLUMNS = (
    "accuracy",
    "precision",
    "recall",
    "specificity",
    "fpr",
    "fnr",
    "iou",
    "dice",
)


def ensure_series_axis(array: np.ndarray, *, name: str) -> np.ndarray:
    """Ensure data has shape (T, H, W, ...), adding T=1 when needed."""
    if array.ndim < 3:
        raise ValueError(f"{name} must have at least 3 dimensions, got shape {array.shape}.")
    if array.ndim == 3:
        return np.expand_dims(array, axis=0)
    return array


def velocity_activity_mask(
    fields: np.ndarray,
    *,
    eps: float,
    ux_channel: int = 1,
    uy_channel: int = 2,
) -> np.ndarray:
    """Build a boolean activity mask from velocity magnitude > eps."""
    fields = ensure_series_axis(np.asarray(fields), name="fields")
    if fields.ndim != 4:
        raise ValueError(f"fields must have shape (T,H,W,C), got {fields.shape}.")

    c = fields.shape[-1]
    if not (0 <= ux_channel < c and 0 <= uy_channel < c):
        raise ValueError(
            f"Velocity channels out of bounds for C={c}: ux={ux_channel}, uy={uy_channel}."
        )

    ux = fields[..., ux_channel]
    uy = fields[..., uy_channel]
    speed_sq = ux * ux + uy * uy
    return speed_sq > float(eps) ** 2


def normalize_matnum_series(matnum: np.ndarray, *, series_len: int) -> np.ndarray:
    """Normalize matnum arrays to shape (T,H,W), repeating T=1 across the series when needed."""
    matnum = np.asarray(matnum)

    if matnum.ndim == 4 and matnum.shape[-1] == 1:
        matnum = matnum[..., 0]
    elif matnum.ndim == 2:
        matnum = np.expand_dims(matnum, axis=0)
    elif matnum.ndim not in (3, 4):
        raise ValueError(f"Unsupported matnum shape: {matnum.shape}.")

    if matnum.ndim == 4:
        raise ValueError(f"Unsupported matnum shape after squeeze: {matnum.shape}.")
    if matnum.ndim != 3:
        raise ValueError(f"matnum must resolve to (T,H,W), got {matnum.shape}.")

    if matnum.shape[0] == 1 and series_len > 1:
        matnum = np.repeat(matnum, series_len, axis=0)

    if matnum.shape[0] != series_len:
        raise ValueError(
            f"matnum series length mismatch: expected {series_len}, got {matnum.shape[0]}."
        )
    return matnum


def matnum_activity_mask(
    matnum: np.ndarray,
    *,
    series_len: int,
    matnum_value: int = 1,
) -> np.ndarray:
    """Build a boolean mask where matnum equals matnum_value."""
    aligned = normalize_matnum_series(matnum, series_len=series_len)
    return aligned == int(matnum_value)


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def binary_metrics_from_counts(tp: int, tn: int, fp: int, fn: int) -> dict[str, float]:
    total = tp + tn + fp + fn
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)

    return {
        "accuracy": _safe_div(tp + tn, total),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "fpr": _safe_div(fp, fp + tn),
        "fnr": _safe_div(fn, fn + tp),
        "iou": _safe_div(tp, tp + fp + fn),
        "dice": _safe_div(2 * tp, 2 * tp + fp + fn),
    }


def binary_metrics(lhs_mask: np.ndarray, rhs_mask: np.ndarray) -> dict[str, float | int]:
    lhs_mask = np.asarray(lhs_mask, dtype=bool)
    rhs_mask = np.asarray(rhs_mask, dtype=bool)
    if lhs_mask.shape != rhs_mask.shape:
        raise ValueError(f"Mask shape mismatch: {lhs_mask.shape} vs {rhs_mask.shape}.")

    tp = int(np.logical_and(lhs_mask, rhs_mask).sum())
    tn = int(np.logical_and(~lhs_mask, ~rhs_mask).sum())
    fp = int(np.logical_and(lhs_mask, ~rhs_mask).sum())
    fn = int(np.logical_and(~lhs_mask, rhs_mask).sum())

    metrics = binary_metrics_from_counts(tp, tn, fp, fn)
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        **metrics,
    }
