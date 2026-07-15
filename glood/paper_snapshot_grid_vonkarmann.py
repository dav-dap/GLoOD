from __future__ import annotations

from pathlib import Path

import hydra
import matplotlib
import numpy as np
from matplotlib import colors
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from omegaconf import DictConfig, OmegaConf

import glood.src.utils.env as env

matplotlib.use("Agg")


def _velocity_magnitude(series: np.ndarray) -> np.ndarray:
    if series.ndim != 4 or series.shape[-1] < 3:
        raise ValueError(f"Expected a sequence with shape (T,H,W,C), C >= 3; got {series.shape}")
    return np.sqrt(series[..., 1] ** 2 + series[..., 2] ** 2)


def _norm(array: np.ndarray) -> colors.Normalize:
    vmin = float(np.nanmin(array))
    vmax = float(np.nanmax(array))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        raise ValueError("The reference snapshot contains no finite values")
    if abs(vmax - vmin) < 1.0e-12:
        delta = 1.0 if abs(vmax) < 1.0e-12 else abs(vmax) * 0.05
        vmin -= delta
        vmax += delta
    return colors.Normalize(vmin=vmin, vmax=vmax)


def _load_velocity(path: Path, name: str) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {name}: {path}")
    return _velocity_magnitude(np.asarray(np.load(path), dtype=float))


def _render(cfg: DictConfig, out_dir: Path) -> None:
    input_root = Path(str(cfg.input_root)).expanduser()
    second_input_root = Path(str(cfg.second_input_root)).expanduser()
    true = _load_velocity(input_root / "true.npy", "true.npy")
    pred = _load_velocity(input_root / "pred.npy", "pred.npy")
    second_true = _load_velocity(second_input_root / "true.npy", "second true.npy")
    second_pred = _load_velocity(second_input_root / "pred.npy", "second pred.npy")
    if true.shape != pred.shape or second_true.shape != second_pred.shape or true.shape != second_true.shape:
        raise ValueError(
            "Shape mismatch: "
            f"swin true={true.shape}, swin pred={pred.shape}, "
            f"second true={second_true.shape}, second pred={second_pred.shape}"
        )

    n_snapshots = true.shape[0]
    if n_snapshots < 5:
        raise ValueError(f"Expected at least 5 snapshots, got {n_snapshots}")
    row_sequences = (true[-5:], pred[-5:], second_true[-5:], second_pred[-5:])
    n_cols = 5
    norms = (_norm(true[-1]), _norm(second_true[-1]))
    cmap = str(cfg.cmap)
    fig_width = float(cfg.figure.width)
    colorbar_width = float(cfg.figure.colorbar_width)
    horizontal_gap = float(cfg.figure.horizontal_gap)
    vertical_gap = float(cfg.figure.vertical_gap)
    left_margin = 0.04
    right_margin = 0.98
    usable_width = fig_width * (right_margin - left_margin)
    fig_height = usable_width * (4.0 + 3.0 * vertical_gap) / (
        n_cols + colorbar_width + 5.0 * horizontal_gap
    )
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=int(cfg.figure.dpi))
    grid = fig.add_gridspec(
        4,
        n_cols + 1,
        width_ratios=[1.0] * n_cols + [colorbar_width],
        wspace=horizontal_gap,
        hspace=vertical_gap,
        left=left_margin,
        right=right_margin,
        bottom=0.0,
        top=1.0,
    )

    axes = np.empty((4, n_cols), dtype=object)
    for row_idx in range(4):
        for col_idx in range(n_cols):
            axes[row_idx, col_idx] = fig.add_subplot(grid[row_idx, col_idx])
    colorbar_axes = [fig.add_subplot(grid[0:2, n_cols]), fig.add_subplot(grid[2:4, n_cols])]
    for colorbar_ax in colorbar_axes:
        position = colorbar_ax.get_position()
        colorbar_ax.set_position(
            [position.x0, position.y0 + 0.02 * position.height, position.width, 0.96 * position.height]
        )

    for row_idx, sequence in enumerate(row_sequences):
        norm = norms[row_idx // 2]
        for col_idx, snapshot in enumerate(sequence):
            ax = axes[row_idx, col_idx]
            ax.imshow(
                snapshot,
                cmap=cmap,
                norm=norm,
                origin=str(cfg.figure.origin),
                interpolation=str(cfg.figure.interpolation),
            )
            ax.set_aspect("equal", adjustable="box")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(False)
            for spine in ax.spines.values():
                spine.set_visible(False)
    for row_idx, label in enumerate(("True", "swin", "True", "swin")):
        y = (axes[row_idx, 0].get_position().y0 + axes[row_idx, 0].get_position().y1) / 2.0
        fig.text(0.032, y, label, fontsize=8, rotation=90, ha="center", va="center")

    for colorbar_ax, norm, title in zip(
        colorbar_axes,
        norms,
        (str(cfg.dataset_title), str(cfg.second_dataset_title)),
    ):
        cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=colorbar_ax)
        cbar.ax.tick_params(labelsize=7, pad=1, length=2)
        cbar.set_label(title, fontsize=8, labelpad=1)
        cbar.ax.yaxis.set_label_coords(2.2, 0.5)

    fig.savefig(out_dir / str(cfg.output_filename), bbox_inches="tight", pad_inches=float(cfg.pad_inches))
    plt.close(fig)


@hydra.main(version_base=None, config_path="conf", config_name="paper_snapshot_grid_vonkarmann")
def main(cfg: DictConfig) -> None:
    out_dir = env.out()
    with (out_dir / "conf_paper_snapshot_grid_vonkarmann.yaml").open("w") as stream:
        stream.write(OmegaConf.to_yaml(cfg))
    _render(cfg, out_dir)


if __name__ == "__main__":
    main()
