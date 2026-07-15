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


def _norm(array: np.ndarray) -> colors.Normalize:
    vmin = float(np.nanmin(array))
    vmax = float(np.nanmax(array))
    if abs(vmax - vmin) < 1.0e-12:
        delta = 1.0 if abs(vmax) < 1.0e-12 else abs(vmax) * 0.05
        vmin -= delta
        vmax += delta
    return colors.Normalize(vmin=vmin, vmax=vmax)


def _render(cfg: DictConfig, out_dir: Path) -> None:
    input_root = Path(str(cfg.input_root)).expanduser()
    rows = list(cfg.rows)
    column_labels = [str(label) for label in cfg.column_labels]
    n_rows = len(rows)
    n_cols = len(column_labels)

    if not input_root.is_dir():
        raise FileNotFoundError(f"Saved panel directory does not exist: {input_root}")
    if any(len(row.files) != n_cols for row in rows):
        raise ValueError("Every saved-grid row must contain one file per column")

    payloads = []
    for row in rows:
        arrays = []
        for file_name in row.files:
            path = input_root / str(file_name)
            if not path.is_file():
                raise FileNotFoundError(f"Missing saved panel: {path}")
            arrays.append(np.asarray(np.load(path), dtype=float))
        payloads.append(
            {
                "arrays": arrays,
                "cmap": str(cfg.cmaps.p if str(row.field) == "p" else cfg.cmaps.u),
                "label": str("Pressure" if str(row.field) == "p" else "Velocity"),
                "norm": _norm(arrays[0]),
            }
        )

    fig_width = float(cfg.figure.width)
    colorbar_width = float(cfg.figure.colorbar_width)
    fig_height = fig_width * float(n_rows) / float(n_cols + colorbar_width)
    title_gap = float(cfg.figure.title_gap)
    group_title_gap = float(cfg.figure.group_title_gap)
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=int(cfg.figure.dpi))
    grid = fig.add_gridspec(
        n_rows,
        n_cols + 1,
        width_ratios=[1.0] * n_cols + [colorbar_width],
        wspace=0.0,
        hspace=0.0,
    )
    axes = np.empty((n_rows, n_cols), dtype=object)
    colorbar_axes = []
    for row_idx in range(n_rows):
        for col_idx in range(n_cols):
            axes[row_idx, col_idx] = fig.add_subplot(grid[row_idx, col_idx])
        colorbar_ax = fig.add_subplot(grid[row_idx, n_cols])
        position = colorbar_ax.get_position()
        colorbar_ax.set_position(
            [position.x0, position.y0 + 0.02 * position.height, position.width, 0.96 * position.height]
        )
        colorbar_axes.append(colorbar_ax)

    for row_idx, payload in enumerate(payloads):
        for col_idx in range(n_cols):
            ax = axes[row_idx, col_idx]
            ax.imshow(
                payload["arrays"][col_idx],
                cmap=payload["cmap"],
                norm=payload["norm"],
                origin=str(cfg.figure.origin),
                interpolation=str(cfg.figure.interpolation),
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(False)
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_aspect("equal")
            if row_idx == 0:
                ax.set_title(column_labels[col_idx], fontsize=10, pad=72.0 * fig_width * title_gap)

        cbar = fig.colorbar(ScalarMappable(norm=payload["norm"], cmap=payload["cmap"]), cax=colorbar_axes[row_idx])
        cbar.ax.tick_params(labelsize=7, pad=1, length=2)
        cbar.set_label(payload["label"], fontsize=8, labelpad=3)
        cbar.ax.yaxis.set_label_coords(2.8, 0.5)

    for group_idx, title in enumerate(cfg.group_titles):
        first_row = 2 * group_idx
        last_row = first_row + 1
        y = (axes[first_row, 0].get_position().y0 + axes[last_row, 0].get_position().y1) / 2.0
        x = axes[first_row, 0].get_position().x0 - group_title_gap
        fig.text(x, y, str(title), rotation=90, ha="center", va="center", fontsize=10)

    fig.savefig(out_dir / str(cfg.output_filename), bbox_inches="tight", pad_inches=float(cfg.pad_inches))
    plt.close(fig)


@hydra.main(version_base=None, config_path="conf", config_name="paper_snapshot_grid_from_saved")
def main(cfg: DictConfig) -> None:
    out_dir = env.out()
    with (out_dir / "conf_paper_snapshot_grid_from_saved.yaml").open("w") as stream:
        stream.write(OmegaConf.to_yaml(cfg))
    _render(cfg, out_dir)


if __name__ == "__main__":
    main()
