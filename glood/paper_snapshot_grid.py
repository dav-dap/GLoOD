from __future__ import annotations

from pathlib import Path

import hydra
import matplotlib
import numpy as np
from matplotlib import colors
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from omegaconf import DictConfig, OmegaConf

import glood.src.utils.env as env

matplotlib.use("Agg")


def _load_series(path: Path, *, name: str) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {name}: {path}")
    array = np.load(path)
    if array.ndim == 3:
        array = np.expand_dims(array, axis=0)
    if array.ndim != 4:
        raise ValueError(f"{name} must resolve to (T,H,W,C), got {array.shape}")
    return np.asarray(array)


def _u_magnitude(series: np.ndarray) -> np.ndarray:
    return np.sqrt(series[..., 1] ** 2 + series[..., 2] ** 2)


def _load_sequence_panels(example_dir: Path) -> dict[str, np.ndarray]:
    true = np.asarray(_load_series(example_dir / "true.npy", name="true.npy"), dtype=float)
    pred = np.asarray(_load_series(example_dir / "pred.npy", name="pred.npy"), dtype=float)
    if true.shape != pred.shape:
        raise ValueError(f"Shape mismatch in {example_dir}: true={true.shape}, pred={pred.shape}")
    return {
        "p_true": true[..., 0],
        "p_pred": pred[..., 0],
        "u_true": _u_magnitude(true),
        "u_pred": _u_magnitude(pred),
    }


def _iter_example_dirs(input_root: Path) -> list[Path]:
    return sorted(path for path in input_root.iterdir() if path.is_dir())


def _norm_from_arrays(*arrays: np.ndarray) -> colors.Normalize:
    vmin = min(float(np.nanmin(arr)) for arr in arrays)
    vmax = max(float(np.nanmax(arr)) for arr in arrays)
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        vmin, vmax = 0.0, 1.0
    if abs(vmax - vmin) < 1.0e-12:
        delta = 1.0 if abs(vmax) < 1.0e-12 else abs(vmax) * 0.05
        vmin -= delta
        vmax += delta
    return colors.Normalize(vmin=vmin, vmax=vmax)


def _add_colorbar(ax: plt.Axes, *, norm: colors.Normalize, cmap: str, label: str) -> None:
    cax = inset_axes(
        ax,
        width="86%",
        height="5%",
        loc="lower center",
        bbox_to_anchor=(0.07, -0.12, 0.86, 1.0),
        bbox_transform=ax.transAxes,
        borderpad=0.0,
    )
    cbar = plt.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
    cbar.ax.tick_params(labelsize=7, pad=1, length=2)
    cbar.set_label(label, fontsize=8, labelpad=2)


def _render_summary_page_from_views(cfg: DictConfig, *, out_dir: Path) -> None:
    columns = list(cfg.columns)
    views_dir = out_dir / str(cfg.last_timestep_panels.views_subdir)
    blank_shape = tuple(int(x) for x in cfg.summary_from_views.blank_shape)
    blank_value = float(cfg.summary_from_views.blank_value)
    column_labels = [str(x) for x in cfg.summary_from_views.column_labels]
    grid_files = [[str(cell) for cell in row] for row in cfg.summary_from_views.grid_files]
    n_rows = len(grid_files)
    n_cols = len(column_labels)
    if len(columns) != n_rows:
        raise ValueError(f"summary_from_views.grid_files must have {len(columns)} rows, got {n_rows}")
    for row in grid_files:
        if len(row) != n_cols:
            raise ValueError(f"Each summary_from_views.grid_files row must have {n_cols} entries, got {len(row)}")

    row_payloads: list[dict[str, object]] = []
    for row_idx, column_cfg in enumerate(columns):
        field = str(column_cfg.field)
        arrays = [np.full(blank_shape, blank_value, dtype=float) for _ in range(n_cols)]
        for col_idx in range(n_cols):
            file_name = grid_files[row_idx][col_idx].strip()
            if not file_name:
                continue
            view_path = views_dir / file_name
            if not view_path.is_file():
                raise FileNotFoundError(f"Missing summary-from-views grid file: {view_path}")
            arrays[col_idx] = np.load(view_path)
        cmap = str(cfg.cmaps.p if field == "p" else cfg.cmaps.u)
        row_payloads.append(
            {
                "field_label": str(column_cfg.colorbar_label),
                "cmap": cmap,
                "norm": _norm_from_arrays(arrays[0]),
                "arrays": arrays,
            }
        )

    fig_width = float(cfg.figure.from_views_width)
    fig_height = fig_width * float(n_rows) / float(n_cols + 0.14)
    title_gap = 0.006
    group_title_gap = 0.012
    title_pad = 72.0 * fig_width * title_gap
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=int(cfg.figure.dpi))
    grid = fig.add_gridspec(
        n_rows,
        n_cols + 1,
        width_ratios=[1.0] * n_cols + [0.14],
        wspace=0.02,
        hspace=float(cfg.summary_from_views.panel_gap),
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

    for row_idx, payload in enumerate(row_payloads):
        for col_idx in range(n_cols):
            ax = axes[row_idx, col_idx]
            ax.imshow(
                payload["arrays"][col_idx],
                cmap=str(payload["cmap"]),
                norm=payload["norm"],
                origin=str(getattr(cfg.figure, "origin", "lower")),
                interpolation=str(getattr(cfg.figure, "interpolation", "nearest")),
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(False)
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_aspect("equal")
            if row_idx == 0:
                ax.set_title(column_labels[col_idx], fontsize=10, pad=title_pad)
        cbar = fig.colorbar(
            ScalarMappable(norm=payload["norm"], cmap=str(payload["cmap"])),
            cax=colorbar_axes[row_idx],
        )
        cbar.ax.tick_params(labelsize=7, pad=1, length=2)
        cbar.set_label(str(payload["field_label"]), fontsize=8, labelpad=3)
        cbar.ax.yaxis.set_label_coords(2.8, 0.5)

    group_titles = [str(x) for x in getattr(cfg.summary_from_views, "row_group_titles", [])]
    if group_titles:
        if len(group_titles) * 2 != n_rows:
            raise ValueError("summary_from_views.row_group_titles must contain one title per pair of rows")
        for group_idx, title in enumerate(group_titles):
            first_row = 2 * group_idx
            last_row = first_row + 1
            y = (axes[first_row, 0].get_position().y0 + axes[last_row, 0].get_position().y1) / 2.0
            x = axes[first_row, 0].get_position().x0 - group_title_gap
            fig.text(x, y, title, rotation=90, ha="center", va="center", fontsize=10)

    fig.savefig(
        out_dir / str(cfg.outputs.summary_from_views_filename),
        bbox_inches="tight",
        pad_inches=float(cfg.outputs.pad_inches),
    )
    plt.close(fig)


def _render_sequence_pages(cfg: DictConfig, *, out_dir: Path) -> None:
    input_root = Path(str(cfg.sequence_pages.input_root)).expanduser()
    example_dirs = _iter_example_dirs(input_root)
    output_dir = out_dir / str(cfg.sequence_pages.output_subdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    column_specs = (
        ("p_true", "p true", str(cfg.cmaps.p), "Pressure"),
        ("p_pred", "p pred", str(cfg.cmaps.p), "Pressure"),
        ("u_true", "u true", str(cfg.cmaps.u), "Velocity"),
        ("u_pred", "u pred", str(cfg.cmaps.u), "Velocity"),
    )

    for example_dir in example_dirs:
        panels = _load_sequence_panels(example_dir)
        n_rows = int(panels["p_true"].shape[0])
        p_norm = _norm_from_arrays(panels["p_true"])
        u_norm = _norm_from_arrays(panels["u_true"])
        norms = {
            "p_true": p_norm,
            "p_pred": p_norm,
            "u_true": u_norm,
            "u_pred": u_norm,
        }

        fig, axes = plt.subplots(
            n_rows,
            len(column_specs),
            figsize=tuple(float(x) for x in cfg.figure.sequence_figsize),
            dpi=int(cfg.figure.dpi),
            constrained_layout=True,
        )
        if n_rows == 1:
            axes = np.expand_dims(axes, axis=0)

        for col_idx, (_, title, _, _) in enumerate(column_specs):
            axes[0, col_idx].set_title(title, fontsize=10, pad=6)

        for row_idx in range(n_rows):
            for col_idx, (key, _, cmap, _) in enumerate(column_specs):
                ax = axes[row_idx, col_idx]
                ax.imshow(
                    panels[key][row_idx],
                    cmap=cmap,
                    norm=norms[key],
                    origin=str(getattr(cfg.figure, "origin", "lower")),
                    interpolation=str(getattr(cfg.figure, "interpolation", "nearest")),
                )
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                if col_idx == 0:
                    ax.set_ylabel(f"t{row_idx}", fontsize=8, rotation=90, ha="center", va="center", labelpad=10)

        for col_idx, (key, _, cmap, label) in enumerate(column_specs):
            _add_colorbar(axes[n_rows - 1, col_idx], norm=norms[key], cmap=cmap, label=label)

        if bool(getattr(cfg.sequence_pages, "show_suptitle", True)):
            fig.suptitle(example_dir.name, fontsize=12)

        fig.savefig(output_dir / f"{example_dir.name}.png", bbox_inches="tight", pad_inches=float(cfg.outputs.pad_inches))
        plt.close(fig)


def _render_sequence_last_timesteps(cfg: DictConfig, *, out_dir: Path) -> None:
    input_root = Path(str(cfg.sequence_pages.input_root)).expanduser()
    example_dirs = _iter_example_dirs(input_root)
    views_dir = out_dir / str(cfg.last_timestep_panels.views_subdir)
    from_views_dir = out_dir / str(cfg.last_timestep_panels.from_views_subdir)
    views_dir.mkdir(parents=True, exist_ok=True)
    from_views_dir.mkdir(parents=True, exist_ok=True)

    column_specs = (
        ("p_true", "p_true", str(cfg.cmaps.p), "Pressure"),
        ("p_pred", "p_pred", str(cfg.cmaps.p), "Pressure"),
        ("u_true", "u_true", str(cfg.cmaps.u), "Velocity"),
        ("u_pred", "u_pred", str(cfg.cmaps.u), "Velocity"),
    )

    for example_dir in example_dirs:
        panels = _load_sequence_panels(example_dir)
        p_norm = _norm_from_arrays(panels["p_true"])
        u_norm = _norm_from_arrays(panels["u_true"])
        norms = {
            "p_true": p_norm,
            "p_pred": p_norm,
            "u_true": u_norm,
            "u_pred": u_norm,
        }

        for key, _, cmap, label in column_specs:
            last_view = np.asarray(panels[key][-1], dtype=float)
            np.save(views_dir / f"{example_dir.name}__last_timestep__{key}.npy", last_view)

            loaded_view = np.load(views_dir / f"{example_dir.name}__last_timestep__{key}.npy")
            fig, ax = plt.subplots(
                1,
                1,
                figsize=tuple(float(x) for x in cfg.figure.last_timestep_figsize),
                dpi=int(cfg.figure.dpi),
                constrained_layout=True,
            )
            ax.imshow(
                loaded_view,
                cmap=cmap,
                norm=norms[key],
                origin=str(getattr(cfg.figure, "origin", "lower")),
                interpolation=str(getattr(cfg.figure, "interpolation", "nearest")),
            )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_title(f"{example_dir.name} | {key} | t_last from view", fontsize=10, pad=8)
            _add_colorbar(ax, norm=norms[key], cmap=cmap, label=label)
            fig.savefig(
                from_views_dir / f"{example_dir.name}__last_timestep__{key}.png",
                bbox_inches="tight",
                pad_inches=float(cfg.outputs.pad_inches),
            )
            plt.close(fig)


@hydra.main(version_base=None, config_path="conf", config_name="paper_snapshot_grid_generic")
def main(cfg: DictConfig) -> None:
    out_dir = env.out()
    with (out_dir / "conf_paper_snapshot_grid.yaml").open("w") as f:
        f.write(OmegaConf.to_yaml(cfg))
    if bool(getattr(cfg.outputs, "render_sequence_pages", True)):
        _render_sequence_pages(cfg, out_dir=out_dir)
    if bool(getattr(cfg.outputs, "render_last_timestep_views", True)):
        _render_sequence_last_timesteps(cfg, out_dir=out_dir)
    if bool(getattr(cfg.outputs, "render_summary_page_from_views", True)):
        _render_summary_page_from_views(cfg, out_dir=out_dir)


if __name__ == "__main__":
    main()
