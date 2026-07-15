from __future__ import annotations

import hydra
import matplotlib
import numpy as np
from matplotlib.patches import Rectangle
from omegaconf import DictConfig, OmegaConf

import glood.src.utils.env as env

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@hydra.main(version_base=None, config_path="conf", config_name="plot_square_domain_sections_generic")
def main(cfg: DictConfig) -> None:
    out_dir = env.out()
    with (out_dir / "conf_plot_square_domain_sections.yaml").open("w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    side = float(cfg.domain.side_length)
    x_fractions = [float(value) for value in cfg.domain.x_fractions]

    fig, ax = plt.subplots(figsize=(float(cfg.plotting.figure_size), float(cfg.plotting.figure_size)))
    domain = Rectangle(
        (0.0, 0.0),
        side,
        side,
        facecolor=str(cfg.plotting.domain_facecolor),
        edgecolor=str(cfg.plotting.domain_edgecolor),
        linewidth=float(cfg.plotting.domain_linewidth),
    )
    ax.add_patch(domain)

    for fraction in x_fractions:
        x_pos = fraction * side
        ax.plot(
            [x_pos, x_pos],
            [0.0, side],
            linestyle=str(cfg.plotting.section_linestyle),
            linewidth=float(cfg.plotting.section_linewidth),
            color=str(cfg.plotting.section_color),
        )
        ax.text(x_pos, side + float(cfg.plotting.section_label_offset), f"x/L={fraction:.2f}", ha="center", va="bottom", fontsize=int(cfg.plotting.fontsize))

    ax.text(0.5 * side, 0.5 * side, str(cfg.plotting.domain_label), ha="center", va="center", fontsize=int(cfg.plotting.title_fontsize))
    ax.annotate("", xy=(0.98 * side, -0.08 * side), xytext=(0.02 * side, -0.08 * side), arrowprops={"arrowstyle": "<->", "linewidth": 1.5})
    ax.text(0.5 * side, -0.11 * side, "L", ha="center", va="top", fontsize=int(cfg.plotting.fontsize))
    ax.annotate("", xy=(-0.08 * side, 0.98 * side), xytext=(-0.08 * side, 0.02 * side), arrowprops={"arrowstyle": "<->", "linewidth": 1.5})
    ax.text(-0.11 * side, 0.5 * side, "L", ha="right", va="center", rotation=90, fontsize=int(cfg.plotting.fontsize))

    margin = float(cfg.plotting.outer_margin)
    ax.set_xlim(-margin * side, (1.0 + 0.03) * side)
    ax.set_ylim(-margin * side, (1.0 + 0.10) * side)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()

    stem = out_dir / str(cfg.outputs.figure_stem)
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    csv_path = out_dir / str(cfg.outputs.sections_csv)
    np.savetxt(
        csv_path,
        np.array([[fraction] for fraction in x_fractions], dtype=float),
        delimiter=",",
        header="x_over_L",
        comments="",
    )


if __name__ == "__main__":
    main()
