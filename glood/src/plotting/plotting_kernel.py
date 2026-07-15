import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import Callable

import glood.src.utils.env as env
from glood.src.plotting.data_bundles import PlottableProcessed
from glood.src.plotting.plotting_handle import PlottingHandle


def plotting_kernel (
    handle: PlottingHandle,
    plottable: PlottableProcessed,
    get_plot_path: Callable | None,
    i: int = 0,
    j: int = 0,
) -> None:
    """Materially draws the plot."""

    def add_cbar(ax, mappable, remove_title=False, width=0.04, pad=0.02, length=0.70):
        cbar = ax.figure.colorbar(
            mappable, ax=ax,
            fraction=width,
            pad=pad,
            shrink=length,
            anchor=(0, 0.5),   # left edge, vertically centered
            orientation="vertical"
        )
        if remove_title:
            cbar.ax.tick_params(labelsize=16)
        return cbar


    if handle.X is None or handle.Y is None:
        handle.X, handle.Y = np.meshgrid (
            np.arange(plottable.data[0].shape[0]),
            np.arange(plottable.data[0].shape[1])
        )

    handle.ax[i, j].set_xticks([])
    handle.ax[i, j].set_yticks([])
    if not handle.remove_title:
        handle.ax[i, j].set_title(plottable.title_extended)

    if handle.pc[i, j] is None:
        handle.pc[i, j] = handle.ax[i, j].pcolormesh (
            handle.X,
            handle.Y,
            plottable.data[0],
            cmap=plottable.cmap,
            shading='auto',
            vmin=plottable.extremes[0],
            vmax=plottable.extremes[1],
        )
        handle.cbar[i, j] = add_cbar(handle.ax[i, j], handle.pc[i, j], handle.remove_title)
        handle.ax[i, j].set_box_aspect(1) 
    else:
        handle.pc[i, j].set_array(plottable.data[0].ravel())
        handle.pc[i, j].set_clim(plottable.extremes[0], plottable.extremes[1])
        handle.pc[i, j].set_cmap(plottable.cmap)
        handle.pc[i, j].changed() 

        handle.cbar[i, j].update_normal(handle.pc[i, j])
        handle.cbar[i, j].update_ticks()

    if not handle.remove_title:
        handle.cbar[i, j].ax.set_title(plottable.units)
    
    if len(plottable.data) == 3:
        def compute_uu_vv():
            uu = plottable.data[1][::plottable.stride, ::plottable.stride]
            vv = plottable.data[2][::plottable.stride, ::plottable.stride]
            mag = np.hypot(uu, vv)
            #mag_max = float(mag.max())
            #return uu/mag_max, vv/mag_max
            mag[mag==0] = 1.
            return uu/mag, vv/mag

        if handle.qv[i, j] is None:
            XX = handle.X[::plottable.stride, ::plottable.stride]
            YY = handle.Y[::plottable.stride, ::plottable.stride]
            uu, vv = compute_uu_vv()
            handle.qv[i, j] = handle.ax[i, j].quiver(XX, YY, uu, vv, color='k')

        else:
            uu, vv = compute_uu_vv()
            handle.qv[i, j].set_UVC(uu, vv)

        handle.qv[i, j].set_visible(True)

    elif len(plottable.data) not in {1, 3}:
        raise ValueError(f'plottable has length {len(plottable.data)} whereas it is expected either 1 or 3.')

    else:
        if handle.qv[i, j] is not None:
            handle.qv[i, j].set_visible(False)


    if get_plot_path is not None:
        pic_path = get_plot_path (
            model_dir  = env.out(),
            example_id = plottable.id,
            epoch_no   = plottable.epoch,
            series_i   = plottable.series_i,
        )
        pic_path.mkdir(parents=True, exist_ok=True)

        handle.fig.savefig (
            pic_path / f'{plottable.title_abbr}.png',
            bbox_inches='tight',
            pad_inches=0
        )