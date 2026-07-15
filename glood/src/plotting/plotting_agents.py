import math
import numpy as np
from typing import Sequence, Callable

from glood.src.plotting.plotting_kernel import plotting_kernel
from glood.src.plotting.data_bundles import PlottableProcessed
from glood.src.plotting.plotting_handle import PlottingHandle
import glood.src.plotting.plot_paths as pp


def _get_series_len(provider):
    first_iter = iter(provider)
    first_item = next(first_iter)
    series_len = first_item.pred.shape[0]
    del first_item
    return series_len

def _initialize_plottables(plottable_input, protocols):
    # creating a list of processed plottables
    plottables = [processing_protocol(plottable_input) for processing_protocol in protocols]
    group_extremes = dict()

    # populating the group extremes
    for plottable in plottables:
        if plottable.group_dominant:
            group_extremes[plottable.group] = (np.min(plottable.data[0]), np.max(plottable.data[0]))
    
    #setting the plottable extremes
    for plottable in plottables:
        plottable.extremes = group_extremes[plottable.group]

    return plottables


def plot_single (
        providers,
        protocols: Sequence[Callable],
        remove_title: bool = False,
        ) -> None:
    for provider in providers:
        series_len = _get_series_len(provider)
        handle = PlottingHandle(1, 1, remove_title)

        # cycle over the inputs
        for plottable_input_series in provider:

            # creating a list of processed plottable series
            plottable_series_list = _initialize_plottables(plottable_input_series, protocols)

            # cycle over the processed plottable series, populate other params and produce the single plots
            for plottable_series in plottable_series_list:
                plottables = [
                    PlottableProcessed (
                        id             = plottable_series.id,
                        epoch          = plottable_series.epoch,
                        base_dir       = plottable_series.base_dir,
                        series_i       = i,
                        cmap           = plottable_series.cmap,
                        title_abbr     = f'{i}_{plottable_series.title_abbr}',
                        title_extended = f'Timestep {i}: {plottable_series.title_extended}',
                        units          = plottable_series.units,
                        group          = plottable_series.group,
                        group_dominant = plottable_series.group_dominant,
                        extremes       = plottable_series.extremes,
                        data           = [data[i] for data in plottable_series.data],
                    )
                    for i in range(series_len)
                ]

                # plot the single plots
                for plottable in plottables:
                    plotting_kernel(handle, plottable, pp.get_plot_snapshot_partials_path)

        del handle


def plot_snapshot (
        providers,
        protocols: Sequence[Callable],
        remove_title: bool = False,
        ) -> None:
    for provider in providers:
        series_len = _get_series_len(provider)
        protocol_len = len(protocols)
        handles = [PlottingHandle(1, len(protocols), remove_title) for _ in range(series_len)]

        # cycle over the inputs
        for plottable_input_series in provider:

            # creating a list of processed plottable series
            plottable_series_list = _initialize_plottables(plottable_input_series, protocols)

            # cycle over the processed plottable series, populate other params and produce the single plots
            for i_protocol, plottable_series in enumerate(plottable_series_list):
                plottables = [
                    PlottableProcessed (
                        id             = plottable_series.id,
                        epoch          = plottable_series.epoch,
                        base_dir       = plottable_series.base_dir,
                        series_i       = i_len,
                        cmap           = plottable_series.cmap,
                        title_abbr     = f'timestep_{i_len}',
                        title_extended = f'Timestep {i_len}: {plottable_series.title_extended}',
                        units          = plottable_series.units,
                        group          = plottable_series.group,
                        group_dominant = plottable_series.group_dominant,
                        extremes       = plottable_series.extremes,
                        data           = [data[i_len] for data in plottable_series.data],
                    )
                    for i_len in range(series_len)
                ]

                # plot the single plots
                for handle, plottable in zip(handles, plottables):
                    plotting_kernel (
                        handle,
                        plottable,
                        pp.get_plot_snapshot_path if i_protocol==protocol_len-1 else None,
                        0,
                        i_protocol
                    )

        del handles


def plot_epoch (
        providers,
        protocols: Sequence[Callable],
        remove_title: bool = False,
        ) -> None:
    for provider in providers:
        snapshots_per_row = 2
        series_len = _get_series_len(provider)
        protocol_len = len(protocols)
        nrows = math.ceil(series_len / snapshots_per_row)
        ncols = snapshots_per_row * protocol_len
        handle = PlottingHandle(nrows, ncols, remove_title)

        # cycle over the inputs
        for plottable_input_series in provider:

            # creating a list of processed plottable series
            plottable_series_list = _initialize_plottables(plottable_input_series, protocols)

            # cycle over the processed plottable series, populate other params and produce the single plots
            for i_protocol, plottable_series in enumerate(plottable_series_list):
                plottables = [
                    PlottableProcessed (
                        id             = plottable_series.id,
                        epoch          = plottable_series.epoch,
                        base_dir       = plottable_series.base_dir,
                        series_i       = i_len,
                        cmap           = plottable_series.cmap,
                        title_abbr     = f'epoch{plottable_series.epoch}',
                        title_extended = f'ep={plottable_series.epoch}, iT={i_len}: {plottable_series.title_extended}',
                        units          = plottable_series.units,
                    group          = plottable_series.group,
                        group_dominant = plottable_series.group_dominant,
                        extremes       = plottable_series.extremes,
                        data           = [data[i_len] for data in plottable_series.data],
                    )
                    for i_len in range(series_len)
                ]

                for i_len, plottable in enumerate(plottables):
                    i = i_len*protocol_len + i_protocol
                    irow, icol = divmod(i, ncols)
                    plotting_kernel (
                        handle,
                        plottable,
                        pp.get_plot_base_path if i_protocol==protocol_len-1 and i_len==series_len-1 else None,
                        irow,
                        icol
                    )

        del handle