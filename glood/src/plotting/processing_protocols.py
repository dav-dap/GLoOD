import numpy as np

from glood.src.assessment.mask_metrics import normalize_matnum_series
from glood.src.plotting.data_bundles import PlottableInput as PlottableInput
from glood.src.plotting.data_bundles import PlottableProcessed as PlottableProcessed


def p_pred(input: PlottableInput) -> PlottableProcessed:
    return input.build_processed (
        title_abbr     = 'p_pred',
        title_extended = 'Pressure, Predicted',
        units          = 'Pa',
        cmap           = 'plasma',
        group          = 'p',
        group_dominant = False,
        data           = [input.pred[..., 0]]
    )


def p_true(input: PlottableInput) -> PlottableProcessed:
    return input.build_processed (
        title_abbr     = 'p_true',
        title_extended = 'Pressure, True',
        units          = 'Pa',
        cmap           = 'plasma',
        group          = 'p',
        group_dominant = True,
        data           = [input.true[..., 0]]
    )


def p_mse(input: PlottableInput) -> PlottableProcessed:
    return input.build_processed (
        title_abbr     = 'p_mse',
        title_extended = 'Pressure, MSE',
        units          = '',
        cmap           = 'plasma',
        group          = 'p_mse',
        group_dominant = True,
        data           = [np.square(input.true[..., 0] - input.pred[..., 0])]
    )


def u_pred(input: PlottableInput) -> PlottableProcessed:
    ux, uy = input.pred[..., 1], input.pred[..., 2]
    return input.build_processed (
        title_abbr     = 'u_pred',
        title_extended = 'Velocity, Predicted',
        units          = 'm/s',
        cmap           = 'cividis',
        group          = 'u',
        group_dominant = False,
        data           = [np.sqrt(ux**2 + uy**2), ux, uy]
    )


def u_true(input: PlottableInput) -> PlottableProcessed:
    ux, uy = input.true[..., 1], input.true[..., 2]
    return input.build_processed (
        title_abbr     = 'u_true',
        title_extended = 'Velocity, True',
        units          = 'm/s',
        cmap           = 'cividis',
        group          = 'u',
        group_dominant = True,
        data           = [np.sqrt(ux**2 + uy**2), ux, uy]
    )


def u_mse(input: PlottableInput) -> PlottableProcessed:
    ux_pred = input.pred[..., 1]
    uy_pred = input.pred[..., 2]
    ux_true = input.true[..., 1]
    uy_true = input.true[..., 2]
    return input.build_processed (
        title_abbr     = 'u_mse',
        title_extended = 'Velocity, MSE',
        units          = '',
        cmap           = 'cividis',
        group          = 'u_mse',
        group_dominant = True,
        data           = [np.square(ux_true - ux_pred) + np.square(uy_true - uy_pred)]
    )


def u_cos_dist(input: PlottableInput) -> PlottableProcessed:
    u_pred = input.pred[..., 1:3]
    u_true = input.true[..., 1:3]
    dot = np.einsum('...i,...i->...', u_true, u_pred)
    mag_true = np.linalg.norm(u_true, axis=-1)
    mag_pred = np.linalg.norm(u_pred, axis=-1)
    return input.build_processed (
        title_abbr     = 'u_cos_dist',
        title_extended = 'Velocity, Cos Dist',
        units          = '',
        cmap           = 'cividis',
        group          = 'u_cos_dist',
        group_dominant = True,
        data           = [1. - dot / (mag_true * mag_pred + 1e-8)]
    )


def matnum_true(input: PlottableInput) -> PlottableProcessed:
    if input.matnum is None:
        raise ValueError("matnum_true requires 'matnum' data, but none was provided.")

    matnum = normalize_matnum_series(input.matnum, series_len=input.pred.shape[0])

    return input.build_processed(
        title_abbr="matnum_true",
        title_extended="Material Number",
        units="",
        cmap="rainbow",
        group="matnum",
        group_dominant=True,
        data=[matnum],
    )
