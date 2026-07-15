from __future__ import annotations

import numpy as np

from glood.src.assessment.mask_metrics import ensure_series_axis, normalize_matnum_series


def _shift_axis0_minus(a: np.ndarray) -> np.ndarray:
    # Shift "up" (row-1) with edge replication to keep shape and avoid wraparound.
    out = np.empty_like(a)
    out[0, :] = a[0, :]
    out[1:, :] = a[:-1, :]
    return out


def _shift_axis0_plus(a: np.ndarray) -> np.ndarray:
    # Shift "down" (row+1) with edge replication.
    out = np.empty_like(a)
    out[-1, :] = a[-1, :]
    out[:-1, :] = a[1:, :]
    return out


def _shift_axis1_minus(a: np.ndarray) -> np.ndarray:
    # Shift "left" (col-1) with edge replication.
    out = np.empty_like(a)
    out[:, 0] = a[:, 0]
    out[:, 1:] = a[:, :-1]
    return out


def _shift_axis1_plus(a: np.ndarray) -> np.ndarray:
    # Shift "right" (col+1) with edge replication.
    out = np.empty_like(a)
    out[:, -1] = a[:, -1]
    out[:, :-1] = a[:, 1:]
    return out


def mixed_derivative(field: np.ndarray, fluid_mask: np.ndarray, *, axis: int, spacing: float) -> np.ndarray:
    """
    Compute first derivative with mixed stencils:
    - central where both neighbors are fluid
    - one-sided where one neighbor is non-fluid
    """
    field = np.asarray(field, dtype=float)
    fluid_mask = np.asarray(fluid_mask, dtype=bool)
    if field.shape != fluid_mask.shape:
        raise ValueError(f"field/fluid_mask shape mismatch: {field.shape} vs {fluid_mask.shape}.")

    h = float(spacing)
    if h <= 0:
        raise ValueError("spacing must be positive.")

    if axis == 0:
        fm = _shift_axis0_minus(fluid_mask)
        fp = _shift_axis0_plus(fluid_mask)
        xm = _shift_axis0_minus(field)
        xp = _shift_axis0_plus(field)
    elif axis == 1:
        fm = _shift_axis1_minus(fluid_mask)
        fp = _shift_axis1_plus(fluid_mask)
        xm = _shift_axis1_minus(field)
        xp = _shift_axis1_plus(field)
    else:
        raise ValueError("axis must be 0 or 1.")

    grad = np.zeros_like(field, dtype=float)
    # Mask partitions indicate which stencil is admissible at each fluid node.
    # Nodes with no valid neighbor on either side keep gradient 0.0; these are
    # typically outside the analyzed fluid region or isolated by masking.
    center = fluid_mask & fm & fp
    forward = fluid_mask & (~fm) & fp
    backward = fluid_mask & fm & (~fp)

    grad[center] = (xp[center] - xm[center]) / (2.0 * h)
    grad[forward] = (xp[forward] - field[forward]) / h
    grad[backward] = (field[backward] - xm[backward]) / h
    return grad


def compute_snapshot_force_components(
    fields: np.ndarray,
    matnum: np.ndarray,
    *,
    p_channel: int = 0,
    ux_channel: int = 1,
    uy_channel: int = 2,
    fluid_value: int = 1,
    wall_value: int = 2,
    crop_margin: int = 3,
    dx: float = 1.0,
    dy: float = 1.0,
    viscous_prefactor: float = 1.0,
) -> dict[str, float]:
    """
    Compute pressure and viscous force components for one (H,W,C) snapshot.

    Convention used here:
    - Array axis 0 increases downward (y in image coordinates).
    - Array axis 1 increases to the right (x in image coordinates).
    - Face normals are defined from fluid cell toward adjacent wall cell.
    """
    if fields.ndim != 3:
        raise ValueError(f"fields must have shape (H,W,C), got {fields.shape}.")
    h, w, c = fields.shape
    for ch in (p_channel, ux_channel, uy_channel):
        if not (0 <= ch < c):
            raise ValueError(f"Channel index {ch} out of bounds for C={c}.")

    p = fields[..., p_channel]
    ux = fields[..., ux_channel]
    uy = fields[..., uy_channel]

    mat = np.asarray(matnum)
    if mat.shape != (h, w):
        raise ValueError(f"matnum shape mismatch: expected {(h, w)}, got {mat.shape}.")

    # Crop out domain borders to avoid wall/boundary artefacts not related to
    # internal obstacles.
    interior = np.zeros((h, w), dtype=bool)
    if crop_margin * 2 >= min(h, w):
        raise ValueError("crop_margin too large for grid size.")
    interior[crop_margin : h - crop_margin, crop_margin : w - crop_margin] = True

    fluid = (mat == int(fluid_value)) & interior
    wall = mat == int(wall_value)

    dudx = mixed_derivative(ux, fluid, axis=1, spacing=dx)
    dudy = mixed_derivative(ux, fluid, axis=0, spacing=dy)
    dvdx = mixed_derivative(uy, fluid, axis=1, spacing=dx)
    dvdy = mixed_derivative(uy, fluid, axis=0, spacing=dy)

    tau_xx = 2.0 * dudx
    tau_yy = 2.0 * dvdy
    tau_xy = dudy + dvdx
    tau_yx = tau_xy

    # Detect wall adjacency for each fluid cell via 4-neighborhood face contacts.
    # Diagonals are not used because traction is accumulated on faces, not corners.
    wall_right = _shift_axis1_plus(wall)
    wall_left = _shift_axis1_minus(wall)
    wall_down = _shift_axis0_plus(wall)
    wall_up = _shift_axis0_minus(wall)

    # Face groups are keyed by fluid-to-wall normal direction in array coordinates.
    face_px = fluid & wall_right   # n=(+1,0)
    face_nx = fluid & wall_left    # n=(-1,0)
    face_py = fluid & wall_down    # n=(0,+1)
    face_ny = fluid & wall_up      # n=(0,-1)

    # Pressure traction: t_p = -p * n, accumulated per face group.
    # Example: if wall is on the right of a fluid node, n=(+1,0), so x-force
    # contribution is -p from that node.
    fx_p = float((-p[face_px]).sum() + (p[face_nx]).sum())
    fy_p = float((-p[face_py]).sum() + (p[face_ny]).sum())

    # Viscous traction: t_v = tau * n, accumulated per face group.
    # tau_xx/tau_yy are normal stresses, tau_xy/tau_yx are shear stresses.
    fx_v_raw = float(
        (tau_xx[face_px]).sum()
        + (-tau_xx[face_nx]).sum()
        + (tau_xy[face_py]).sum()
        + (-tau_xy[face_ny]).sum()
    )
    fy_v_raw = float(
        (tau_yx[face_px]).sum()
        + (-tau_yx[face_nx]).sum()
        + (tau_yy[face_py]).sum()
        + (-tau_yy[face_ny]).sum()
    )

    # Useful diagnostic: how many interface faces contributed at this timestep.
    # It should match between pred/true when the same matnum mask is used.
    faces_count = int(face_px.sum() + face_nx.sum() + face_py.sum() + face_ny.sum())
    return {
        "interface_faces": faces_count,
        "Fx_p": fx_p,
        "Fy_p": fy_p,
        "Fx_v_raw": fx_v_raw,
        "Fy_v_raw": fy_v_raw,
        "Fx_v": float(viscous_prefactor) * fx_v_raw,
        "Fy_v": float(viscous_prefactor) * fy_v_raw,
    }


def compute_series_force_components(
    fields_series: np.ndarray,
    matnum: np.ndarray,
    *,
    p_channel: int = 0,
    ux_channel: int = 1,
    uy_channel: int = 2,
    fluid_value: int = 1,
    wall_value: int = 2,
    crop_margin: int = 3,
    dx: float = 1.0,
    dy: float = 1.0,
    viscous_prefactor: float = 1.0,
) -> list[dict[str, float]]:
    """
    Compute force components independently at each timestep.

    `matnum` can be provided as a static (H,W) map or as a time series; helper
    normalization aligns it to `fields_series` length before looping.
    """
    fields_series = ensure_series_axis(np.asarray(fields_series), name="fields_series")
    if fields_series.ndim != 4:
        raise ValueError(f"fields_series must have shape (T,H,W,C), got {fields_series.shape}.")

    t = fields_series.shape[0]
    mat_series = normalize_matnum_series(matnum, series_len=t)

    # Compute force components independently per timestep, sharing the aligned matnum series.
    rows: list[dict[str, float]] = []
    for i in range(t):
        rows.append(
            compute_snapshot_force_components(
                fields_series[i],
                mat_series[i],
                p_channel=p_channel,
                ux_channel=ux_channel,
                uy_channel=uy_channel,
                fluid_value=fluid_value,
                wall_value=wall_value,
                crop_margin=crop_margin,
                dx=dx,
                dy=dy,
                viscous_prefactor=viscous_prefactor,
            )
        )
    return rows
