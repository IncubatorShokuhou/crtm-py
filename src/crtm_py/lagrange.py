"""Four-point quadratic interpolation from ``CRTM_Interpolation``.

``Find_Random_Index`` selects four nodes. ``LPoly`` blends two overlapping
quadratics. ``Interp_1D`` and ``Interp_2D`` apply those weights. Cloud
optical properties and the NPOESS land reflectance table both use this
scheme.
"""

from __future__ import annotations

import numpy as np


def find_random_index(grid: np.ndarray, value: float) -> tuple[int, float]:
    """Return the 0-based start of the four-node stencil and the clipped value."""
    x = np.asarray(grid, dtype=np.float64)
    n = x.size
    if n < 4:
        raise ValueError("CRTM quadratic interpolation needs at least 4 nodes")
    xi = float(value)
    if xi < x[0] or xi > x[-1]:
        xi = float(np.clip(xi, x[0], x[-1]))
    k0 = int(np.searchsorted(x, xi, side="left"))
    if k0 >= n:
        k0 = n - 1
    i1_f = min(max(1, (k0 + 1) - 2), n - 3)
    return i1_f - 1, xi


def lagrange_weights(nodes, xi: float):
    """``LPoly`` weights for four ascending nodes."""
    nodes = np.asarray(nodes, dtype=np.float64)

    def _poly(x3):
        dxi = xi - x3
        dx1 = x3[0] - x3[1]
        dx2 = x3[0] - x3[2]
        dx3 = x3[1] - x3[2]
        lp = np.array(
            [
                dxi[1] * dxi[2] / (dx1 * dx2),
                dxi[0] * dxi[2] / (-dx1 * dx3),
                dxi[0] * dxi[1] / (dx2 * dx3),
            ]
        )
        return dxi, dx3, lp

    dxi_left, dx3_left, lp_left = _poly(nodes[:3])
    _, _, lp_right = _poly(nodes[1:])
    if xi < nodes[1]:
        w_right = 0.0
    elif xi > nodes[2]:
        w_right = 1.0
    else:
        w_right = dxi_left[1] / (-dx3_left)
    return 1.0 - w_right, w_right, lp_left, lp_right


def interp_1d(values, weights) -> float:
    """``Interp_1D``."""
    w_left, w_right, lp_left, lp_right = weights
    left = lp_left[0] * values[0] + lp_left[1] * values[1] + lp_left[2] * values[2]
    right = lp_right[0] * values[1] + lp_right[1] * values[2] + lp_right[2] * values[3]
    return float(w_left * left + w_right * right)


def interp_2d(block, freq_w, radius_w) -> float:
    """``Interp_2D``: frequency first, then the second coordinate."""
    along_freq = np.array([interp_1d(block[:, i], freq_w) for i in range(4)])
    return interp_1d(along_freq, radius_w)
