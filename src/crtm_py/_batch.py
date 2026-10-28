"""Pack WRF-shaped arrays into CRTM top-down profile batches."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Layout:
    """How a caller shaped the vertical columns.

    ``profile`` is one column. ``batch`` is ``(n_profiles, n_levels)``
    with the level axis last. ``grid`` is ``(n_levels, ny, nx)``, which
    is the ``upp-py`` WRF state layout.
    """

    kind: str
    n_profiles: int
    ny: int | None = None
    nx: int | None = None

    def restore(self, values: np.ndarray) -> np.ndarray:
        """Map ``(n_profiles, n_channels)`` back to the caller shape.

        A single profile returns ``(n_channels,)``. A grid returns
        ``(ny, nx, n_channels)``.
        """
        values = np.asarray(values)
        if self.kind == "profile":
            return values[0]
        if self.kind == "batch":
            return values
        return values.reshape(self.ny, self.nx, values.shape[-1])


def normalize_vertical(arr, *, n_levels: int, vertical_order: str, name: str) -> tuple[np.ndarray, Layout]:
    """Return a top-down ``(n_profiles, n_levels)`` array and its layout.

    Parameters
    ----------
    arr : array_like
        1D column, 2D ``(n_profiles, n_levels)``, or 3D ``(n_levels, ny, nx)``.
    n_levels : int
        Expected length of the vertical axis.
    vertical_order : {'top_down', 'bottom_up'}
        ``bottom_up`` is WRF ``bottom_top`` (index 0 at the surface).
        ``top_down`` is UPP / CRTM (index 0 at the model top).
    name : str
        Used in error messages.
    """
    if vertical_order not in ("top_down", "bottom_up"):
        raise ValueError("vertical_order must be 'top_down' or 'bottom_up'")
    a = np.asarray(arr, dtype=np.float64)
    flip = vertical_order == "bottom_up"
    if a.ndim == 1:
        if a.shape[0] != n_levels:
            raise ValueError(f"{name}: expected length {n_levels}, got {a.shape[0]}")
        if flip:
            a = a[::-1]
        return np.ascontiguousarray(a.reshape(1, n_levels)), Layout("profile", 1)
    if a.ndim == 2:
        if a.shape[1] != n_levels:
            raise ValueError(
                f"{name}: expected shape (n_profiles, {n_levels}), got {a.shape}"
            )
        if flip:
            a = a[:, ::-1]
        return np.ascontiguousarray(a), Layout("batch", int(a.shape[0]))
    if a.ndim == 3:
        if a.shape[0] != n_levels:
            raise ValueError(
                f"{name}: grid vertical axis 0 must have length {n_levels}, got {a.shape}"
            )
        if flip:
            a = a[::-1, :, :]
        _, ny, nx = a.shape
        flat = np.moveaxis(a, 0, -1).reshape(ny * nx, n_levels)
        return np.ascontiguousarray(flat), Layout("grid", ny * nx, ny, nx)
    raise ValueError(f"{name}: expected 1D, 2D or 3D, got ndim {a.ndim}")


def same_layout(left: Layout, right: Layout, name: str) -> None:
    if left != right:
        raise ValueError(f"{name}: layout {right} does not match {left}")


def normalize_horizontal(arr, layout: Layout, name: str) -> np.ndarray:
    """Broadcast a surface/horizontal field onto ``layout`` and ravel it."""
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim == 0 or a.size == 1:
        return np.full(layout.n_profiles, float(np.reshape(a, ())), dtype=np.float64)
    if layout.kind == "grid" and a.shape == (layout.ny, layout.nx):
        return np.ascontiguousarray(a.reshape(-1))
    flat = a.reshape(-1)
    if flat.shape == (layout.n_profiles,):
        return np.ascontiguousarray(flat)
    raise ValueError(
        f"{name}: cannot broadcast shape {a.shape} onto {layout.n_profiles} profiles"
    )
