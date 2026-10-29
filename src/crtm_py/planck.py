"""Monochromatic Planck radiance and brightness temperature.

The constants and the algebraic form follow CRTM ``Planck_Radiance`` /
``Planck_Temperature``. When ``planck_coeffs`` is provided it is the
two-coefficient polychromatic correction stored on a SpcCoeff record::

    T_channel = b0 + b1 * T_monochromatic

Radiance is inverted through that linear map before the Planck function.
Coefficients ``(0, 1)`` leave the monochromatic result unchanged. This
module does not read SpcCoeff files.
"""

from __future__ import annotations

import numpy as np

from crtm_py.constants import PLANCK_C1, PLANCK_C2


def planck_radiance(frequency_cm, temperature, planck_coeffs=None):
    """Spectral radiance in mW m^-2 sr^-1 (cm^-1)^-1.

    Parameters
    ----------
    frequency_cm : array_like
        Wavenumber in cm^-1. Must be positive.
    temperature : array_like
        Temperature in kelvin. Must be positive. Broadcast with
        ``frequency_cm``.
    planck_coeffs : array_like, optional
        Shape ``(2,)`` or broadcastable to ``(..., 2)`` as ``(b0, b1)``.

    Returns
    -------
    ndarray
        Radiance. Non-positive temperatures yield NaN.
    """
    freq = np.asarray(frequency_cm, dtype=np.float64)
    temp = np.asarray(temperature, dtype=np.float64)
    freq, temp = np.broadcast_arrays(freq, temp)
    mono = temp
    if planck_coeffs is not None:
        coeff = np.asarray(planck_coeffs, dtype=np.float64)
        b0 = coeff[..., 0]
        b1 = coeff[..., 1]
        mono = (temp - b0) / b1
    radiance = np.full(freq.shape, np.nan, dtype=np.float64)
    ok = (freq > 0.0) & (mono > 0.0)
    if not np.any(ok):
        return radiance
    f = freq[ok]
    x = PLANCK_C2 * f / mono[ok]
    # expm1 overflows for the cosmic background at infrared wavenumbers.
    # The radiance is then far below any infrared signal.
    safe = x < 100.0
    out = np.zeros(f.shape, dtype=np.float64)
    out[safe] = PLANCK_C1 * f[safe] ** 3 / np.expm1(x[safe])
    radiance[ok] = out
    return radiance


def planck_temperature(frequency_cm, radiance, planck_coeffs=None):
    """Brightness temperature in kelvin.

    Parameters
    ----------
    frequency_cm : array_like
        Wavenumber in cm^-1.
    radiance : array_like
        Spectral radiance in mW m^-2 sr^-1 (cm^-1)^-1.
    planck_coeffs : array_like, optional
        ``(b0, b1)`` applied as ``b0 + b1 * T_mono`` after the inverse
        Planck function.

    Returns
    -------
    ndarray
        Brightness temperature. Non-positive radiance yields NaN.
    """
    freq = np.asarray(frequency_cm, dtype=np.float64)
    rad = np.asarray(radiance, dtype=np.float64)
    freq, rad = np.broadcast_arrays(freq, rad)
    out = np.full(freq.shape, np.nan, dtype=np.float64)
    ok = (freq > 0.0) & (rad > 0.0)
    if not np.any(ok):
        return out
    f = freq[ok]
    t_mono = PLANCK_C2 * f / np.log(PLANCK_C1 * f**3 / rad[ok] + 1.0)
    if planck_coeffs is None:
        out[ok] = t_mono
        return out
    coeff = np.asarray(planck_coeffs, dtype=np.float64)
    b0 = np.broadcast_to(np.asarray(coeff[..., 0], dtype=np.float64), freq.shape)
    b1 = np.broadcast_to(np.asarray(coeff[..., 1], dtype=np.float64), freq.shape)
    out[ok] = b0[ok] + b1[ok] * t_mono
    return out
