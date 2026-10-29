"""Plane-parallel thermal emission with isothermal layers.

This is the no-scattering limit of the CRTM forward model: each layer is
isothermal at the CRTM layer temperature, optical depth scales with the
secant of the sensor zenith, and the surface is Lambertian. Downwelling
at the surface uses the diffusivity secant 5/3 rather than the sensor
secant. The top boundary is the Planck radiance at 2.7253 K.

Scattering (CRTM ADA / SOI), cloud overlap beyond a per-layer fraction,
and reflected solar radiance are not in this solver.
"""

from __future__ import annotations

import numpy as np

from crtm_py.constants import COSMIC_BACKGROUND_TEMPERATURE, LIMIT_EXP, SECANT_DIFFUSIVITY
from crtm_py.planck import planck_radiance, planck_temperature


def _transmittance(tau: np.ndarray, secant: np.ndarray) -> np.ndarray:
    """``exp(-tau * secant)`` with the path optical depth clipped."""
    # tau (n, nlay, nchan), secant (n, 1, 1)
    path = np.minimum(tau * secant, LIMIT_EXP)
    return np.exp(-path)


def emission_radiance(
    temperature,
    tau,
    sensor_zenith_deg,
    surface_emission,
    surface_reflectance,
    wavenumber_cm,
    planck_coeffs=None,
):
    """Upwelling spectral radiance at the top of the atmosphere.

    Parameters
    ----------
    temperature : ndarray
        Layer temperature, shape ``(n_profiles, n_layers)``, top-down.
    tau : ndarray
        Nadir absorption optical depth, shape ``(n_profiles, n_layers, n_channels)``.
    sensor_zenith_deg : ndarray
        Shape ``(n_profiles,)``. Callers should already have rejected
        angles outside the UPP window; this function still requires
        ``cos(zenith) > 0``.
    surface_emission, surface_reflectance : ndarray
        Shapes ``(n_profiles, n_channels)``. See
        :func:`crtm_py.surface.surface_emission`.
    wavenumber_cm : ndarray
        Shape ``(n_channels,)``.
    planck_coeffs : array_like, optional
        Per-channel ``(b0, b1)`` polychromatic correction.

    Returns
    -------
    ndarray
        Radiance, shape ``(n_profiles, n_channels)``.
    """
    temp = np.asarray(temperature, dtype=np.float64)
    tau = np.asarray(tau, dtype=np.float64)
    zen = np.asarray(sensor_zenith_deg, dtype=np.float64)
    wn = np.asarray(wavenumber_cm, dtype=np.float64)
    n, nlay = temp.shape
    mu = np.cos(np.deg2rad(zen))
    if np.any(mu <= 0.0):
        raise ValueError("sensor zenith must be below 90 degrees inside the emission solver")
    secant_up = (1.0 / mu).reshape(n, 1, 1)
    secant_down = np.full((n, 1, 1), SECANT_DIFFUSIVITY)
    trans_up = _transmittance(tau, secant_up)
    trans_down = _transmittance(tau, secant_down)
    layer_b = planck_radiance(wn.reshape(1, 1, -1), temp[..., None], planck_coeffs)
    cosmic = planck_radiance(wn, COSMIC_BACKGROUND_TEMPERATURE, planck_coeffs)
    down = np.broadcast_to(np.asarray(cosmic, dtype=np.float64).reshape(1, -1), (n, wn.size)).copy()
    for k in range(nlay):
        down = down * trans_down[:, k, :] + layer_b[:, k, :] * (1.0 - trans_down[:, k, :])
    up = np.asarray(surface_emission, dtype=np.float64) + np.asarray(surface_reflectance, dtype=np.float64) * down
    for k in range(nlay - 1, -1, -1):
        up = up * trans_up[:, k, :] + layer_b[:, k, :] * (1.0 - trans_up[:, k, :])
    return up


def radiance_to_brightness_temperature(radiance, wavenumber_cm, planck_coeffs=None):
    """Inverse Planck wrapper used by the forward model."""
    return planck_temperature(wavenumber_cm.reshape(1, -1), radiance, planck_coeffs)
