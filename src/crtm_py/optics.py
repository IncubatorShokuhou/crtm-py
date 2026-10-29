"""Layer optical depth for the emission solver.

Two pieces are added:

* ``synthetic_grey_gas`` — mass absorption coefficients times the layer
  column of dry air, water vapor and ozone. The coefficients are
  order-of-magnitude weights so that ABI band 8 (upper-tropospheric
  water vapor) is opaque, band 14 (window) is not, band 12 responds to
  ozone and band 16 responds to a dry-air (CO2) background. They are not
  CRTM ODPS or ODAS predictors and they do not use ``TauCoeff.bin``.
* ``geometric_optics_cloud`` — extinction ``3 Qext / (4 rho re)`` with
  ``Qext = 2``, times the grid-mean water content. Single-scatter albedo
  is zero, so the emission solver treats the extinction as absorption.
  This is not the CloudCoeff lookup table. A positive water content with
  a zero radius uses :data:`crtm_py.constants.CLOUD_RADIUS_FLOOR_UM`.

The sum is the nadir absorption optical depth. The solver applies the
zenith secant.
"""

from __future__ import annotations

import numpy as np

from crtm_py.atmosphere import Atmosphere
from crtm_py.constants import CLOUD_DENSITY, CLOUD_RADIUS_FLOOR_UM, G, O3_TO_DRY_MASS

# (k_h2o, k_o3, k_dry) in m^2 kg^-1 for each ABI IR band.
SYNTHETIC_MASS_ABSORPTION: dict[int, tuple[float, float, float]] = {
    7: (0.008, 0.0, 1.2e-5),
    8: (3.5, 0.0, 2.0e-5),
    9: (1.2, 0.0, 2.0e-5),
    10: (0.45, 0.0, 2.0e-5),
    11: (0.02, 0.0, 1.5e-5),
    12: (0.015, 400.0, 1.5e-5),
    13: (0.008, 0.0, 1.2e-5),
    14: (0.006, 0.0, 1.0e-5),
    15: (0.012, 0.0, 1.5e-5),
    16: (0.02, 0.0, 1.8e-4),
}

OPTICAL_DEPTH_MODEL = "synthetic_grey_gas+geometric_optics_cloud"


def layer_columns(atmosphere: Atmosphere):
    """Dry-air, water-vapor and ozone column amounts in kg m^-2.

    Hydrostatic mass uses the model interface pressures (pascals), so the
    CRTM top-of-atmosphere substitute (0.005 hPa) does not inflate the
    top layer. Water vapor is the dry-air mass times the g kg^-1 mixing
    ratio. Ozone ppmv is converted with :data:`O3_TO_DRY_MASS`.
    """
    pint = atmosphere.interface_pressure_pa
    mass = np.diff(pint, axis=-1) / G
    mmr = atmosphere.h2o_gkg / 1000.0
    dry = mass / (1.0 + mmr)
    h2o = mmr * dry
    o3 = atmosphere.o3_ppmv * 1.0e-6 * O3_TO_DRY_MASS * dry
    return dry, h2o, o3


def gas_optical_depth(atmosphere: Atmosphere, channels: tuple[int, ...]) -> np.ndarray:
    """Nadir gaseous absorption optical depth, shape ``(n, n_layers, n_channels)``."""
    dry, h2o, o3 = layer_columns(atmosphere)
    tau = np.empty(dry.shape + (len(channels),), dtype=np.float64)
    for i, band in enumerate(channels):
        try:
            k_h2o, k_o3, k_dry = SYNTHETIC_MASS_ABSORPTION[int(band)]
        except KeyError as exc:
            raise KeyError(f"no synthetic absorption for ABI band {band}") from exc
        tau[..., i] = k_h2o * h2o + k_o3 * o3 + k_dry * dry
    return np.maximum(tau, 0.0)


def cloud_optical_depth(atmosphere: Atmosphere, n_channels: int) -> np.ndarray:
    """Nadir cloud absorption optical depth, same value on every channel."""
    n, nlay = atmosphere.temperature.shape
    tau = np.zeros((n, nlay), dtype=np.float64)
    for cloud in atmosphere.clouds:
        rho = CLOUD_DENSITY[cloud.cloud_type]
        floor = CLOUD_RADIUS_FLOOR_UM[cloud.cloud_type]
        re = np.where(cloud.effective_radius > 0.0, cloud.effective_radius, floor)
        re_m = re * 1.0e-6
        k_ext = 3.0 * 2.0 / (4.0 * rho * re_m)
        layer = k_ext * np.maximum(cloud.water_content, 0.0)
        layer = np.where(cloud.water_content > 0.0, layer, 0.0)
        tau += layer
    if atmosphere.cloud_fraction is not None:
        tau = tau * atmosphere.cloud_fraction
    return np.broadcast_to(tau[..., None], (n, nlay, n_channels)).copy()


def total_optical_depth(atmosphere: Atmosphere, channels: tuple[int, ...], *, include_clouds: bool = True) -> np.ndarray:
    """Nadir absorption optical depth used by the emission solver."""
    tau = gas_optical_depth(atmosphere, channels)
    if include_clouds and atmosphere.clouds:
        tau = tau + cloud_optical_depth(atmosphere, len(channels))
    return tau
