"""Pure-Python forward path for UPP v4.1 ABI brightness temperature.

Numerics that UPP computes before ``CRTM_Forward`` follow
``CALRAD_WCLOUD_newcrtm.f`` in NOAA-EMC/UPP tag ``dtc_post_v4.1.0``:
absorber units, hydrometeor column water, ``EFFR``, land-type tables and
the geostationary zenith angle.

When a coefficient directory supplies SpcCoeff and ODPS TauCoeff, gas
optical depth, CloudCoeff cloud optics, Nalli sea emissivity, NPOESS land
emissivity and the emission or ADA solution follow CRTM v2.4. Without
those files the optical depth is a synthetic grey gas and the brightness
temperatures are not CRTM output.
"""

from crtm_py.atmosphere import Atmosphere, atmosphere_from_upp
from crtm_py.clouds import Hydrometeors, effective_radius
from crtm_py.coeffs import coeff_path, coefficient_releases, fetch_coefficients, set_coeff_path
from crtm_py.forward import ForwardModel, ForwardResult, brightness_temperature
from crtm_py.geometry import Geometry, geostationary_zenith
from crtm_py.sensors import ABI_IR_CHANNELS
from crtm_py.surface import Surface, surface_from_upp

__all__ = [
    "ABI_IR_CHANNELS",
    "Atmosphere",
    "ForwardModel",
    "ForwardResult",
    "Geometry",
    "Hydrometeors",
    "Surface",
    "atmosphere_from_upp",
    "brightness_temperature",
    "coeff_path",
    "coefficient_releases",
    "effective_radius",
    "fetch_coefficients",
    "geostationary_zenith",
    "set_coeff_path",
    "surface_from_upp",
]
__version__ = "0.1.0"
