"""Physical constants used by UPP v4.1 ``CALRAD_WCLOUD_newcrtm.f``.

Values match ``sorc/ncep_post.fd/params.F`` in NOAA-EMC/UPP tag
``dtc_post_v4.1.0``, plus the CRTM parameters that routine imports
(``TOA_PRESSURE``, ``MAX_SENSOR_SCAN_ANGLE``). Float64 is used throughout;
UPP default ``REAL`` is typically 32-bit, so results are not bit-identical.
"""

from __future__ import annotations

# params.F
G = 9.81
RD = 287.04
D608 = 0.608
PI = 3.141592653589793
CAPA = 0.28589641
P1000_PA = 1000.0e2
H1000 = 1000.0
H1 = 1.0
# RG = 1/GI and GI = 1/G, so RG equals gravity. EFFR's Thompson graupel
# branch divides by this symbol.
RG = G

# CALRAD_WCLOUD_newcrtm.f
# ppmv per (kg/kg): 1e6 * M_dry / M_ozone ≈ 1e6 * 28.964 / 48.
CONSTOZ = 604229.0
OZSMALL = 1.0e-10
QCONV = 0.1e-3
# Comment in the convective-cloud block: TRAD_ice = 253.15 K.
TRAD_ICE = 253.15

# CRTM_Parameters (the names UPP imports from crtm_module).
TOA_PRESSURE_HPA = 0.005
MAX_SENSOR_SCAN_ANGLE = 80.0
# Thermal downwelling on a Lambertian surface. CRTM uses 5/3.
SECANT_DIFFUSIVITY = 5.0 / 3.0
# Clip exponent so exp(-tau/mu) underflows cleanly.
LIMIT_EXP = 80.0

# Planck function, CRTM_Planck_Functions: frequency in cm^-1,
# radiance in mW m^-2 sr^-1 (cm^-1)^-1.
PLANCK_C1 = 1.191042722e-5
PLANCK_C2 = 1.4387752
COSMIC_BACKGROUND_TEMPERATURE = 2.7253

# Molecular weight ratio M_O3 / M_dry used to turn ppmv into a mass
# mixing ratio inside the synthetic absorber. Not used by UPP itself.
O3_TO_DRY_MASS = 47.9982 / 28.9644

# Default IR emissivities. These are not the CRTM IRSSE / IRwater LUTs.
EMIS_WATER = 0.98
EMIS_LAND = 0.97
EMIS_ICE = 0.98
EMIS_SNOW = 0.99

# Geometric-optics bulk densities (kg m^-3) for the placeholder cloud
# extinction. Thompson/WSM6 particle densities from EFFR, not CloudCoeff.
CLOUD_DENSITY = {
    "water": 1000.0,
    "ice": 890.0,
    "rain": 1000.0,
    "snow": 100.0,
    "graupel": 500.0,
    "hail": 900.0,
}

# Floor radius (microns) used only when a positive water content arrives
# with a zero effective radius. Stored radii are left unchanged.
CLOUD_RADIUS_FLOOR_UM = {
    "water": 5.0,
    "ice": 10.0,
    "rain": 50.0,
    "snow": 50.0,
    "graupel": 50.0,
    "hail": 100.0,
}
