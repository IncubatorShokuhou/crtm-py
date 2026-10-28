"""Land-surface state in the form ``CALRAD`` loads into ``CRTM_Surface``.

Vegetation-type tables are the three ``model_to_crtm`` arrays selected by
``ivegsrc`` (0 USGS, 1 IGBP, 2 old GFS). Category numbers are the WRF
1-based indices. Category 0 is clamped to 1; the USGS Fortran array is
1-based and a 0 index is out of range there.

Coverage fractions follow the water / sea-ice / land split. For IGBP
(``novegtype == 20``) category 15 the four coverages are replaced by
snow = 1, which is the guard that keeps CRTM from rejecting that class.
Skin temperature over water and land both use
``theta_sfc * (p_sfc / 1e5) ** CAPA`` when the caller passes potential
temperature instead of a skin temperature.

``soil_moisture_content`` is ``smstot / 10`` when the value is finite,
else the CRTM default 0.05. ``soil_temperature`` is the constant 283 K
written in ``CALRAD`` (the soil-temperature profile is commented out).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from crtm_py.constants import CAPA, EMIS_ICE, EMIS_LAND, EMIS_SNOW, EMIS_WATER, P1000_PA

INVALID_LAND = 0
COMPACTED_SOIL = 1
TILLED_SOIL = 2
SAND = 3
ROCK = 4
IRRIGATED_LOW_VEGETATION = 5
MEADOW_GRASS = 6
SCRUB = 7
BROADLEAF_FOREST = 8
PINE_FOREST = 9
TUNDRA = 10
GRASS_SOIL = 11
BROADLEAF_PINE_FOREST = 12
GRASS_SCRUB = 13
SOIL_GRASS_SCRUB = 14
URBAN_CONCRETE = 15
PINE_BRUSH = 16
BROADLEAF_BRUSH = 17
WET_SOIL = 18
SCRUB_SOIL = 19
BROADLEAF70_PINE30 = 20

# 1-based category order. Index 0 of each Python tuple is WRF category 1.
_IGBP = (
    PINE_FOREST,
    BROADLEAF_FOREST,
    PINE_FOREST,
    BROADLEAF_FOREST,
    BROADLEAF_PINE_FOREST,
    SCRUB,
    SCRUB_SOIL,
    BROADLEAF_BRUSH,
    BROADLEAF_BRUSH,
    SCRUB,
    BROADLEAF_BRUSH,
    TILLED_SOIL,
    URBAN_CONCRETE,
    TILLED_SOIL,
    INVALID_LAND,
    COMPACTED_SOIL,
    INVALID_LAND,
    TUNDRA,
    TUNDRA,
    TUNDRA,
)
_USGS = (
    URBAN_CONCRETE,
    COMPACTED_SOIL,
    IRRIGATED_LOW_VEGETATION,
    GRASS_SOIL,
    MEADOW_GRASS,
    MEADOW_GRASS,
    MEADOW_GRASS,
    SCRUB,
    GRASS_SCRUB,
    MEADOW_GRASS,
    BROADLEAF_FOREST,
    PINE_FOREST,
    BROADLEAF_FOREST,
    PINE_FOREST,
    BROADLEAF_PINE_FOREST,
    COMPACTED_SOIL,
    WET_SOIL,
    WET_SOIL,
    IRRIGATED_LOW_VEGETATION,
    TUNDRA,
    TUNDRA,
    TUNDRA,
    TUNDRA,
    COMPACTED_SOIL,
)
# GFS table index 0 (water) is unused by the land-type assignment, which
# clamps categories to 1..13. Stored here starting at category 1.
_GFS = (
    BROADLEAF_FOREST,
    BROADLEAF_FOREST,
    BROADLEAF_PINE_FOREST,
    PINE_FOREST,
    PINE_FOREST,
    BROADLEAF_BRUSH,
    SCRUB,
    SCRUB,
    SCRUB_SOIL,
    TUNDRA,
    COMPACTED_SOIL,
    TILLED_SOIL,
    COMPACTED_SOIL,
)
_TABLES = {0: _USGS, 1: _IGBP, 2: _GFS}


def map_land_type(ivgtyp, ivegsrc: int):
    """Map a WRF vegetation category to a CRTM land-type code.

    Parameters
    ----------
    ivgtyp : array_like
        Integer vegetation category. USGS / IGBP / GFS use 1-based ids.
    ivegsrc : int
        0 USGS, 1 IGBP, 2 old GFS. Any other value raises ``ValueError``,
        which is the case where ``CALRAD`` returns without calling CRTM.

    Returns
    -------
    ndarray of int
        CRTM land type. Categories outside the table are clamped.
    """
    try:
        table = _TABLES[int(ivegsrc)]
    except KeyError as exc:
        raise ValueError(f"ivegsrc {ivegsrc} is not 0 (USGS), 1 (IGBP) or 2 (GFS)") from exc
    cat = np.asarray(ivgtyp, dtype=np.int64)
    cat = np.clip(cat, 1, len(table))
    lookup = np.asarray(table, dtype=np.int64)
    return lookup[cat - 1]


def skin_temperature_from_theta(theta_sfc, surface_pressure_pa):
    """``ths * (p_sfc / 1e5) ** CAPA``, the skin temperature in ``CALRAD``."""
    theta = np.asarray(theta_sfc, dtype=np.float64)
    ps = np.asarray(surface_pressure_pa, dtype=np.float64)
    return theta * (ps / P1000_PA) ** CAPA


@dataclass
class Surface:
    """One CRTM surface per profile, shapes ``(n_profiles,)`` unless noted.

    ``emissivity`` is an optional override of shape ``(n_profiles, n_channels)``
    or broadcastable to that. ``None`` selects the constant IR emissivities
    in :mod:`crtm_py.constants`. Those constants are not IRSSE.
    """

    land_type: np.ndarray
    wind_speed: np.ndarray
    water_coverage: np.ndarray
    land_coverage: np.ndarray
    ice_coverage: np.ndarray
    snow_coverage: np.ndarray
    land_temperature: np.ndarray
    snow_temperature: np.ndarray
    water_temperature: np.ndarray
    ice_temperature: np.ndarray
    soil_moisture_content: np.ndarray
    vegetation_fraction: np.ndarray
    soil_temperature: np.ndarray
    snow_depth: np.ndarray
    emissivity: np.ndarray | None = None

    def __post_init__(self) -> None:
        for name in (
            "wind_speed",
            "water_coverage",
            "land_coverage",
            "ice_coverage",
            "snow_coverage",
            "land_temperature",
            "snow_temperature",
            "water_temperature",
            "ice_temperature",
            "soil_moisture_content",
            "vegetation_fraction",
            "soil_temperature",
            "snow_depth",
        ):
            setattr(self, name, np.asarray(getattr(self, name), dtype=np.float64))
        self.land_type = np.asarray(self.land_type, dtype=np.int64)
        if self.emissivity is not None:
            self.emissivity = np.asarray(self.emissivity, dtype=np.float64)

    @property
    def n_profiles(self) -> int:
        return int(self.land_temperature.shape[0])


def _as1d(value, n: int, name: str) -> np.ndarray:
    a = np.asarray(value, dtype=np.float64)
    if a.ndim == 0 or a.size == 1:
        return np.full(n, float(np.reshape(a, ())), dtype=np.float64)
    flat = a.reshape(-1)
    if flat.shape != (n,):
        raise ValueError(f"{name}: expected {n} values, got shape {a.shape}")
    return flat


def surface_from_upp(
    *,
    skin_temperature,
    sea_mask,
    ivgtyp,
    ivegsrc: int = 1,
    novegtype: int | None = None,
    snow_fraction=0.0,
    sea_ice=0.0,
    vegetation_fraction=0.0,
    u10=0.0,
    v10=0.0,
    soil_moisture=np.nan,
    snow_depth=0.0,
    emissivity=None,
) -> Surface:
    """Build a :class:`Surface` the way the non-nadir ``CALRAD`` block does.

    Parameters
    ----------
    skin_temperature : array_like
        Kelvin. For WRF this is ``skin_temperature_from_theta``.
    sea_mask : array_like
        UPP ``sm``. Values above 0.1 are water. This is not the WRF
        ``LANDMASK`` (that one is 1 on land). Pass ``1 - landmask`` for a
        WRF land mask.
    ivgtyp : array_like
        Vegetation category.
    ivegsrc : int
        Vegetation-table selector.
    novegtype : int, optional
        If 20, IGBP category 15 is forced to full snow cover. Defaults to
        the length of the selected table.
    snow_fraction : array_like
        Fraction in ``[0, 1]`` before the water/land split. WRF-ARW (UPP
        model name ``NCAR``) copies ``pctsno`` here. The ``SNFRAC``
        routines are not part of this package.
    sea_ice : array_like
        UPP ``sice``. Values above 0.1 are sea ice.
    vegetation_fraction : array_like
        Set to 0 over water inside this function.
    u10, v10 : array_like
        10 m wind components, m s^-1. Speed is ``hypot(u10, v10)``.
    soil_moisture : array_like
        ``smstot``. Non-finite values become 0.05 after the ``/ 10``
        conversion is skipped.
    snow_depth : array_like
        Millimetres, UPP ``si``. Non-finite values become 0.
    emissivity : array_like, optional
        Forwarded onto :class:`Surface`.

    Returns
    -------
    Surface
        Flat ``(n_profiles,)`` fields. Two-dimensional inputs are raveled
        in C order.
    """
    skin = np.asarray(skin_temperature, dtype=np.float64)
    if skin.ndim == 0:
        n = 1
        skin = np.full(1, float(skin))
    else:
        n = int(skin.size)
        skin = skin.reshape(-1)
    sea = _as1d(sea_mask, n, "sea_mask")
    ice = _as1d(sea_ice, n, "sea_ice")
    snow = _as1d(snow_fraction, n, "snow_fraction")
    veg = _as1d(vegetation_fraction, n, "vegetation_fraction")
    uu = _as1d(u10, n, "u10")
    vv = _as1d(v10, n, "v10")
    sm = _as1d(soil_moisture, n, "soil_moisture")
    depth = _as1d(snow_depth, n, "snow_depth")
    cat = np.asarray(ivgtyp, dtype=np.int64).reshape(-1)
    if cat.size == 1 and n != 1:
        cat = np.full(n, int(cat.reshape(-1)[0]), dtype=np.int64)
    elif cat.shape != (n,):
        raise ValueError(f"ivgtyp: expected {n} values, got shape {cat.shape}")

    table = _TABLES[int(ivegsrc)]
    if novegtype is None:
        novegtype = len(table)
    water = sea > 0.1
    seaice = (~water) & (ice > 0.1)
    land = (~water) & (~seaice)
    snow_on = snow > 0.0
    water_cov = np.where(water, np.where(snow_on, 1.0 - snow, 1.0), 0.0)
    ice_cov = np.where(seaice, np.where(snow_on, 1.0 - snow, 1.0), 0.0)
    land_cov = np.where(land, np.where(snow_on, 1.0 - snow, 1.0), 0.0)
    snow_cov = snow.copy()
    igbp15 = (int(novegtype) == 20) & (cat == 15) & (snow_cov < 1.0)
    water_cov = np.where(igbp15, 0.0, water_cov)
    land_cov = np.where(igbp15, 0.0, land_cov)
    ice_cov = np.where(igbp15, 0.0, ice_cov)
    snow_cov = np.where(igbp15, 1.0, snow_cov)
    veg = np.where(water, 0.0, veg)
    soil = np.where(np.isfinite(sm), sm / 10.0, 0.05)
    depth = np.where(np.isfinite(depth), depth, 0.0)
    return Surface(
        land_type=map_land_type(cat, ivegsrc),
        wind_speed=np.hypot(uu, vv),
        water_coverage=water_cov,
        land_coverage=land_cov,
        ice_coverage=ice_cov,
        snow_coverage=snow_cov,
        land_temperature=skin,
        snow_temperature=np.minimum(skin, 280.0),
        water_temperature=np.maximum(skin, 270.0),
        ice_temperature=np.minimum(skin, 280.0),
        soil_moisture_content=soil,
        vegetation_fraction=veg,
        soil_temperature=np.full(n, 283.0),
        snow_depth=depth,
        emissivity=None if emissivity is None else np.asarray(emissivity, dtype=np.float64),
    )


def surface_emission(surface: Surface, wavenumber_cm, planck_coeffs=None):
    """Coverage-weighted surface emission and reflectance.

    Returns
    -------
    emission, reflectance : ndarray
        Shapes ``(n_profiles, n_channels)``. Emission is
        ``sum(coverage * emissivity * B(T))`` and reflectance is
        ``sum(coverage * (1 - emissivity))``. Negative coverages are
        clipped and the four fractions are renormalized when their sum
        is positive.
    """
    from crtm_py.planck import planck_radiance

    wn = np.asarray(wavenumber_cm, dtype=np.float64).reshape(1, -1)
    n = surface.n_profiles
    nchan = wn.shape[1]
    pieces = (
        (surface.water_coverage, surface.water_temperature, EMIS_WATER),
        (surface.land_coverage, surface.land_temperature, EMIS_LAND),
        (surface.ice_coverage, surface.ice_temperature, EMIS_ICE),
        (surface.snow_coverage, surface.snow_temperature, EMIS_SNOW),
    )
    cover = np.stack([np.clip(c, 0.0, None) for c, _, _ in pieces], axis=0)
    total = cover.sum(axis=0, keepdims=True)
    total = np.where(total > 0.0, total, 1.0)
    weight = cover / total
    emission = np.zeros((n, nchan), dtype=np.float64)
    reflectance = np.zeros((n, nchan), dtype=np.float64)
    if surface.emissivity is None:
        for i, (_, temp, emis) in enumerate(pieces):
            bb = planck_radiance(wn, temp.reshape(-1, 1), planck_coeffs)
            e = np.full((n, 1), emis)
            w = weight[i].reshape(-1, 1)
            emission += w * e * bb
            reflectance += w * (1.0 - e)
        return emission, reflectance
    user = np.broadcast_to(np.asarray(surface.emissivity, dtype=np.float64), (n, nchan))
    reflectance = (1.0 - user).copy()
    for i, (_, temp, _) in enumerate(pieces):
        bb = planck_radiance(wn, temp.reshape(-1, 1), planck_coeffs)
        emission += weight[i].reshape(-1, 1) * user * bb
    return emission, reflectance
