"""CRTM atmosphere profiles filled the way UPP ``CALRAD`` fills them.

Vertical order inside :class:`Atmosphere` is top-down: index 0 is the
top of the model, matching CRTM and UPP. ``atmosphere_from_upp`` accepts
WRF bottom-up arrays and flips them.

Absorbers are the two UPP requests:

* water vapor, mass mixing ratio in g kg^-1,
  ``max(0, q * 1000 / (1 - q))``
* ozone, ppmv, ``max(1e-10, o3_kgkg * 604229)``

WRF-ARW in UPP zeros ozone before that conversion, so a missing ozone
argument becomes the floor ``1e-10`` ppmv.

``level_pressure_hpa[..., 0]`` is CRTM ``TOA_PRESSURE`` (0.005 hPa).
Layer mass for condensate and for the synthetic absorber uses the model
interface pressures, not that substituted top. Adjacent mid-layer
pressures closer than 0.005 hPa are separated by adding 0.005 hPa to
the upper layer, which is the 2018 change in ``CALRAD``.

Convective condensate (``icu_physics == 2``) adds
``cnvcfr * 0.1e-3 * dp / g`` to ice where ``T < 253.15`` and to cloud
water otherwise, between ``htop`` and ``hbot`` when ``hbot - htop > 1``.
With ``vertical_order='bottom_up'`` those indices are WRF 1-based
bottom-up levels and are converted to the top-down index first.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from crtm_py._batch import Layout, normalize_vertical, same_layout
from crtm_py.clouds import (
    Hydrometeors,
    column_water_content,
    effective_radius,
    species_for_microphysics,
)
from crtm_py.constants import CONSTOZ, G, H1000, OZSMALL, QCONV, TOA_PRESSURE_HPA, TRAD_ICE


@dataclass
class Cloud:
    """One CRTM cloud species on a batch of profiles."""

    cloud_type: str
    water_content: np.ndarray
    effective_radius: np.ndarray


@dataclass
class Atmosphere:
    """Top-down CRTM atmosphere.

    Array shapes are ``(n_profiles, n_layers)`` except
    ``level_pressure_hpa`` and ``interface_pressure_pa``, which have a
    level axis of ``n_layers + 1``.
    """

    level_pressure_hpa: np.ndarray
    pressure_hpa: np.ndarray
    temperature: np.ndarray
    h2o_gkg: np.ndarray
    o3_ppmv: np.ndarray
    clouds: tuple[Cloud, ...]
    interface_pressure_pa: np.ndarray
    cloud_fraction: np.ndarray | None = None
    imp_physics: int | None = None
    layout: Layout | None = None

    def __post_init__(self) -> None:
        self.level_pressure_hpa = np.asarray(self.level_pressure_hpa, dtype=np.float64)
        self.pressure_hpa = np.asarray(self.pressure_hpa, dtype=np.float64)
        self.temperature = np.asarray(self.temperature, dtype=np.float64)
        self.h2o_gkg = np.asarray(self.h2o_gkg, dtype=np.float64)
        self.o3_ppmv = np.asarray(self.o3_ppmv, dtype=np.float64)
        self.interface_pressure_pa = np.asarray(self.interface_pressure_pa, dtype=np.float64)
        if self.cloud_fraction is not None:
            self.cloud_fraction = np.asarray(self.cloud_fraction, dtype=np.float64)

    @property
    def n_profiles(self) -> int:
        return int(self.temperature.shape[0])

    @property
    def n_layers(self) -> int:
        return int(self.temperature.shape[1])


def specific_humidity_to_mixing_ratio_gkg(specific_humidity):
    """UPP water-vapor absorber: ``q * 1000 / (1 - q)`` g kg^-1, floored at 0."""
    q = np.asarray(specific_humidity, dtype=np.float64)
    if np.any(q >= 1.0):
        raise ValueError("specific humidity must be < 1 kg/kg")
    return np.maximum(0.0, q * H1000 / (1.0 - q))


def ozone_kgkg_to_ppmv(ozone_kgkg):
    """UPP ozone absorber. ``None`` or zero becomes ``OZSMALL`` (1e-10 ppmv)."""
    if ozone_kgkg is None:
        return None
    o3 = np.asarray(ozone_kgkg, dtype=np.float64)
    return np.maximum(OZSMALL, o3 * CONSTOZ)


def _separate_identical_pressures(pressure_hpa: np.ndarray) -> np.ndarray:
    out = np.array(pressure_hpa, dtype=np.float64, copy=True)
    for k in range(out.shape[1] - 1):
        close = np.abs(out[:, k] - out[:, k + 1]) < 0.005
        out[:, k] = np.where(close, out[:, k] + 0.005, out[:, k])
    return out


def _add_convection(
    water: np.ndarray,
    ice: np.ndarray,
    temperature: np.ndarray,
    interface_pa: np.ndarray,
    hbot,
    htop,
    cnvcfr,
    vertical_order: str,
) -> None:
    """Add BMJ convective condensate. Indices are 1-based."""
    nlay = temperature.shape[1]
    nprof = temperature.shape[0]

    def _per_profile(values, name):
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        if arr.size == 1:
            return np.full(nprof, arr[0])
        if arr.size != nprof:
            raise ValueError(f"{name}: expected {nprof} values or a scalar, got {arr.size}")
        return arr

    bot = np.rint(_per_profile(hbot, "hbot")).astype(np.int64)
    top = np.rint(_per_profile(htop, "htop")).astype(np.int64)
    frac = _per_profile(cnvcfr, "cnvcfr")
    if vertical_order == "bottom_up":
        bot = nlay - bot + 1
        top = nlay - top + 1
    dp = np.diff(interface_pa, axis=-1)
    q_conv = frac * QCONV
    for i in range(temperature.shape[0]):
        if bot[i] - top[i] <= 1:
            continue
        k0 = int(np.clip(top[i], 1, nlay)) - 1
        k1 = int(np.clip(bot[i], 1, nlay)) - 1
        if k1 < k0:
            continue
        extra = dp[i, k0 : k1 + 1] / G * q_conv[i]
        cold = temperature[i, k0 : k1 + 1] < TRAD_ICE
        ice[i, k0 : k1 + 1] += np.where(cold, extra, 0.0)
        water[i, k0 : k1 + 1] += np.where(~cold, extra, 0.0)


def atmosphere_from_upp(
    pressure_interface,
    pressure,
    temperature,
    specific_humidity,
    *,
    ozone=None,
    hydrometeors: Hydrometeors | None = None,
    imp_physics: int = 8,
    vertical_order: str = "bottom_up",
    icu_physics: int = 0,
    hbot=None,
    htop=None,
    cnvcfr=None,
    cloud_fraction=None,
) -> Atmosphere:
    """Build an :class:`Atmosphere` from the arrays UPP feeds CRTM.

    Parameters
    ----------
    pressure_interface : array_like
        Interface pressure in pascals, one longer than ``pressure`` on
        the vertical axis. WRF order has index 0 at the ground when
        ``vertical_order='bottom_up'``.
    pressure : array_like
        Mid-layer pressure in pascals.
    temperature : array_like
        Mid-layer temperature in kelvin.
    specific_humidity : array_like
        kg kg^-1. UPP's ``q`` on WRF-ARW is specific humidity.
    ozone : array_like, optional
        kg kg^-1. Omitted values follow the WRF branch, which zeros
        ozone and then applies the 1e-10 ppmv floor.
    hydrometeors : Hydrometeors, optional
        Grid-mean condensate. Schemes with ``n_clouds = 0`` ignore it.
    imp_physics : int
        WRF microphysics option. Selects the cloud-species list and the
        radius formula.
    vertical_order : {'bottom_up', 'top_down'}
        ``bottom_up`` is a wrfout column. ``top_down`` is UPP after
        ``INITPOST`` has flipped the column.
    icu_physics, hbot, htop, cnvcfr : optional
        Convective cloud (``icu_physics == 2`` only). ``hbot`` / ``htop``
        are 1-based. In bottom-up order they are WRF level indices.
    cloud_fraction : array_like, optional
        Not set by UPP v4.1. Stored for a later overlap scheme. The
        emission solver multiplies cloud optical depth by this fraction
        when it is present.

    Returns
    -------
    Atmosphere
    """
    temp_in = np.asarray(temperature)
    if temp_in.ndim == 1:
        n_layers = int(temp_in.shape[0])
    elif temp_in.ndim == 2:
        n_layers = int(temp_in.shape[1])
    elif temp_in.ndim == 3:
        n_layers = int(temp_in.shape[0])
    else:
        raise ValueError("temperature: expected 1D, 2D or 3D")
    if n_layers < 1:
        raise ValueError("temperature column is empty")

    temperature_td, layout = normalize_vertical(
        temperature, n_levels=n_layers, vertical_order=vertical_order, name="temperature"
    )
    pressure_td, lay_p = normalize_vertical(
        pressure, n_levels=n_layers, vertical_order=vertical_order, name="pressure"
    )
    same_layout(layout, lay_p, "pressure")
    q_td, lay_q = normalize_vertical(
        specific_humidity,
        n_levels=n_layers,
        vertical_order=vertical_order,
        name="specific_humidity",
    )
    same_layout(layout, lay_q, "specific_humidity")
    interface_td, lay_i = normalize_vertical(
        pressure_interface,
        n_levels=n_layers + 1,
        vertical_order=vertical_order,
        name="pressure_interface",
    )
    same_layout(layout, lay_i, "pressure_interface")

    if (
        n_layers > 1
        and np.nanmean(pressure_td[:, 0]) > np.nanmean(pressure_td[:, -1]) + 1.0
    ):
        raise ValueError(
            "layer pressure decreases downward after ordering; "
            "check vertical_order (WRF bottom_top is 'bottom_up')"
        )

    if ozone is None:
        o3_ppmv = np.full_like(temperature_td, OZSMALL)
    else:
        o3_td, lay_o = normalize_vertical(
            ozone, n_levels=n_layers, vertical_order=vertical_order, name="ozone"
        )
        same_layout(layout, lay_o, "ozone")
        o3_ppmv = ozone_kgkg_to_ppmv(o3_td)

    h2o = specific_humidity_to_mixing_ratio_gkg(q_td)
    level_hpa = interface_td / 100.0
    level_hpa = np.array(level_hpa, copy=True)
    level_hpa[:, 0] = TOA_PRESSURE_HPA
    pressure_hpa = _separate_identical_pressures(pressure_td / 100.0)

    hydro = hydrometeors or Hydrometeors()
    species = species_for_microphysics(imp_physics)
    clouds: list[Cloud] = []
    if species:
        fields = {}
        for key in ("qqw", "qqi", "qqr", "qqs", "qqg", "f_rimef", "nlice", "nrain", "qqnr", "qqni"):
            raw = getattr(hydro, key)
            if raw is None:
                fields[key] = np.zeros_like(temperature_td)
            else:
                arr, lay = normalize_vertical(
                    raw, n_levels=n_layers, vertical_order=vertical_order, name=key
                )
                same_layout(layout, lay, key)
                fields[key] = arr
        content = {
            name: column_water_content(fields[key], interface_td)
            for name, key in (
                ("water", "qqw"),
                ("ice", "qqi"),
                ("rain", "qqr"),
                ("snow", "qqs"),
                ("graupel", "qqg"),
            )
        }
        radii = _radii_for_scheme(
            imp_physics,
            species,
            pressure_pa=pressure_td,
            temperature=temperature_td,
            specific_humidity=q_td,
            fields=fields,
        )
        # Ferrier's inline path partitions snow by rime factor and does not
        # use EFFR. Zhao uses fixed radii. Both run before the convective
        # condensate is added, matching the order in CALRAD.
        if imp_physics in (5, 85, 95):
            content, radii = _ferrier_inline(
                fields, interface_td, pressure_td, temperature_td, q_td
            )
        elif imp_physics in (98, 99):
            radii = {
                "water": np.full_like(temperature_td, 10.0),
                "ice": np.full_like(temperature_td, 50.0),
            }
        if icu_physics == 2:
            if hbot is None or htop is None or cnvcfr is None:
                raise ValueError("icu_physics=2 requires hbot, htop and cnvcfr")
            _add_convection(
                content["water"],
                content["ice"],
                temperature_td,
                interface_td,
                hbot,
                htop,
                cnvcfr,
                vertical_order,
            )
        for name in species:
            clouds.append(
                Cloud(
                    cloud_type=name,
                    water_content=content[name],
                    effective_radius=radii[name],
                )
            )

    frac = None
    if cloud_fraction is not None:
        frac, lay_c = normalize_vertical(
            cloud_fraction, n_levels=n_layers, vertical_order=vertical_order, name="cloud_fraction"
        )
        same_layout(layout, lay_c, "cloud_fraction")

    return Atmosphere(
        level_pressure_hpa=level_hpa,
        pressure_hpa=pressure_hpa,
        temperature=temperature_td,
        h2o_gkg=h2o,
        o3_ppmv=o3_ppmv,
        clouds=tuple(clouds),
        interface_pressure_pa=interface_td,
        cloud_fraction=frac,
        imp_physics=int(imp_physics),
        layout=layout,
    )


def _radii_for_scheme(imp_physics, species, *, pressure_pa, temperature, specific_humidity, fields):
    """Radii from ``EFFR``. Ferrier and Zhao replace these values afterwards."""
    letter = {
        "water": "C",
        "ice": "I",
        "rain": "R",
        "snow": "S",
        "graupel": "G",
        "hail": "H",
    }
    out = {}
    for name in species:
        out[name] = effective_radius(
            pressure_pa,
            temperature,
            specific_humidity,
            fields["qqw"],
            fields["qqi"],
            fields["qqr"],
            fields["f_rimef"],
            fields["nlice"],
            fields["nrain"],
            fields["qqs"],
            fields["qqg"],
            fields["qqnr"],
            fields["qqni"],
            imp_physics,
            letter[name],
        )
    return out


def _ferrier_inline(fields, interface_pa, pressure_pa, temperature, specific_humidity):
    """Cloud block for ``imp_physics`` 5, 85 and 95.

    Precipitating ice (``qqs``) is filed as snow, graupel or hail from
    ``f_rimef`` thresholds 5 and 20. Radii use the inline expressions,
    so ice is 75 µm rather than the 25 µm ``EFFR`` returns.
    """
    from crtm_py.clouds import air_density

    content = {
        "water": column_water_content(fields["qqw"], interface_pa),
        "ice": column_water_content(fields["qqi"], interface_pa),
        "rain": column_water_content(fields["qqr"], interface_pa),
        "snow": column_water_content(fields["qqs"], interface_pa),
        "graupel": column_water_content(fields["qqs"], interface_pa),
        "hail": column_water_content(fields["qqs"], interface_pa),
    }
    rime = fields["f_rimef"]
    snow_m = rime <= 5.0
    grau_m = (rime > 5.0) & (rime <= 20.0)
    hail_m = rime > 20.0
    content["snow"] = np.where(snow_m, content["snow"], 0.0)
    content["graupel"] = np.where(grau_m, content["graupel"], 0.0)
    content["hail"] = np.where(hail_m, content["hail"], 0.0)
    rho = air_density(pressure_pa, temperature, specific_humidity)
    nrain = fields["nrain"]
    nlice = fields["nlice"]
    qqr = fields["qqr"]
    qqs = fields["qqs"]
    with np.errstate(divide="ignore", invalid="ignore"):
        rain_re = 1.0e6 * 1.5 * (rho * qqr / (np.pi * 1000.0 * nrain)) ** (1.0 / 3.0)
        snow_re = 1.0e6 * 1.5 * (rho * qqs / (np.pi * 100.0 * nlice)) ** (1.0 / 3.0)
        grau_re = 1.0e6 * 1.5 * (rho * qqs / (np.pi * 400.0 * nlice)) ** (1.0 / 3.0)
        hail_re = 1.0e6 * 1.5 * (rho * qqs / (np.pi * 900.0 * nlice)) ** (1.0 / 3.0)
    radii = {
        "water": np.full_like(temperature, 10.0),
        "ice": np.full_like(temperature, 75.0),
        "rain": np.where((nrain > 0.0) & np.isfinite(rain_re), rain_re, 0.0),
        "snow": np.where(snow_m & (nlice > 0.0) & np.isfinite(snow_re), snow_re, 0.0),
        "graupel": np.where(grau_m & (nlice > 0.0) & np.isfinite(grau_re), grau_re, 0.0),
        "hail": np.where(hail_m & (nlice > 0.0) & np.isfinite(hail_re), hail_re, 0.0),
    }
    return content, radii
