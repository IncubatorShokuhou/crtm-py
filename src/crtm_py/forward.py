"""Forward brightness temperature for the ABI channels UPP posts.

The public entry point is :func:`brightness_temperature`. It builds the
UPP atmosphere and surface and the geostationary zenith, then integrates.

When the coefficient directory contains ``SpcCoeff`` and an ODPS
``TauCoeff`` for the sensor, gas optical depth is ODPS, cloud optics come
from ``CloudCoeff``, and sea / land emissivity come from Nalli and NPOESS.
``include_scattering=False`` keeps the specular emission solution.
``include_scattering=True`` runs ADA. Without those files the call falls
back to the synthetic grey gas and geometric-optics extinction.

Profiles UPP would skip (sensor zenith outside ``[0, 80]``, or a
non-finite temperature or pressure) are returned as NaN. UPP writes its
``spval`` into those columns; NaN is the array equivalent.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, replace

import numpy as np

from crtm_py._batch import normalize_horizontal
from crtm_py.atmosphere import Atmosphere, atmosphere_from_upp
from crtm_py.clouds import Hydrometeors
from crtm_py.coeffs import coeff_path as configured_coeff_path
from crtm_py.crtm_solver import brightness_crtm, load_sensor
from crtm_py.emission import emission_radiance, radiance_to_brightness_temperature
from crtm_py.geometry import Geometry, geostationary_zenith
from crtm_py.optics import OPTICAL_DEPTH_MODEL, total_optical_depth
from crtm_py.sensors import Sensor, get_sensor, resolve_channels
from crtm_py.surface import COMPACTED_SOIL, Surface, surface_from_upp

_SYNTHETIC_WARNING = (
    "crtm_py optical depths are the synthetic grey-gas model plus "
    "geometric-optics cloud extinction, not CRTM ODPS/CloudCoeff. "
    "Brightness temperatures will not match unipost.exe."
)
_warned = False


def reset_synthetic_warning() -> None:
    """Allow the synthetic-model warning to be issued again. Used by tests."""
    global _warned
    _warned = False


def _warn_once() -> None:
    global _warned
    if not _warned:
        warnings.warn(_SYNTHETIC_WARNING, UserWarning, stacklevel=3)
        _warned = True


@dataclass
class ForwardResult:
    """Brightness temperature and the radiance it was inverted from.

    Attributes
    ----------
    brightness_temperature : ndarray
        Kelvin. Shape follows the input layout: ``(n_channels,)`` for one
        column, ``(n_profiles, n_channels)`` for a batch, or
        ``(ny, nx, n_channels)`` for a grid.
    radiance : ndarray
        mW m^-2 sr^-1 (cm^-1)^-1, same shape as the brightness temperature.
    channels : tuple of int
        ABI band numbers, in the requested order.
    sensor_id : str
    wavenumber_cm : ndarray
        Central wavenumber for each returned channel.
    optical_depth_model : str
        ``odps+specular_emission`` or ``odps+cloudcoeff+ada`` when
        official coefficients loaded. Otherwise
        ``synthetic_grey_gas+geometric_optics_cloud``.
    valid : ndarray
        Boolean mask of profiles that were integrated, in batch layout
        (not restored to the grid).
    notes : tuple of str
        Limitations that apply to this call.
    """

    brightness_temperature: np.ndarray
    radiance: np.ndarray
    channels: tuple[int, ...]
    sensor_id: str
    wavenumber_cm: np.ndarray
    optical_depth_model: str
    valid: np.ndarray
    notes: tuple[str, ...]


class ForwardModel:
    """Sensor-fixed forward operator.

    Parameters
    ----------
    sensor : str
        ``abi_g16``, ``abi_g17`` or ``abi_gr``.
    channels : sequence of int, optional
        ABI band numbers. The default is 7 through 16.
    coeff_path : path-like, optional
        Directory of official CRTM coefficient files. When it contains
        this sensor's ``SpcCoeff.bin`` and ODPS ``TauCoeff.bin`` those
        files are read. A missing pair keeps the synthetic optical depth.
    include_clouds : bool
        Add cloud optical depth. Gas optical depth is always included.
    """

    def __init__(self, sensor: str, channels=None, coeff_path=None, include_clouds: bool = True):
        self.sensor: Sensor = get_sensor(sensor)
        self.channels = resolve_channels(self.sensor, None if channels is None else tuple(channels))
        self.coeff_path = coeff_path
        self.include_clouds = include_clouds

    def run(
        self,
        atmosphere: Atmosphere,
        surface: Surface,
        geometry: Geometry,
        *,
        include_scattering: bool = False,
    ) -> ForwardResult:
        """Integrate one batch. Arrays on the three arguments are top-down."""
        if atmosphere.n_profiles != surface.n_profiles:
            raise ValueError(
                f"atmosphere has {atmosphere.n_profiles} profiles and surface has {surface.n_profiles}"
            )
        if geometry.sensor_zenith_angle.shape != (atmosphere.n_profiles,):
            raise ValueError("sensor zenith must have one angle per profile")
        loaded = _loaded_coefficients(self.sensor.sensor_id, self.coeff_path)
        if loaded is not None:
            return _run_crtm(self, atmosphere, surface, geometry, loaded, include_scattering)
        _warn_once()
        wn = np.array([self.sensor.channels[c] for c in self.channels], dtype=np.float64)
        valid = geometry.valid_zenith & np.isfinite(atmosphere.temperature).all(axis=1)
        valid = valid & np.isfinite(atmosphere.pressure_hpa).all(axis=1)
        tau = total_optical_depth(atmosphere, self.channels, include_clouds=self.include_clouds)
        # Invalid columns still need a finite secant for the vectorized loop.
        zenith = np.where(valid, geometry.sensor_zenith_angle, 0.0)
        emission, reflectance = surface_emission_for(surface, wn)
        radiance = emission_radiance(
            atmosphere.temperature,
            tau,
            zenith,
            emission,
            reflectance,
            wn,
        )
        tb = radiance_to_brightness_temperature(radiance, wn)
        radiance = np.where(valid[:, None], radiance, np.nan)
        tb = np.where(valid[:, None], tb, np.nan)
        notes = [_SYNTHETIC_WARNING]
        if self.coeff_path is not None:
            notes.append(
                f"coefficient path {self.coeff_path} did not yield SpcCoeff and ODPS TauCoeff; "
                "the synthetic optical depth is still used"
            )
        if include_scattering:
            notes.append("include_scattering has no effect without SpcCoeff and ODPS TauCoeff")
        notes.append("ABI channel 7 has no reflected solar term")
        notes.append("surface emissivity is a coverage-weighted constant, not IRSSE or IRwater")
        layout = atmosphere.layout
        if layout is not None:
            tb_out = layout.restore(tb)
            rad_out = layout.restore(radiance)
        else:
            tb_out, rad_out = tb, radiance
        return ForwardResult(
            brightness_temperature=tb_out,
            radiance=rad_out,
            channels=self.channels,
            sensor_id=self.sensor.sensor_id,
            wavenumber_cm=wn,
            optical_depth_model=OPTICAL_DEPTH_MODEL,
            valid=valid,
            notes=tuple(notes),
        )


def _loaded_coefficients(sensor_id: str, explicit):
    root = explicit if explicit is not None else configured_coeff_path()
    if root is None:
        return None
    try:
        return load_sensor(sensor_id, root)
    except (OSError, ValueError):
        return None


def _run_crtm(model, atmosphere, surface, geometry, loaded, include_scattering: bool):
    if not model.include_clouds:
        atmosphere = replace(atmosphere, clouds=())
    valid = geometry.valid_zenith & np.isfinite(atmosphere.temperature).all(axis=1)
    valid = valid & np.isfinite(atmosphere.pressure_hpa).all(axis=1)
    zenith = geometry.sensor_zenith_angle.copy()
    zenith = np.where(valid, zenith, 0.0)
    safe_geometry = type(geometry)(zenith, geometry.solar_zenith_angle, geometry.sensor_scan_angle)
    tb, rad = brightness_crtm(
        loaded,
        atmosphere,
        surface,
        safe_geometry,
        model.channels,
        scattering=include_scattering,
    )
    tb = np.where(valid[:, None], tb, np.nan)
    rad = np.where(valid[:, None], rad, np.nan)
    wn = np.array([loaded.spc.wavenumber[loaded.spc.channel_index(c)] for c in model.channels])
    notes = [
        "gas optical depth is ODPS (group "
        f"{loaded.odps.group}), including compact OPTRAN where OSignificance is 1",
        "cloud optics use CloudCoeff IR mass extinction, albedo and Legendre terms"
        if loaded.cloud is not None
        else "CloudCoeff.bin was not found; clouds are omitted",
        "sea emissivity is Nalli IRSSEM (Wu-Smith if that is the file present); land emissivity is NPOESS"
        if loaded.water is not None and loaded.land is not None
        else "Wu-Smith or NPOESS coefficients were not both found",
        "ice and snow emissivity are still the constant infrared values",
        "daytime solar reflection on ABI channel 7 is not applied",
        "ADA is used when scattering is requested; otherwise the specular emission branch is used",
    ]
    layout = atmosphere.layout
    if layout is not None:
        tb = layout.restore(tb)
        rad = layout.restore(rad)
    return ForwardResult(
        brightness_temperature=tb,
        radiance=rad,
        channels=model.channels,
        sensor_id=model.sensor.sensor_id,
        wavenumber_cm=wn,
        optical_depth_model="odps+cloudcoeff+ada" if include_scattering else "odps+specular_emission",
        valid=valid,
        notes=tuple(notes),
    )


def surface_emission_for(surface: Surface, wavenumber_cm):
    from crtm_py.surface import surface_emission

    return surface_emission(surface, wavenumber_cm)


def _simple_land(n: int, skin: np.ndarray, emissivity) -> Surface:
    return Surface(
        land_type=np.full(n, COMPACTED_SOIL, dtype=np.int64),
        wind_speed=np.zeros(n),
        water_coverage=np.zeros(n),
        land_coverage=np.ones(n),
        ice_coverage=np.zeros(n),
        snow_coverage=np.zeros(n),
        land_temperature=skin,
        snow_temperature=np.minimum(skin, 280.0),
        water_temperature=np.maximum(skin, 270.0),
        ice_temperature=np.minimum(skin, 280.0),
        soil_moisture_content=np.full(n, 0.05),
        vegetation_fraction=np.zeros(n),
        soil_temperature=np.full(n, 283.0),
        snow_depth=np.zeros(n),
        emissivity=emissivity,
    )


def brightness_temperature(
    pressure_interface,
    pressure,
    temperature,
    specific_humidity,
    *,
    sensor: str = "abi_g16",
    channels=None,
    vertical_order: str = "bottom_up",
    ozone=None,
    hydrometeors: Hydrometeors | None = None,
    imp_physics: int = 8,
    surface: Surface | None = None,
    latitude=None,
    longitude=None,
    sensor_zenith=None,
    solar_zenith=None,
    emissivity=None,
    skin_temperature=None,
    sea_mask=None,
    landmask=None,
    snow_fraction=0.0,
    sea_ice=0.0,
    vegetation_fraction=0.0,
    u10=0.0,
    v10=0.0,
    ivgtyp=1,
    ivegsrc: int = 1,
    icu_physics: int = 0,
    hbot=None,
    htop=None,
    cnvcfr=None,
    cloud_fraction=None,
    include_clouds: bool = True,
    include_scattering: bool = False,
    coeff_path=None,
) -> ForwardResult:
    """Brightness temperature for one column, a batch, or a WRF grid.

    Parameters
    ----------
    pressure_interface, pressure : array_like
        Pascals. The interface array is one level longer than ``pressure``.
        With the default ``vertical_order='bottom_up'`` (wrfout), index 0
        is the ground. Pass ``vertical_order='top_down'`` for arrays that
        already follow UPP / CRTM.
    temperature : array_like
        Kelvin, mid-layer.
    specific_humidity : array_like
        kg kg^-1.
    sensor : str
        ``abi_g16``, ``abi_g17`` or ``abi_gr``.
    channels : sequence of int, optional
        ABI bands. Default 7–16.
    ozone : array_like, optional
        kg kg^-1. Omitted ozone becomes 1e-10 ppmv, as in the WRF branch
        of ``CALRAD``.
    hydrometeors, imp_physics : optional
        Condensate and the microphysics option that selects radii.
        ``imp_physics=0`` (or any scheme UPP does not list) is clear sky.
    surface : Surface, optional
        Pre-built surface in flattened profile order. When omitted, a
        land column at ``skin_temperature`` (or the lowest-layer
        temperature) is used, unless ``sea_mask`` or ``landmask`` is set,
        in which case :func:`surface_from_upp` is called.
    latitude, longitude : array_like, optional
        Degrees, positive east. Used to compute the geostationary zenith
        when ``sensor_zenith`` is omitted. ``abi_gr`` stays at nadir
        unless ``sensor_zenith`` is passed. With neither latitude nor a
        zenith angle the column is treated as nadir.
    sensor_zenith, solar_zenith : array_like, optional
        Degrees. Solar zenith is stored and not used (no channel-7 solar
        reflection yet).
    skin_temperature : array_like, optional
        Kelvin. Overrides the lowest-layer temperature for the default
        land surface.
    sea_mask, landmask : array_like, optional
        ``sea_mask`` matches UPP ``sm`` (water where the value exceeds
        0.1). ``landmask`` matches WRF (1 on land) and is converted with
        ``sea_mask = 1 - landmask`` when ``sea_mask`` is omitted.
    emissivity : array_like, optional
        Overrides the constant IR emissivities.
    include_clouds : bool
        When false, hydrometeor optical depth is omitted.
    include_scattering : bool
        With official coefficients, run ADA. The default is the
        no-scattering emission solution. Without coefficients the flag
        is ignored and recorded on ``notes``.
    coeff_path : path-like, optional
        Directory of CRTM coefficient binaries. Falls back to
        :func:`crtm_py.coeffs.coeff_path`.

    Returns
    -------
    ForwardResult
        ``brightness_temperature`` is in kelvin.

    Notes
    -----
    Profile conversion, effective radius and zenith angle follow
    ``CALRAD``. With SpcCoeff and ODPS TauCoeff the radiance follows
    CRTM v2.4 for the loaded channels. Without them the optical depth is
    a synthetic grey gas plus geometric-optics cloud extinction, and the
    temperatures are not CRTM output.
    """
    atmosphere = atmosphere_from_upp(
        pressure_interface,
        pressure,
        temperature,
        specific_humidity,
        ozone=ozone,
        hydrometeors=hydrometeors,
        imp_physics=imp_physics,
        vertical_order=vertical_order,
        icu_physics=icu_physics,
        hbot=hbot,
        htop=htop,
        cnvcfr=cnvcfr,
        cloud_fraction=cloud_fraction,
    )
    layout = atmosphere.layout
    assert layout is not None
    n = atmosphere.n_profiles
    if surface is None:
        if skin_temperature is None:
            skin = atmosphere.temperature[:, -1].copy()
        else:
            skin = normalize_horizontal(skin_temperature, layout, "skin_temperature")
        if sea_mask is not None or landmask is not None:
            if sea_mask is None:
                sea = 1.0 - normalize_horizontal(landmask, layout, "landmask")
            else:
                sea = normalize_horizontal(sea_mask, layout, "sea_mask")
            surface = surface_from_upp(
                skin_temperature=skin,
                sea_mask=sea,
                ivgtyp=normalize_horizontal(ivgtyp, layout, "ivgtyp"),
                ivegsrc=ivegsrc,
                snow_fraction=normalize_horizontal(snow_fraction, layout, "snow_fraction"),
                sea_ice=normalize_horizontal(sea_ice, layout, "sea_ice"),
                vegetation_fraction=normalize_horizontal(
                    vegetation_fraction, layout, "vegetation_fraction"
                ),
                u10=normalize_horizontal(u10, layout, "u10"),
                v10=normalize_horizontal(v10, layout, "v10"),
                emissivity=emissivity,
            )
        else:
            surface = _simple_land(n, skin, emissivity)
    sensor_obj = get_sensor(sensor)
    if sensor_zenith is not None:
        zen = normalize_horizontal(sensor_zenith, layout, "sensor_zenith")
    elif sensor_obj.nadir:
        zen = np.zeros(n)
    elif latitude is not None and longitude is not None:
        lat = normalize_horizontal(latitude, layout, "latitude")
        lon = normalize_horizontal(longitude, layout, "longitude")
        zen = geostationary_zenith(lat, lon, sensor_obj.sublat, sensor_obj.sublon)
    else:
        zen = np.zeros(n)
    if solar_zenith is None:
        solar = np.zeros(n)
    else:
        solar = normalize_horizontal(solar_zenith, layout, "solar_zenith")
    geometry = Geometry(
        sensor_zenith_angle=zen,
        solar_zenith_angle=solar,
        sensor_scan_angle=np.zeros(n),
    )
    model = ForwardModel(sensor, channels, coeff_path=coeff_path, include_clouds=include_clouds)
    return model.run(atmosphere, surface, geometry, include_scattering=include_scattering)
