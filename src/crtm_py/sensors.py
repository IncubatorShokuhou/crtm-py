"""ABI infrared channel catalog used by UPP v4.1.

UPP requests ten brightness temperatures per GOES-R series imager, ABI
bands 7 through 16 (``iget`` 927–936 for GOES-16, 937–946 for GOES-17,
958–967 for the nadir ``abi_gr`` fallback). Wavenumbers are
``10000 / lambda`` at the published ABI band-center wavelengths. They
are not the channel-integrated frequencies stored in ``SpcCoeff``.

Subsatellite longitudes are the literals in ``CALRAD_WCLOUD_newcrtm.f``:
GOES-16 at 75.2 W, GOES-17 at 137.2 W. ``abi_gr`` is the nadir path
(sensor zenith forced to 0 when the caller does not pass one).
"""

from __future__ import annotations

from dataclasses import dataclass


def _wn(wavelength_um: float) -> float:
    return 10000.0 / wavelength_um


# GOES-R ABI IR band centers (micrometres).
_ABI_IR_WAVELENGTH_UM = {
    7: 3.90,
    8: 6.19,
    9: 6.95,
    10: 7.34,
    11: 8.50,
    12: 9.61,
    13: 10.35,
    14: 11.20,
    15: 12.30,
    16: 13.30,
}

ABI_IR_CHANNELS: dict[int, float] = {
    band: _wn(wave) for band, wave in _ABI_IR_WAVELENGTH_UM.items()
}


@dataclass(frozen=True)
class Sensor:
    """One CRTM sensor id as UPP initializes it.

    Attributes
    ----------
    sensor_id : str
        CRTM sensor id (``abi_g16``, ``abi_g17``, ``abi_gr``).
    wmo_satellite_id, wmo_sensor_id : int
        Values assigned in ``CALRAD`` after ``CRTM_Init``. The coefficient
        file's own WMO ids are overwritten there for these three sensors.
    sublat, sublon : float
        Geostationary subsatellite point, degrees, longitude positive east.
    nadir : bool
        UPP takes the nadir branch and sets the sensor zenith to 0.
    channels : dict
        ABI band number to central wavenumber (cm^-1).
    """

    sensor_id: str
    wmo_satellite_id: int
    wmo_sensor_id: int
    sublat: float
    sublon: float
    nadir: bool
    channels: dict[int, float]


def _abi(sensor_id: str, sat: int, sublon: float, nadir: bool) -> Sensor:
    return Sensor(
        sensor_id=sensor_id,
        wmo_satellite_id=sat,
        wmo_sensor_id=617,
        sublat=0.0,
        sublon=sublon,
        nadir=nadir,
        channels=dict(ABI_IR_CHANNELS),
    )


SENSORS: dict[str, Sensor] = {
    "abi_g16": _abi("abi_g16", 270, -75.2, nadir=False),
    "abi_g17": _abi("abi_g17", 271, -137.2, nadir=False),
    "abi_gr": _abi("abi_gr", 270, -75.2, nadir=True),
}


def get_sensor(sensor_id: str) -> Sensor:
    """Return the catalog entry for ``sensor_id``.

    Matching is case-insensitive and ignores trailing spaces, which is how
    the fixed-length Fortran sensor list is written.
    """
    key = sensor_id.strip().lower()
    try:
        return SENSORS[key]
    except KeyError as exc:
        known = ", ".join(sorted(SENSORS))
        raise KeyError(f"unknown sensor {sensor_id!r}; known sensors: {known}") from exc


def resolve_channels(sensor: Sensor, channels: tuple[int, ...] | None) -> tuple[int, ...]:
    """Validate ABI band numbers. ``None`` selects bands 7–16 in order."""
    if channels is None:
        return tuple(sensor.channels)
    out = tuple(int(c) for c in channels)
    if not out:
        raise ValueError("channels is empty")
    missing = [c for c in out if c not in sensor.channels]
    if missing:
        raise ValueError(
            f"sensor {sensor.sensor_id} has no ABI band(s) {missing}; "
            f"available bands are {tuple(sensor.channels)}"
        )
    return out
