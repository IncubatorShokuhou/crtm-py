"""Geostationary zenith angle from UPP ``GEO_ZENITH_ANGLE.f``.

The routine is the Kleespies spherical-geometry calculation (1987/1989)
as converted by Chuang. Constants are the Fortran DATA values, including
``C2 = 538527888``, which does not match the comment's ``2*RE*RES``.
Longitude handling is the same branch: longitudes above 180 are shifted
by 360 before the absolute difference. The Fortran header notes trouble
near the date line, the prime meridian, and the poles.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from crtm_py.constants import MAX_SENSOR_SCAN_ANGLE

_DTR = 0.017453293
_RE = 6370.0
_PI = 3.1415926
_RTD = 57.29577951
_RES = 40589641.0
_RZS = 1786245696.0
_C1 = 1826835337.0
_C2 = 538527888.0


@dataclass
class Geometry:
    """CRTM geometry for one batch of profiles.

    Angles are degrees. ``sensor_scan_angle`` is stored because UPP fills
    it, then overwrites it with 0 before ``CRTM_Forward``. The emission
    solver uses ``sensor_zenith_angle`` only. ``solar_zenith_angle`` is
    retained for a later channel-7 solar term and is not used yet.

    Attributes
    ----------
    sensor_zenith_angle, solar_zenith_angle, sensor_scan_angle : ndarray
        Shape ``(n_profiles,)``.
    """

    sensor_zenith_angle: np.ndarray
    solar_zenith_angle: np.ndarray
    sensor_scan_angle: np.ndarray

    def __post_init__(self) -> None:
        self.sensor_zenith_angle = np.asarray(self.sensor_zenith_angle, dtype=np.float64)
        self.solar_zenith_angle = np.asarray(self.solar_zenith_angle, dtype=np.float64)
        self.sensor_scan_angle = np.asarray(self.sensor_scan_angle, dtype=np.float64)

    @property
    def valid_zenith(self) -> np.ndarray:
        """Profiles UPP would pass into ``CRTM_Forward``.

        The test is ``0 <= sensor_zenith <= MAX_SENSOR_SCAN_ANGLE`` (80°),
        which is the comparison in ``CALRAD`` even though the CRTM symbol
        is named for the scan angle.
        """
        z = self.sensor_zenith_angle
        return (z >= 0.0) & (z <= MAX_SENSOR_SCAN_ANGLE) & np.isfinite(z)


def geostationary_zenith(latitude, longitude, sublat, sublon):
    """Local zenith angle (degrees) of a geostationary satellite.

    Parameters
    ----------
    latitude, longitude : array_like
        Earth location in degrees. Longitude is positive east. Values
        above 180 are treated as degrees east of the prime meridian and
        wrapped with the Fortran branch ``RLON - 360``.
    sublat, sublon : float
        Subsatellite latitude and longitude in degrees.

    Returns
    -------
    ndarray
        Zenith angle in degrees, clipped at 0. A result above 90 means
        the location is past the limb in this spherical geometry.
    """
    lat = np.asarray(latitude, dtype=np.float64)
    lon = np.asarray(longitude, dtype=np.float64)
    lat, lon = np.broadcast_arrays(lat, lon)
    a = (90.0 - lat) * _DTR
    b = (90.0 - float(sublat)) * _DTR
    c = np.where(lon > 180.0, np.abs(lon - 360.0 - sublon), np.abs(lon - sublon)) * _DTR
    cosd = np.cos(a) * np.cos(b) + np.sin(a) * np.sin(b) * np.cos(c)
    pp = _C1 - _C2 * cosd
    # A location past the limb can make pp negative in floating point.
    pp = np.maximum(pp, 0.0)
    p = np.sqrt(pp)
    cose = (pp + _RES - _RZS) / (2.0 * _RE * p)
    cose = np.clip(cose, -1.0, 1.0)
    e = np.arccos(cose)
    za = (_PI - e) * _RTD
    return np.maximum(za, 0.0)


def solar_zenith_from_cosine(cosine):
    """Solar zenith angle in degrees from UPP ``czen`` (``acos``)."""
    c = np.clip(np.asarray(cosine, dtype=np.float64), -1.0, 1.0)
    return np.degrees(np.arccos(c))
