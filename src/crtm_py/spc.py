"""CRTM SpcCoeff and the channel Planck function.

``CRTM_Planck_Radiance`` is

    T_eff = Band_C1 + Band_C2 * T
    R = Planck_C1 / (exp(Planck_C2 / T_eff) - 1)

``Planck_C1`` and ``Planck_C2`` already include the wavenumber factors.
They are not the spectral constants ``c1`` and ``c2``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from crtm_py.crtm_binary import FortranFile


@dataclass
class SpcCoeff:
    sensor_id: str
    sensor_type: int
    wmo_satellite_id: int
    wmo_sensor_id: int
    sensor_channel: np.ndarray
    frequency: np.ndarray
    wavenumber: np.ndarray
    planck_c1: np.ndarray
    planck_c2: np.ndarray
    band_c1: np.ndarray
    band_c2: np.ndarray
    cosmic_background: np.ndarray
    solar_irradiance: np.ndarray

    def channel_index(self, band: int) -> int:
        hits = np.flatnonzero(self.sensor_channel == int(band))
        if hits.size != 1:
            raise KeyError(f"{self.sensor_id} has no channel {band}")
        return int(hits[0])


def load_spccoeff(path) -> SpcCoeff:
    """Read ``*.SpcCoeff.bin``."""
    ff = FortranFile(path)
    ff.read_i32s((2,))
    n_ch = int(ff.read_i32())
    sensor = ff.read_record()
    sensor_id = sensor[:20].decode("ascii").strip()
    sensor_type, wmo_sat, wmo_sensor = np.frombuffer(sensor[20:], dtype=np.dtype(ff.endian + "i4"))
    raw = ff.read_record()
    n4 = n_ch * 4
    ints = np.frombuffer(raw[: 3 * n4], dtype=np.dtype(ff.endian + "i4")).astype(np.int64)
    doubles = np.frombuffer(raw[3 * n4 :], dtype=np.dtype(ff.endian + "f8")).astype(np.float64)
    if doubles.size != 8 * n_ch:
        raise ValueError(f"SpcCoeff channel record has {doubles.size} doubles, expected {8 * n_ch}")
    return SpcCoeff(
        sensor_id=sensor_id,
        sensor_type=int(sensor_type),
        wmo_satellite_id=int(wmo_sat),
        wmo_sensor_id=int(wmo_sensor),
        sensor_channel=ints[:n_ch].copy(),
        frequency=doubles[0 * n_ch : 1 * n_ch].copy(),
        wavenumber=doubles[1 * n_ch : 2 * n_ch].copy(),
        planck_c1=doubles[2 * n_ch : 3 * n_ch].copy(),
        planck_c2=doubles[3 * n_ch : 4 * n_ch].copy(),
        band_c1=doubles[4 * n_ch : 5 * n_ch].copy(),
        band_c2=doubles[5 * n_ch : 6 * n_ch].copy(),
        cosmic_background=doubles[6 * n_ch : 7 * n_ch].copy(),
        solar_irradiance=doubles[7 * n_ch : 8 * n_ch].copy(),
    )


def channel_radiance(spc: SpcCoeff, index: int, temperature):
    """Planck radiance for one SpcCoeff channel."""
    temp = np.asarray(temperature, dtype=np.float64)
    teff = spc.band_c1[index] + spc.band_c2[index] * temp
    return spc.planck_c1[index] / np.expm1(spc.planck_c2[index] / teff)


def channel_temperature(spc: SpcCoeff, index: int, radiance):
    """Brightness temperature for one SpcCoeff channel."""
    rad = np.asarray(radiance, dtype=np.float64)
    teff = spc.planck_c2[index] / np.log(spc.planck_c1[index] / rad + 1.0)
    return (teff - spc.band_c1[index]) / spc.band_c2[index]
