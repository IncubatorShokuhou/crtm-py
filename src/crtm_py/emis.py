"""Infrared water emissivity and NPOESS land reflectance.

Water follows ``CRTM_Compute_IRSSEM``: trilinear interpolation in zenith
angle (degrees), wavenumber and wind speed. ``CRTM_Init`` loads
``Nalli.IRwater.EmisCoeff.bin``; ``WuSmith.IRwater.EmisCoeff.bin`` is the
alternate table and uses the same record layout. Land follows
``SEcategory_Emissivity``: linear interpolation in wavenumber of
``1 - reflectance`` for the CRTM land-type index
(``NPOESS.IRland.EmisCoeff.bin``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from crtm_py.crtm_binary import FortranFile


def _weights(grid: np.ndarray, value: float):
    x = float(value)
    if x <= grid[0]:
        return 0, 0, 1.0, 0.0
    if x >= grid[-1]:
        return grid.size - 1, grid.size - 1, 1.0, 0.0
    i2 = int(np.searchsorted(grid, x, side="right"))
    i1 = i2 - 1
    span = grid[i2] - grid[i1]
    w2 = (x - grid[i1]) / span if span else 0.0
    return i1, i2, 1.0 - w2, w2


@dataclass
class IRWaterCoeff:
    angle: np.ndarray
    frequency: np.ndarray
    wind_speed: np.ndarray
    emissivity: np.ndarray  # (n_angle, n_frequency, n_wind)


@dataclass
class IRLandCoeff:
    frequency: np.ndarray
    reflectance: np.ndarray  # (n_frequency, n_types)
    surface_type_name: list[str]


def load_irwater(path) -> IRWaterCoeff:
    ff = FortranFile(path)
    rest = []
    while True:
        try:
            rest.append(ff.read_record())
        except EOFError:
            break
    # Dimension record is three int32: n_angle, n_frequency, n_wind.
    dims = None
    dim_at = None
    for i, rec in enumerate(rest):
        if len(rec) == 12:
            vals = np.frombuffer(rec, dtype=np.dtype(ff.endian + "i4"))
            if vals[0] > 1 and vals[1] > 1 and vals[2] > 1 and np.prod(vals) > 1000:
                dims = [int(v) for v in vals]
                dim_at = i
                break
    if dims is None:
        raise ValueError(f"no IR water dimension record in {path}")
    n_ang, n_freq, n_wind = dims
    coord = None
    cube = None
    need = (n_ang + n_freq + n_wind) * 8
    cube_n = n_ang * n_freq * n_wind * 8
    for rec in rest[dim_at + 1 :]:
        if coord is None and len(rec) == need:
            coord = np.frombuffer(rec, dtype=np.dtype(ff.endian + "f8")).astype(np.float64)
        elif cube is None and len(rec) == cube_n:
            cube = np.frombuffer(rec, dtype=np.dtype(ff.endian + "f8")).astype(np.float64)
    if coord is None or cube is None:
        raise ValueError(f"incomplete IR water coefficient file {path}")
    angle = coord[:n_ang].copy()
    frequency = coord[n_ang : n_ang + n_freq].copy()
    wind = coord[n_ang + n_freq :].copy()
    emis = cube.reshape((n_ang, n_freq, n_wind), order="F")
    return IRWaterCoeff(angle=angle, frequency=frequency, wind_speed=wind, emissivity=emis)


def load_irland(path) -> IRLandCoeff:
    ff = FortranFile(path)
    rest = []
    while True:
        try:
            rest.append(ff.read_record())
        except EOFError:
            break
    # After the datatype name, release/version, then n_frequencies, n_types.
    dims = None
    for rec in rest:
        if len(rec) == 8:
            vals = np.frombuffer(rec, dtype=np.dtype(ff.endian + "i4"))
            if int(vals[0]) > 1 and int(vals[0]) == int(vals[1]) and int(vals[0]) <= 64:
                dims = [int(v) for v in vals]
    if dims is None:
        raise ValueError(f"no NPOESS land dimensions in {path}")
    n_freq, n_type = dims
    frequency = None
    reflectance = None
    for rec in rest:
        if len(rec) == n_freq * 8 and frequency is None:
            frequency = np.frombuffer(rec, dtype=np.dtype(ff.endian + "f8")).astype(np.float64).copy()
        elif len(rec) == n_freq * n_type * 8:
            reflectance = np.frombuffer(rec, dtype=np.dtype(ff.endian + "f8")).astype(np.float64)
            reflectance = reflectance.reshape((n_freq, n_type), order="F")
    if frequency is None or reflectance is None:
        raise ValueError(f"incomplete NPOESS land file {path}")
    return IRLandCoeff(frequency=frequency, reflectance=reflectance, surface_type_name=[])


def water_emissivity(coeff: IRWaterCoeff, wavenumber: float, zenith_deg: float, wind_speed: float) -> float:
    i1, i2, wi1, wi2 = _weights(coeff.angle, abs(zenith_deg))
    j1, j2, wj1, wj2 = _weights(coeff.frequency, wavenumber)
    k1, k2, wk1, wk2 = _weights(coeff.wind_speed, wind_speed)
    table = coeff.emissivity
    value = 0.0
    for ia, wa in ((i1, wi1), (i2, wi2)):
        for jf, wf in ((j1, wj1), (j2, wj2)):
            for kv, wv in ((k1, wk1), (k2, wk2)):
                value += wa * wf * wv * table[ia, jf, kv]
    return float(np.clip(value, 0.0, 1.0))


def land_emissivity(coeff: IRLandCoeff, wavenumber: float, land_type: int) -> float:
    n_type = coeff.reflectance.shape[1]
    itype = int(np.clip(land_type, 1, n_type)) - 1
    i1, i2, w1, w2 = _weights(coeff.frequency, wavenumber)
    reflectance = w1 * coeff.reflectance[i1, itype] + w2 * coeff.reflectance[i2, itype]
    return float(np.clip(1.0 - reflectance, 0.0, 1.0))
