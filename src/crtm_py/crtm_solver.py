"""Run the CRTM emission or ADA solution for one batch of profiles.

Gas optical depth is ODPS. Cloud optical properties come from CloudCoeff.
Sea emissivity is Nalli (Wu-Smith if that file is the one present). Land
emissivity is the NPOESS category table. Ice and snow still use the
constant infrared values.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from crtm_py.batch_forward import _Cloud, forward_chunk
from crtm_py.cloudlut import load_cloudcoeff
from crtm_py.emis import load_irland, load_irwater
from crtm_py.odps import load_odps_file
from crtm_py.spc import load_spccoeff

_CACHE: dict[tuple[str, str], "CRTMCoeff"] = {}


@dataclass
class CRTMCoeff:
    sensor_id: str
    spc: object
    odps: object
    cloud: object | None
    water: object | None
    land: object | None
    root: Path


def _find(root: Path, name: str) -> Path | None:
    direct = list(root.rglob(name))
    if not direct:
        return None
    # Prefer little-endian copies. A big-endian file is still readable.
    direct.sort(key=lambda path: ("little_endian" not in str(path).lower(), "big_endian" in str(path).lower()))
    return direct[0]


def load_sensor(sensor_id: str, root) -> CRTMCoeff | None:
    """Load SpcCoeff and ODPS TauCoeff for ``sensor_id``. ``None`` if absent."""
    root = Path(root)
    key = (sensor_id, str(root.resolve()) if root.exists() else str(root))
    if key in _CACHE:
        return _CACHE[key]
    spc_path = _find(root, f"{sensor_id}.SpcCoeff.bin")
    tau_path = _find(root, f"{sensor_id}.TauCoeff.bin")
    if spc_path is None or tau_path is None:
        return None
    # Visible-channel duplicates are named ``v.abi_g16``. Skip those.
    if spc_path.name.startswith("v."):
        matches = [p for p in root.rglob(f"{sensor_id}.SpcCoeff.bin") if not p.name.startswith("v.")]
        if not matches:
            return None
        spc_path = matches[0]
    tau_matches = [p for p in root.rglob(f"{sensor_id}.TauCoeff.bin") if "ODPS" in str(p) or "odps" in str(p)]
    if tau_matches:
        tau_path = tau_matches[0]
    cloud_path = _find(root, "CloudCoeff.bin")
    # CRTM_Init defaults to Nalli; Wu-Smith is the alternate IRSSEM table.
    water_path = _find(root, "Nalli.IRwater.EmisCoeff.bin")
    if water_path is None:
        water_path = _find(root, "WuSmith.IRwater.EmisCoeff.bin")
    land_path = _find(root, "NPOESS.IRland.EmisCoeff.bin")
    loaded = CRTMCoeff(
        sensor_id=sensor_id,
        spc=load_spccoeff(spc_path),
        odps=load_odps_file(tau_path),
        cloud=load_cloudcoeff(cloud_path) if cloud_path else None,
        water=load_irwater(water_path) if water_path else None,
        land=load_irland(land_path) if land_path else None,
        root=root,
    )
    _CACHE[key] = loaded
    return loaded


def brightness_crtm(
    coeff: CRTMCoeff,
    atmosphere,
    surface,
    geometry,
    bands,
    *,
    scattering: bool,
    chunk_size: int = 256,
    n_jobs: int = 1,
):
    """Brightness temperature, shape ``(n_profiles, n_channels)``.

    Columns are integrated in chunks of ``chunk_size``. ``n_jobs`` above 1
    farms those chunks out with ``concurrent.futures``. The default is one
    process; the speedup is the vectorized chunk, not the process pool.
    Scattering eigenvalues are computed only for layers with cloud.
    """
    n = atmosphere.n_profiles
    chunk_size = max(int(chunk_size), 1)
    jobs = n_jobs if n_jobs and n_jobs > 0 else 1
    slices = [(start, min(start + chunk_size, n)) for start in range(0, n, chunk_size)]
    payloads = [_payload(atmosphere, surface, geometry, bands, scattering, sl) for sl in slices]
    if jobs == 1 or len(payloads) == 1:
        parts = [_run_payload(coeff, item) for item in payloads]
    else:
        from concurrent.futures import ProcessPoolExecutor

        parts = [None] * len(payloads)
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = [pool.submit(_run_payload, coeff, item) for item in payloads]
            for i, future in enumerate(futures):
                parts[i] = future.result()
    tb = np.concatenate([part[0] for part in parts], axis=0)
    rad = np.concatenate([part[1] for part in parts], axis=0)
    return tb, rad


def _payload(atmosphere, surface, geometry, bands, scattering, bounds):
    start, stop = bounds
    clouds = [
        (cloud.cloud_type, cloud.water_content[start:stop], cloud.effective_radius[start:stop])
        for cloud in atmosphere.clouds
    ]
    user_emis = None if surface.emissivity is None else np.asarray(surface.emissivity, dtype=np.float64)
    if user_emis is not None and user_emis.shape[0] == atmosphere.n_profiles:
        user_emis = user_emis[start:stop]
    return {
        "level": atmosphere.level_pressure_hpa[start:stop],
        "pressure": atmosphere.pressure_hpa[start:stop],
        "temperature": atmosphere.temperature[start:stop],
        "h2o": atmosphere.h2o_gkg[start:stop],
        "o3": atmosphere.o3_ppmv[start:stop],
        "clouds": clouds,
        "zenith": geometry.sensor_zenith_angle[start:stop],
        "surface": {
            "wind": surface.wind_speed[start:stop],
            "land_type": surface.land_type[start:stop],
            "water": surface.water_coverage[start:stop],
            "land": surface.land_coverage[start:stop],
            "ice": surface.ice_coverage[start:stop],
            "snow": surface.snow_coverage[start:stop],
            "water_t": surface.water_temperature[start:stop],
            "land_t": surface.land_temperature[start:stop],
            "ice_t": surface.ice_temperature[start:stop],
            "snow_t": surface.snow_temperature[start:stop],
            "emissivity": user_emis,
        },
        "bands": tuple(bands),
        "scattering": scattering,
    }


def _run_payload(coeff, item):
    clouds = [_Cloud(kind, water, radius) for kind, water, radius in item["clouds"]]
    return forward_chunk(
        coeff,
        item["level"],
        item["pressure"],
        item["temperature"],
        item["h2o"],
        item["o3"],
        clouds,
        item["zenith"],
        item["surface"],
        item["bands"],
        item["scattering"],
    )

