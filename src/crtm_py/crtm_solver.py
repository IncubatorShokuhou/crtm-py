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

from crtm_py.ada import ada_radiance
from crtm_py.cloudlut import layer_cloud_optics, load_cloudcoeff
from crtm_py.constants import EMIS_ICE, EMIS_SNOW
from crtm_py.emis import land_emissivity, load_irland, load_irwater, water_emissivity
from crtm_py.odps import gaseous_nadir_od, load_odps_file
from crtm_py.spc import channel_radiance, channel_temperature, load_spccoeff
from crtm_py.toa import prepend_standard_atmosphere

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


def _coverage_emissivity(coeff: CRTMCoeff, band_index: int, zenith: float, surface) -> np.ndarray:
    """One emissivity per profile, coverage-weighted."""
    n = surface.n_profiles
    wn = float(coeff.spc.wavenumber[band_index])
    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        weights = np.array(
            [
                max(surface.water_coverage[i], 0.0),
                max(surface.land_coverage[i], 0.0),
                max(surface.ice_coverage[i], 0.0),
                max(surface.snow_coverage[i], 0.0),
            ]
        )
        if weights.sum() <= 0.0:
            weights[1] = 1.0
        weights /= weights.sum()
        parts = np.zeros(4)
        if coeff.water is not None:
            parts[0] = water_emissivity(coeff.water, wn, zenith, float(surface.wind_speed[i]))
        else:
            parts[0] = 0.98
        if coeff.land is not None:
            parts[1] = land_emissivity(coeff.land, wn, int(surface.land_type[i]))
        else:
            parts[1] = 0.97
        parts[2] = EMIS_ICE
        parts[3] = EMIS_SNOW
        if surface.emissivity is not None:
            user = np.asarray(surface.emissivity, dtype=np.float64)
            parts[:] = float(np.reshape(user, (-1,))[i] if user.size == n else user)
        out[i] = float(np.dot(weights, parts))
    return out


def brightness_crtm(coeff: CRTMCoeff, atmosphere, surface, geometry, bands, *, scattering: bool):
    """Brightness temperature, shape ``(n_profiles, n_channels)``."""
    n = atmosphere.n_profiles
    tb = np.empty((n, len(bands)), dtype=np.float64)
    rad = np.empty_like(tb)
    for i in range(n):
        zen = float(geometry.sensor_zenith_angle[i])
        mu = np.cos(np.deg2rad(zen))
        level_p, pressure, temperature, h2o, o3, n_added = prepend_standard_atmosphere(
            atmosphere.level_pressure_hpa[i],
            atmosphere.pressure_hpa[i],
            atmosphere.temperature[i],
            atmosphere.h2o_gkg[i],
            atmosphere.o3_ppmv[i],
        )
        gas = gaseous_nadir_od(
            coeff.odps,
            level_p,
            pressure,
            temperature,
            h2o,
            o3,
            zen,
            bands,
        )
        for ic, band in enumerate(bands):
            idx = coeff.spc.channel_index(int(band))
            wn = float(coeff.spc.wavenumber[idx])
            tau = gas[:, ic].copy()
            ssa = np.zeros(tau.shape)
            phase = None
            if scattering and coeff.cloud is not None and atmosphere.clouds:
                cloud_tau, cloud_ssa, cloud_phase = _clouds_one_profile(
                    coeff.cloud, wn, atmosphere, i, scattering=True, n_added=n_added
                )
                ssa = np.divide(
                    cloud_ssa * cloud_tau, tau + cloud_tau, out=np.zeros_like(tau), where=(tau + cloud_tau) > 0
                )
                tau = tau + cloud_tau
                phase = cloud_phase
            elif coeff.cloud is not None and atmosphere.clouds:
                cloud_tau, _, _ = _clouds_one_profile(
                    coeff.cloud, wn, atmosphere, i, scattering=False, n_added=n_added
                )
                tau = tau + cloud_tau
            planck_layer = channel_radiance(coeff.spc, idx, temperature)
            # Coverage-weighted skin radiance uses each surface temperature.
            emis = float(_coverage_emissivity(coeff, idx, zen, surface)[i])
            skin = _skin_radiance(coeff, idx, surface, i, emis)
            cosmic = float(coeff.spc.cosmic_background[idx])
            streams = 0
            if phase is not None:
                # phase stores 0..n_legendre, and n_legendre is the Mie stream count.
                streams = phase.shape[0] - 1
            radiance = ada_radiance(
                tau,
                ssa,
                phase,
                planck_layer,
                skin,
                emis,
                cosmic,
                mu,
                streams=streams,
            )
            rad[i, ic] = radiance
            tb[i, ic] = float(channel_temperature(coeff.spc, idx, np.array([radiance]))[0])
    return tb, rad


def _clouds_one_profile(cloud_coeff, wavenumber, atmosphere, index, *, scattering: bool, n_added: int):
    clouds = []
    n_layers = atmosphere.n_layers + n_added
    pad = np.zeros(n_added, dtype=np.float64)
    for cloud in atmosphere.clouds:
        clouds.append(
            type("C", (), {
                "cloud_type": cloud.cloud_type,
                "water_content": np.concatenate([pad, cloud.water_content[index]]),
                "effective_radius": np.concatenate([pad, cloud.effective_radius[index]]),
            })()
        )
    return layer_cloud_optics(
        cloud_coeff, wavenumber, clouds, n_layers, scattering=scattering
    )


def _skin_radiance(coeff, band_index, surface, index, emissivity):
    """Planck radiance of the coverage-weighted skin temperature.

    The emissivity is applied in the RT, so this returns ``B(T)`` of the
    warmest contributing surface, weighted by coverage. Mixed pixels use
    the coverage-weighted temperature. That is linear in T, which is what
    a single-stream call can accept; fractional surfaces in CRTM are
    separate RT runs. A pixel that is entirely one type matches CRTM.
    """
    weights = np.array(
        [
            max(surface.water_coverage[index], 0.0),
            max(surface.land_coverage[index], 0.0),
            max(surface.ice_coverage[index], 0.0),
            max(surface.snow_coverage[index], 0.0),
        ]
    )
    temps = np.array(
        [
            surface.water_temperature[index],
            surface.land_temperature[index],
            surface.ice_temperature[index],
            surface.snow_temperature[index],
        ]
    )
    if weights.sum() <= 0.0:
        weights[1] = 1.0
    weights /= weights.sum()
    skin = float(np.dot(weights, temps))
    return float(channel_radiance(coeff.spc, band_index, np.array(skin)))
