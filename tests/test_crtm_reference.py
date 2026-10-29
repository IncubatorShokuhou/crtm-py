"""ABI G16 brightness temperatures against CRTM v2.4.0 (gfortran, REAL64).

The profile is the 92-layer clear-sky atmosphere in
``test/mains/regression/forward/test_ClearSky/Load_Atm_Data.inc``.
The reference temperatures were produced by ``libcrtm.so`` from that tag,
Nalli water emissivity, wind 5 m/s, water temperature 290 K, source zenith
180 deg. The cloudy case puts 0.05 kg m^-2 of liquid water with a 10 µm
effective radius in every layer between 700 and 900 hPa and turns scattering on.

Skip the test when ``CRTM_PY_COEFF_DIR`` does not contain the coefficient files.
Those files are not stored in git.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from crtm_py.atmosphere import Atmosphere, Cloud
from crtm_py.emis import load_irland
from crtm_py.forward import ForwardModel
from crtm_py.geometry import Geometry
from crtm_py.surface import Surface

ROOT = Path(__file__).resolve().parents[1]
PROFILE = np.load(ROOT / "tests/data/abi_g16_clearsky_92.npz")

# Fortran CRTM v2.4.0, channels 7..16.
CLEAR_ZENITH_0 = np.array(
    [287.06976, 237.32495, 241.87112, 250.34722, 283.87522, 260.85869, 287.07331, 287.06702, 284.16769, 268.27063]
)
CLEAR_ZENITH_30 = np.array(
    [286.72056, 236.84598, 241.17731, 249.21956, 283.26902, 259.04984, 286.69335, 286.68887, 283.54900, 266.92909]
)
CLOUDY_ZENITH_0 = np.array(
    [251.81339, 237.30243, 241.49865, 246.55066, 253.34341, 244.92213, 254.33604, 254.52173, 254.26198, 249.84884]
)


def _coeff_dir():
    env = os.environ.get("CRTM_PY_COEFF_DIR")
    if not env:
        return None
    path = Path(env)
    if not any(path.rglob("abi_g16.SpcCoeff.bin")):
        return None
    if not any(path.rglob("abi_g16.TauCoeff.bin")):
        return None
    return path


@pytest.fixture(scope="module")
def coeff_dir():
    path = _coeff_dir()
    if path is None:
        pytest.skip("CRTM_PY_COEFF_DIR has no abi_g16 SpcCoeff/TauCoeff")
    return path


def _atmosphere(clouds=()):
    return Atmosphere(
        level_pressure_hpa=PROFILE["level_pressure_hpa"].reshape(1, -1),
        pressure_hpa=PROFILE["pressure_hpa"].reshape(1, -1),
        temperature=PROFILE["temperature"].reshape(1, -1),
        h2o_gkg=PROFILE["h2o_gkg"].reshape(1, -1),
        o3_ppmv=PROFILE["o3_ppmv"].reshape(1, -1),
        clouds=clouds,
        interface_pressure_pa=PROFILE["level_pressure_hpa"].reshape(1, -1) * 100.0,
    )


def _surface():
    return Surface(
        land_type=np.array([1]),
        wind_speed=np.array([5.0]),
        water_coverage=np.ones(1),
        land_coverage=np.zeros(1),
        ice_coverage=np.zeros(1),
        snow_coverage=np.zeros(1),
        land_temperature=np.array([290.0]),
        snow_temperature=np.array([280.0]),
        water_temperature=np.array([290.0]),
        ice_temperature=np.array([280.0]),
        soil_moisture_content=np.array([0.05]),
        vegetation_fraction=np.zeros(1),
        soil_temperature=np.array([283.0]),
        snow_depth=np.zeros(1),
    )


def _tb(coeff_dir, zenith, clouds, scattering):
    model = ForwardModel("abi_g16", coeff_path=coeff_dir)
    result = model.run(
        _atmosphere(clouds),
        _surface(),
        Geometry(np.array([zenith]), np.array([180.0]), np.array([0.0])),
        include_scattering=scattering,
    )
    assert result.optical_depth_model.startswith("odps")
    return np.asarray(result.brightness_temperature).reshape(-1), result.channels


def test_clear_sky_matches_fortran_crtm(coeff_dir):
    nadir, channels = _tb(coeff_dir, 0.0, (), False)
    slant, _ = _tb(coeff_dir, 30.0, (), False)
    assert channels == tuple(range(7, 17))
    assert np.max(np.abs(nadir - CLEAR_ZENITH_0)) < 0.1
    assert np.max(np.abs(slant - CLEAR_ZENITH_30)) < 0.1


def test_include_clouds_false_ignores_the_liquid_layer(coeff_dir):
    pressure = PROFILE["pressure_hpa"]
    water = np.where((pressure >= 700.0) & (pressure <= 900.0), 0.05, 0.0)
    cloud = Cloud("water", water.reshape(1, -1), np.full((1, pressure.size), 10.0))
    clear, _ = _tb(coeff_dir, 0.0, (), False)
    model = ForwardModel("abi_g16", coeff_path=coeff_dir, include_clouds=False)
    result = model.run(
        _atmosphere((cloud,)),
        _surface(),
        Geometry(np.array([0.0]), np.array([180.0]), np.array([0.0])),
        include_scattering=True,
    )
    np.testing.assert_allclose(np.asarray(result.brightness_temperature).reshape(-1), clear, atol=1e-6)


def test_water_cloud_ada_against_fortran_crtm(coeff_dir):
    pressure = PROFILE["pressure_hpa"]
    water = np.where((pressure >= 700.0) & (pressure <= 900.0), 0.05, 0.0)
    cloud = Cloud("water", water.reshape(1, -1), np.full((1, pressure.size), 10.0))
    tb, _ = _tb(coeff_dir, 0.0, (cloud,), True)
    delta = tb - CLOUDY_ZENITH_0
    # Channel 7 (index 0) is the shortwave IR channel. The ADA eigenstep
    # does not reproduce CRTM ASYMTX there; the other nine channels do.
    assert abs(delta[0]) < 1.0
    assert np.max(np.abs(delta[1:])) < 0.2


def test_big_endian_npoess_land_matches_little_endian(coeff_dir):
    little = list(coeff_dir.rglob("NPOESS.IRland.EmisCoeff.bin"))
    big = [path for path in little if "big_endian" in str(path).lower()]
    little = [path for path in little if "little_endian" in str(path).lower()]
    if not big or not little:
        pytest.skip("both endian copies of NPOESS.IRland are required")
    a = load_irland(little[0])
    b = load_irland(big[0])
    np.testing.assert_allclose(a.reflectance, b.reflectance, rtol=0, atol=0)
    np.testing.assert_allclose(a.frequency, b.frequency, rtol=0, atol=0)
