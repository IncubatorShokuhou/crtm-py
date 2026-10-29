"""ABI G16 brightness temperatures against CRTM v2.4.0 (gfortran, REAL64).

The profile is the 92-layer clear-sky atmosphere in
``test/mains/regression/forward/test_ClearSky/Load_Atm_Data.inc``.
The reference temperatures were produced by ``libcrtm.so`` from that tag,
Nalli water emissivity, wind 5 m/s, water temperature 290 K, source zenith
180 deg. The liquid-cloud case puts 0.05 kg m^-2 of water with a 10 µm
effective radius in every layer between 700 and 900 hPa and turns scattering
on. The extra cases are clear land (NPOESS type 1, 290 K), the same liquid
cloud at 60 deg zenith, and that liquid cloud plus 0.02 kg m^-2 of ice
(20 µm) between 300 and 500 hPa.

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
    [287.06975635, 237.32494695, 241.87112134, 250.34721665, 283.87522321, 260.85868546, 287.07330590, 287.06701774, 284.16769296, 268.27063461]
)
CLEAR_ZENITH_30 = np.array(
    [286.72056234, 236.84597641, 241.17731389, 249.21956220, 283.26902266, 259.04984180, 286.69335275, 286.68886743, 283.54899849, 266.92909081]
)
CLEAR_ZENITH_60 = np.array(
    [284.40643907, 235.32903068, 238.83587086, 245.34332031, 279.73278222, 252.16232419, 283.97950231, 283.99996313, 279.54662119, 260.81019756]
)
CLOUDY_ZENITH_0 = np.array(
    [251.81339, 237.30243, 241.49865, 246.55066, 253.34341, 244.92213, 254.33604, 254.52173, 254.26198, 249.84884]
)
LAND_CLEAR = np.array(
    [284.61940120, 237.32494695, 241.87111532, 250.34213885, 283.01699213, 260.08914847, 285.26307077, 285.43228237, 283.59072654, 268.24551100]
)
CLOUDY_ZENITH_60 = np.array(
    [249.50681129, 235.32846233, 238.78230656, 243.76063638, 251.13283743, 242.35516320, 253.07804732, 253.76082542, 253.49194623, 247.62266598]
)
ICE_AND_WATER = np.array(
    [226.43230312, 228.38106583, 226.88416340, 226.77563880, 223.34768966, 234.12604758, 224.63820523, 224.28135026, 223.74966846, 226.52586201]
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
    steep, _ = _tb(coeff_dir, 60.0, (), False)
    assert channels == tuple(range(7, 17))
    assert np.max(np.abs(nadir - CLEAR_ZENITH_0)) < 0.001
    assert np.max(np.abs(slant - CLEAR_ZENITH_30)) < 0.001
    assert np.max(np.abs(steep - CLEAR_ZENITH_60)) < 0.001


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


def _liquid():
    pressure = PROFILE["pressure_hpa"]
    water = np.where((pressure >= 700.0) & (pressure <= 900.0), 0.05, 0.0)
    return (Cloud("water", water.reshape(1, -1), np.full((1, pressure.size), 10.0)),)


def _land():
    surface = _surface()
    surface.water_coverage[:] = 0.0
    surface.land_coverage[:] = 1.0
    surface.land_type[:] = 1
    surface.land_temperature[:] = 290.0
    return surface


def test_water_cloud_ada_against_fortran_crtm(coeff_dir):
    tb, _ = _tb(coeff_dir, 0.0, _liquid(), True)
    assert np.max(np.abs(tb - CLOUDY_ZENITH_0)) < 0.001


def test_land_clear_sky_matches_fortran_crtm(coeff_dir):
    model = ForwardModel("abi_g16", coeff_path=coeff_dir)
    result = model.run(
        _atmosphere(),
        _land(),
        Geometry(np.array([0.0]), np.array([180.0]), np.array([0.0])),
        include_scattering=False,
    )
    tb = np.asarray(result.brightness_temperature).reshape(-1)
    assert np.max(np.abs(tb - LAND_CLEAR)) < 0.001


def test_water_cloud_at_60_degrees_matches_fortran_crtm(coeff_dir):
    tb, _ = _tb(coeff_dir, 60.0, _liquid(), True)
    assert np.max(np.abs(tb - CLOUDY_ZENITH_60)) < 0.001


def test_ice_and_water_clouds_match_fortran_crtm(coeff_dir):
    pressure = PROFILE["pressure_hpa"]
    ice = np.where((pressure >= 300.0) & (pressure <= 500.0), 0.02, 0.0)
    clouds = _liquid() + (Cloud("ice", ice.reshape(1, -1), np.full((1, pressure.size), 20.0)),)
    tb, _ = _tb(coeff_dir, 0.0, clouds, True)
    assert np.max(np.abs(tb - ICE_AND_WATER)) < 0.001


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
