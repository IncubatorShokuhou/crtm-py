import math

import numpy as np
import pytest

from crtm_py.clouds import gammln, wgamma
from crtm_py.geometry import geostationary_zenith
from crtm_py.planck import planck_radiance, planck_temperature


def test_planck_round_trip():
    freq = np.array([900.0, 1400.0, 2500.0])
    temp = np.array([220.0, 280.0, 310.0])
    radiance = planck_radiance(freq, temp)
    recovered = planck_temperature(freq, radiance)
    np.testing.assert_allclose(recovered, temp, rtol=0, atol=1e-8)


def test_planck_polychromatic_coefficients():
    freq = 900.0
    # T_channel = 2 + 0.9 * T_mono, so a reported 280 K was a mono temperature of (280-2)/0.9.
    coeffs = np.array([2.0, 0.9])
    radiance = planck_radiance(freq, 280.0, coeffs)
    mono = (280.0 - 2.0) / 0.9
    np.testing.assert_allclose(radiance, planck_radiance(freq, mono), rtol=1e-12)
    np.testing.assert_allclose(planck_temperature(freq, radiance, coeffs), 280.0, atol=1e-8)


def test_planck_rejects_non_positive():
    assert np.isnan(planck_radiance(900.0, -1.0))
    assert np.isnan(planck_temperature(900.0, 0.0))


def test_gammln_matches_factorial():
    # Gamma(6) = 5! = 120.
    assert gammln(6.0) == pytest.approx(math.log(120.0), abs=1e-10)
    assert wgamma(6.0) == pytest.approx(120.0, rel=1e-10)


def test_subsatellite_zenith_is_zero():
    za = geostationary_zenith(0.0, -75.2, 0.0, -75.2)
    assert za == pytest.approx(0.0, abs=1e-6)
    wrapped = geostationary_zenith(0.0, 284.8, 0.0, -75.2)
    assert wrapped == pytest.approx(0.0, abs=1e-6)


def test_zenith_grows_away_from_the_subpoint():
    lats = [0.0, 20.0, 45.0, 70.0]
    angles = [float(geostationary_zenith(lat, -75.2, 0.0, -75.2)) for lat in lats]
    assert angles == sorted(angles)
    assert angles[0] < 1e-6
    assert angles[-1] > angles[-2]
