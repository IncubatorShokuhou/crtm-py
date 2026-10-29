import numpy as np
import pytest

from crtm_py import brightness_temperature
from crtm_py.clouds import Hydrometeors
from crtm_py.emission import emission_radiance
from crtm_py.forward import reset_synthetic_warning
from crtm_py.planck import planck_radiance, planck_temperature


def _emission(temperature, tau, tsfc, wn, emissivity=1.0, zenith=0.0):
    temperature = np.atleast_2d(np.asarray(temperature, dtype=np.float64))
    tau = np.asarray(tau, dtype=np.float64)
    if tau.ndim == 1:
        tau = tau.reshape(1, -1, 1)
    n = temperature.shape[0]
    bb = planck_radiance(wn, tsfc)
    emission = np.full((n, 1), emissivity * float(np.reshape(bb, ())))
    reflectance = np.full((n, 1), 1.0 - emissivity)
    return emission_radiance(
        temperature,
        tau,
        np.full(n, zenith),
        emission,
        reflectance,
        np.array([wn]),
    )


def test_transparent_column_returns_the_skin_temperature():
    wn = 900.0
    radiance = _emission([200.0, 250.0], [0.0, 0.0], 300.0, wn)
    tb = planck_temperature(wn, radiance)
    assert tb.item() == pytest.approx(300.0, abs=1e-6)


def test_opaque_layer_hides_the_surface():
    wn = 900.0
    radiance = _emission([250.0], [80.0], 300.0, wn)
    tb = planck_temperature(wn, radiance)
    assert tb.item() == pytest.approx(250.0, abs=1e-4)


def test_opaque_upper_layer_sets_the_brightness_temperature():
    wn = 900.0
    radiance = _emission([220.0, 290.0], [40.0, 0.01], 300.0, wn)
    tb = planck_temperature(wn, radiance)
    assert tb.item() == pytest.approx(220.0, abs=1e-3)


def _column():
    p_int = np.array([1000e2, 700e2, 400e2, 200e2, 50e2])
    p_mid = np.array([850e2, 550e2, 300e2, 120e2])
    t = np.array([290.0, 270.0, 250.0, 220.0])
    q = np.array([8e-3, 2e-3, 2e-4, 1e-5])
    return p_int, p_mid, t, q


def test_water_vapor_channel_cools_when_the_column_is_moist():
    p_int, p_mid, t, q = _column()
    dry = brightness_temperature(
        p_int, p_mid, t, q * 0.0, sensor="abi_g16", channels=(8, 14), sensor_zenith=0.0, skin_temperature=292.0
    )
    moist = brightness_temperature(
        p_int, p_mid, t, q, sensor="abi_g16", channels=(8, 14), sensor_zenith=0.0, skin_temperature=292.0
    )
    assert dry.channels == (8, 14)
    assert moist.brightness_temperature.shape == (2,)
    # Band 8 is the strong water-vapor channel in the synthetic table.
    assert moist.brightness_temperature[0] < dry.brightness_temperature[0] - 15.0
    assert moist.brightness_temperature[0] < moist.brightness_temperature[1] - 15.0
    assert "synthetic_grey_gas" in moist.optical_depth_model
    assert any("unipost" in note for note in moist.notes)


def test_ozone_cools_the_9_6_micron_channel():
    p_int, p_mid, t, q = _column()
    o3 = np.zeros_like(q)
    o3[-1] = 8.0 / 604229.0  # top layer in bottom-up order, about 8 ppmv
    clear = brightness_temperature(
        p_int, p_mid, t, q, sensor="abi_g16", channels=(12, 14), sensor_zenith=0.0, skin_temperature=292.0
    )
    with_o3 = brightness_temperature(
        p_int,
        p_mid,
        t,
        q,
        ozone=o3,
        sensor="abi_g16",
        channels=(12, 14),
        sensor_zenith=0.0,
        skin_temperature=292.0,
    )
    assert with_o3.brightness_temperature[0] < clear.brightness_temperature[0] - 5.0
    assert abs(with_o3.brightness_temperature[1] - clear.brightness_temperature[1]) < 1.0


def test_thick_cloud_pulls_the_window_toward_the_cloud_temperature():
    p_int, p_mid, t, q = _column()
    clear = brightness_temperature(
        p_int, p_mid, t, q, sensor="abi_g16", channels=(14,), sensor_zenith=0.0, skin_temperature=292.0, imp_physics=0
    )
    hydro = Hydrometeors(qqw=np.array([0.0, 2.0e-3, 0.0, 0.0]))
    cloudy = brightness_temperature(
        p_int,
        p_mid,
        t,
        q,
        hydrometeors=hydro,
        imp_physics=8,
        sensor="abi_g16",
        channels=(14,),
        sensor_zenith=0.0,
        skin_temperature=292.0,
    )
    assert clear.brightness_temperature[0] > 280.0
    assert cloudy.brightness_temperature[0] < clear.brightness_temperature[0] - 5.0
    assert cloudy.brightness_temperature[0] < 275.0


def test_scattering_flag_without_coefficients_stays_synthetic():
    p_int, p_mid, t, q = _column()
    result = brightness_temperature(
        p_int,
        p_mid,
        t,
        q,
        sensor="abi_g16",
        channels=(14,),
        sensor_zenith=0.0,
        skin_temperature=292.0,
        include_scattering=True,
    )
    assert "synthetic_grey_gas" in result.optical_depth_model
    assert any("include_scattering" in note for note in result.notes)


def test_zenith_above_the_upp_limit_is_missing():
    p_int, p_mid, t, q = _column()
    result = brightness_temperature(
        p_int, p_mid, t, q, channels=(14,), sensor_zenith=85.0, skin_temperature=292.0
    )
    assert np.isnan(result.brightness_temperature[0])
    assert result.valid[0] == False  # noqa: E712 — ndarray element


def test_grid_layout_and_abi_gr_nadir():
    p_int, p_mid, t, q = _column()
    grid_t = t.reshape(-1, 1, 1)
    grid_p = p_mid.reshape(-1, 1, 1)
    grid_q = q.reshape(-1, 1, 1)
    grid_int = p_int.reshape(-1, 1, 1)
    nadir = brightness_temperature(
        grid_int,
        grid_p,
        grid_t,
        grid_q,
        sensor="abi_gr",
        channels=(14,),
        latitude=np.array([[40.0]]),
        longitude=np.array([[-105.0]]),
        skin_temperature=292.0,
    )
    explicit = brightness_temperature(
        p_int, p_mid, t, q, sensor="abi_g16", channels=(14,), sensor_zenith=0.0, skin_temperature=292.0
    )
    assert nadir.brightness_temperature.shape == (1, 1, 1)
    assert nadir.brightness_temperature[0, 0, 0] == pytest.approx(explicit.brightness_temperature[0])


def test_slant_path_changes_a_partly_transmitting_channel():
    p_int, p_mid, t, q = _column()
    nadir = brightness_temperature(
        p_int, p_mid, t, q, channels=(16,), sensor_zenith=0.0, skin_temperature=292.0
    )
    slant = brightness_temperature(
        p_int, p_mid, t, q, channels=(16,), sensor_zenith=60.0, skin_temperature=292.0
    )
    assert slant.brightness_temperature[0] != pytest.approx(nadir.brightness_temperature[0], abs=0.2)


def test_unknown_channel_is_rejected():
    p_int, p_mid, t, q = _column()
    with pytest.raises(ValueError, match="ABI"):
        brightness_temperature(p_int, p_mid, t, q, channels=(1, 14))


def test_synthetic_warning_is_issued():
    import warnings

    reset_synthetic_warning()
    p_int, p_mid, t, q = _column()
    with warnings.catch_warnings():
        warnings.simplefilter("always", UserWarning)
        with pytest.warns(UserWarning, match="synthetic grey-gas"):
            brightness_temperature(p_int, p_mid, t, q, channels=(14,), sensor_zenith=0.0)
