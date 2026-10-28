import numpy as np
import pytest

from crtm_py.atmosphere import (
    atmosphere_from_upp,
    ozone_kgkg_to_ppmv,
    specific_humidity_to_mixing_ratio_gkg,
)
from crtm_py.clouds import Hydrometeors, air_density, effective_radius
from crtm_py.constants import CONSTOZ, G, PI, RD
from crtm_py.surface import (
    BROADLEAF_FOREST,
    PINE_FOREST,
    URBAN_CONCRETE,
    map_land_type,
    skin_temperature_from_theta,
    surface_from_upp,
)


def test_water_vapor_and_ozone_units():
    q = 0.01
    assert specific_humidity_to_mixing_ratio_gkg(q) == pytest.approx(0.01 * 1000.0 / 0.99)
    assert ozone_kgkg_to_ppmv(0.0) == pytest.approx(1.0e-10)
    assert ozone_kgkg_to_ppmv(1.0e-6) == pytest.approx(1.0e-6 * CONSTOZ)


def test_wsm6_cloud_radius_matches_the_closed_form():
    pressure = 85000.0
    temperature = 280.0
    q = 0.01
    qqw = 1.0e-4
    rho = pressure / (RD * temperature * (1.0 + 0.608 * q))
    diameter_um = 1.0e6 * ((6.0 * rho * qqw) / (PI * 1000.0 * 3.0e8)) ** (1.0 / 3.0)
    got = effective_radius(
        pressure, temperature, q, qqw, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6, "C"
    )
    assert float(got) == pytest.approx(0.5 * diameter_um, rel=1e-12)
    assert air_density(pressure, temperature, q) == pytest.approx(rho)


def test_wsm6_ice_sits_on_the_radius_cap():
    got = effective_radius(
        50000.0, 240.0, 0.001, 0.0, 1.0e-5, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6, "i"
    )
    assert float(got) == pytest.approx(250.0, rel=1e-6)


def test_zero_condensate_has_zero_radius():
    got = effective_radius(
        80000.0, 280.0, 0.005, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0e6, 1.0e5, 8, "R"
    )
    assert float(got) == 0.0


def test_thompson_cloud_radius_is_inside_the_clip():
    got = effective_radius(
        90000.0, 285.0, 0.008, 1.0e-4, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8, "C"
    )
    assert 2.505 <= float(got) <= 25.0


def test_unsupported_microphysics_returns_zero_radius():
    got = effective_radius(
        90000.0, 285.0, 0.008, 1.0e-3, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 28, "C"
    )
    assert float(got) == 0.0


def test_land_type_tables():
    assert int(map_land_type(1, 1)) == PINE_FOREST
    assert int(map_land_type(1, 0)) == URBAN_CONCRETE
    assert int(map_land_type(1, 2)) == BROADLEAF_FOREST
    with pytest.raises(ValueError):
        map_land_type(1, 9)


def test_skin_temperature():
    assert skin_temperature_from_theta(300.0, 1.0e5) == pytest.approx(300.0)
    colder = skin_temperature_from_theta(300.0, 90000.0)
    assert colder < 300.0


def test_surface_water_land_and_igbp_snow():
    water = surface_from_upp(skin_temperature=265.0, sea_mask=1.0, ivgtyp=1, vegetation_fraction=0.4)
    assert water.water_coverage[0] == pytest.approx(1.0)
    assert water.vegetation_fraction[0] == 0.0
    assert water.water_temperature[0] == pytest.approx(270.0)

    snow = surface_from_upp(
        skin_temperature=270.0,
        sea_mask=0.0,
        ivgtyp=4,
        snow_fraction=0.4,
        u10=3.0,
        v10=4.0,
    )
    assert snow.land_coverage[0] == pytest.approx(0.6)
    assert snow.snow_coverage[0] == pytest.approx(0.4)
    assert snow.wind_speed[0] == pytest.approx(5.0)

    ice_class = surface_from_upp(
        skin_temperature=260.0, sea_mask=0.0, ivgtyp=15, ivegsrc=1, novegtype=20, snow_fraction=0.2
    )
    assert ice_class.snow_coverage[0] == pytest.approx(1.0)
    assert ice_class.land_coverage[0] == 0.0


def test_atmosphere_flips_bottom_up_and_converts_units():
    # Bottom-up: index 0 is the ground.
    p_int = np.array([100000.0, 50000.0, 20000.0])
    p_mid = np.array([80000.0, 35000.0])
    t = np.array([290.0, 230.0])
    q = np.array([0.01, 0.001])
    atm = atmosphere_from_upp(p_int, p_mid, t, q, imp_physics=0, vertical_order="bottom_up")
    assert atm.temperature[0, 0] == pytest.approx(230.0)
    assert atm.temperature[0, -1] == pytest.approx(290.0)
    assert atm.level_pressure_hpa[0, 0] == pytest.approx(0.005)
    assert atm.h2o_gkg[0, -1] == pytest.approx(0.01 * 1000.0 / 0.99)
    assert atm.o3_ppmv[0, 0] == pytest.approx(1.0e-10)
    assert atm.clouds == ()


def test_column_water_uses_model_dp():
    p_int = np.array([20000.0, 50000.0, 100000.0])
    p_mid = np.array([35000.0, 80000.0])
    t = np.array([230.0, 290.0])
    q = np.array([0.001, 0.01])
    hydro = Hydrometeors(qqw=np.array([0.0, 0.002]))
    atm = atmosphere_from_upp(
        p_int, p_mid, t, q, hydrometeors=hydro, imp_physics=99, vertical_order="top_down"
    )
    expected_lower = 0.002 * (100000.0 - 50000.0) / G
    water = atm.clouds[0]
    ice = atm.clouds[1]
    assert water.cloud_type == "water"
    assert water.water_content[0, -1] == pytest.approx(expected_lower)
    assert water.effective_radius[0, 0] == pytest.approx(10.0)
    assert ice.effective_radius[0, 0] == pytest.approx(50.0)


def test_identical_layer_pressures_are_nudged():
    p_int = np.array([40000.0, 70000.0, 100000.0])
    p_mid = np.array([50000.0, 50000.1])
    t = np.array([250.0, 280.0])
    q = np.array([0.001, 0.004])
    atm = atmosphere_from_upp(p_int, p_mid, t, q, imp_physics=0, vertical_order="top_down")
    assert atm.pressure_hpa[0, 0] == pytest.approx(500.0 + 0.005)


def test_ferrier_ice_radius_is_the_inline_value():
    p_int = np.array([50000.0, 100000.0])
    p_mid = np.array([80000.0])
    t = np.array([280.0])
    q = np.array([0.005])
    hydro = Hydrometeors(qqi=np.array([1.0e-5]), qqs=np.array([1.0e-4]), f_rimef=np.array([1.0]))
    atm = atmosphere_from_upp(
        p_int, p_mid, t, q, hydrometeors=hydro, imp_physics=5, vertical_order="top_down"
    )
    by_type = {c.cloud_type: c for c in atm.clouds}
    assert by_type["ice"].effective_radius[0, 0] == pytest.approx(75.0)
    assert by_type["snow"].water_content[0, 0] > 0.0
    assert by_type["graupel"].water_content[0, 0] == 0.0
    assert by_type["hail"].water_content[0, 0] == 0.0


def test_convective_condensate_is_added_between_top_and_bottom():
    # Top-down layers 1..4. htop=1, hbot=4 → difference 3 > 1, all four layers.
    p_int = np.array([10000.0, 30000.0, 50000.0, 80000.0, 100000.0])
    p_mid = np.array([20000.0, 40000.0, 65000.0, 90000.0])
    t = np.array([220.0, 240.0, 270.0, 290.0])
    q = np.full(4, 0.001)
    cnvcfr = np.array([0.5])
    atm = atmosphere_from_upp(
        p_int,
        p_mid,
        t,
        q,
        imp_physics=99,
        vertical_order="top_down",
        icu_physics=2,
        hbot=np.array([4.0]),
        htop=np.array([1.0]),
        cnvcfr=cnvcfr,
    )
    water = {c.cloud_type: c for c in atm.clouds}["water"]
    ice = {c.cloud_type: c for c in atm.clouds}["ice"]
    # Cold layers (220, 240) receive ice; warm layers receive cloud water.
    assert ice.water_content[0, 0] > 0.0
    assert water.water_content[0, 0] == 0.0
    assert water.water_content[0, -1] > 0.0
    dp = 100000.0 - 80000.0
    assert water.water_content[0, -1] == pytest.approx(0.5 * 0.1e-3 * dp / G)


def test_reversed_column_is_rejected():
    p_int = np.array([100000.0, 50000.0, 20000.0])
    p_mid = np.array([80000.0, 35000.0])
    t = np.array([290.0, 230.0])
    q = np.array([0.01, 0.001])
    with pytest.raises(ValueError, match="vertical_order"):
        atmosphere_from_upp(p_int, p_mid, t, q, vertical_order="top_down", imp_physics=0)
