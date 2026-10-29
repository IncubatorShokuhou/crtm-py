"""Batched columns agree with a single column and with each other across chunks."""

import numpy as np

from crtm_py.atmosphere import Atmosphere, Cloud
from crtm_py.forward import ForwardModel
from crtm_py.geometry import Geometry
from crtm_py.surface import Surface

from tests.test_crtm_reference import PROFILE, _atmosphere, _surface


def _repeat(n, clouds):
    atm = _atmosphere(clouds)

    def tile(array):
        return np.repeat(array, n, axis=0)

    tiled_clouds = tuple(
        Cloud(cloud.cloud_type, tile(cloud.water_content), tile(cloud.effective_radius)) for cloud in atm.clouds
    )
    return Atmosphere(
        level_pressure_hpa=tile(atm.level_pressure_hpa),
        pressure_hpa=tile(atm.pressure_hpa),
        temperature=tile(atm.temperature),
        h2o_gkg=tile(atm.h2o_gkg),
        o3_ppmv=tile(atm.o3_ppmv),
        clouds=tiled_clouds,
        interface_pressure_pa=tile(atm.interface_pressure_pa),
    )


def _surface_n(n):
    base = _surface()
    values = {}
    for name, value in base.__dict__.items():
        if isinstance(value, np.ndarray):
            values[name] = np.repeat(value, n)
        else:
            values[name] = value
    return Surface(**values)


def test_batch_matches_one_column_with_and_without_scattering(coeff_dir=None):
    import os
    from pathlib import Path

    root = os.environ.get("CRTM_PY_COEFF_DIR")
    if not root or not any(Path(root).rglob("abi_g16.SpcCoeff.bin")):
        import pytest

        pytest.skip("CRTM_PY_COEFF_DIR has no abi_g16 coefficients")
    pressure = PROFILE["pressure_hpa"]
    water = np.where((pressure >= 700.0) & (pressure <= 900.0), 0.05, 0.0)
    liquid = (Cloud("water", water.reshape(1, -1), np.full((1, pressure.size), 10.0)),)
    model = ForwardModel("abi_g16", coeff_path=root)
    for clouds, scattering in (((), False), (liquid, True)):
        one = model.run(
            _atmosphere(clouds),
            _surface(),
            Geometry(np.array([30.0]), np.array([180.0]), np.array([0.0])),
            include_scattering=scattering,
        )
        many = model.run(
            _repeat(5, clouds),
            _surface_n(5),
            Geometry(np.full(5, 30.0), np.full(5, 180.0), np.zeros(5)),
            include_scattering=scattering,
            chunk_size=2,
        )
        reference = np.broadcast_to(one.brightness_temperature, many.brightness_temperature.shape)
        np.testing.assert_allclose(many.brightness_temperature, reference, atol=1e-6, rtol=0)


def test_mixed_clear_and_cloudy_columns_match_runs_alone():
    """One chunk may hold clear columns next to columns cloudy on different layers."""
    import os
    from pathlib import Path

    root = os.environ.get("CRTM_PY_COEFF_DIR")
    if not root or not any(Path(root).rglob("abi_g16.SpcCoeff.bin")):
        import pytest

        pytest.skip("CRTM_PY_COEFF_DIR has no abi_g16 coefficients")
    pressure = PROFILE["pressure_hpa"]
    liquid = np.where((pressure >= 700.0) & (pressure <= 900.0), 0.05, 0.0)
    ice = np.where((pressure >= 300.0) & (pressure <= 500.0), 0.02, 0.0)
    thin = np.zeros_like(pressure)
    near = np.argmin(np.abs(pressure - 800.0))
    thin[near] = 0.05
    n = 4
    water = np.stack([np.zeros_like(pressure), liquid, thin, liquid])
    ice_water = np.stack([np.zeros_like(pressure), np.zeros_like(pressure), np.zeros_like(pressure), ice])
    atm = _repeat(1, ())
    level = np.repeat(atm.level_pressure_hpa, n, axis=0)
    clouds = (
        Cloud("water", water, np.full((n, pressure.size), 10.0)),
        Cloud("ice", ice_water, np.full((n, pressure.size), 20.0)),
    )
    mixed = Atmosphere(
        level_pressure_hpa=level,
        pressure_hpa=np.repeat(atm.pressure_hpa, n, axis=0),
        temperature=np.repeat(atm.temperature, n, axis=0),
        h2o_gkg=np.repeat(atm.h2o_gkg, n, axis=0),
        o3_ppmv=np.repeat(atm.o3_ppmv, n, axis=0),
        clouds=clouds,
        interface_pressure_pa=np.repeat(atm.interface_pressure_pa, n, axis=0),
    )
    model = ForwardModel("abi_g16", coeff_path=root)
    geometry = Geometry(np.zeros(n), np.full(n, 180.0), np.zeros(n))
    together = model.run(mixed, _surface_n(n), geometry, include_scattering=True, chunk_size=n)
    for i in range(n):
        alone_atm = Atmosphere(
            level_pressure_hpa=mixed.level_pressure_hpa[i : i + 1],
            pressure_hpa=mixed.pressure_hpa[i : i + 1],
            temperature=mixed.temperature[i : i + 1],
            h2o_gkg=mixed.h2o_gkg[i : i + 1],
            o3_ppmv=mixed.o3_ppmv[i : i + 1],
            clouds=tuple(
                Cloud(cloud.cloud_type, cloud.water_content[i : i + 1], cloud.effective_radius[i : i + 1])
                for cloud in mixed.clouds
            ),
            interface_pressure_pa=mixed.interface_pressure_pa[i : i + 1],
        )
        alone = model.run(
            alone_atm,
            _surface_n(1),
            Geometry(np.array([0.0]), np.array([180.0]), np.array([0.0])),
            include_scattering=True,
        )
        np.testing.assert_allclose(
            together.brightness_temperature[i],
            alone.brightness_temperature.reshape(-1),
            atol=1e-6,
            rtol=0,
        )


def test_two_processes_match_one_process():
    import os
    from pathlib import Path

    root = os.environ.get("CRTM_PY_COEFF_DIR")
    if not root or not any(Path(root).rglob("abi_g16.SpcCoeff.bin")):
        import pytest

        pytest.skip("CRTM_PY_COEFF_DIR has no abi_g16 coefficients")
    pressure = PROFILE["pressure_hpa"]
    water = np.where((pressure >= 700.0) & (pressure <= 900.0), 0.05, 0.0)
    liquid = (Cloud("water", water.reshape(1, -1), np.full((1, pressure.size), 10.0)),)
    model = ForwardModel("abi_g16", coeff_path=root)
    atm = _repeat(4, liquid)
    surface = _surface_n(4)
    geometry = Geometry(np.zeros(4), np.full(4, 180.0), np.zeros(4))
    one = model.run(atm, surface, geometry, include_scattering=True, chunk_size=2, n_jobs=1)
    two = model.run(atm, surface, geometry, include_scattering=True, chunk_size=2, n_jobs=2)
    np.testing.assert_allclose(two.brightness_temperature, one.brightness_temperature, atol=1e-6, rtol=0)


def _coeff_root():
    import os
    from pathlib import Path

    root = os.environ.get("CRTM_PY_COEFF_DIR")
    if not root or not any(Path(root).rglob("abi_g16.SpcCoeff.bin")):
        import pytest

        pytest.skip("CRTM_PY_COEFF_DIR has no abi_g16 coefficients")
    return root


def test_legendre_count_follows_each_column_and_channel():
    """Stream count does not borrow another column's radius."""
    from crtm_py.batch_forward import _legendre_counts
    from crtm_py.cloudlut import n_streams_for_radius

    radius = np.array([0.0, 0.5, 2.0, 50.0])
    wavenumber = np.array([753.79, 2570.37])
    got = _legendre_counts(radius, wavenumber)
    for i, re in enumerate(radius):
        for j, wn in enumerate(wavenumber):
            expect = 0 if re <= 0.0 else n_streams_for_radius(float(re), float(wn))
            assert got[i, j] == expect
    # 2 µm is 4 streams on the window edge and 6 on channel 7.
    assert got[2, 0] < got[2, 1]
    assert got[1, 1] < got[3, 1]


def test_thin_and_thick_clouds_keep_their_own_stream_count():
    """A thick ice column must not raise the Legendre count of a thin neighbour.

    chunk_size 1 is the per-column path. 7 and 256 put the thin columns in
    the same chunk as the ice.
    """
    root = _coeff_root()
    pressure = PROFILE["pressure_hpa"]
    n_layer = pressure.size
    k_liquid = int(np.argmin(np.abs(pressure - 800.0)))
    # Six small-particle columns, one on-grid column whose channels disagree,
    # one thick ice column, and one wet column whose radius is non-positive.
    n = 9
    water = np.zeros((n, n_layer))
    water[:8, k_liquid] = 1.0e-4
    radius = np.zeros((n, n_layer))
    radius[0:6] = 0.5
    radius[6] = 2.0
    radius[7] = 0.0
    # Column 0's wet layer is 0.5 µm. A dry layer holds 50 µm, so the column
    # maximum is 50 µm. That maximum stays inside the column.
    ice = np.zeros((n, n_layer))
    ice[8] = np.where((pressure >= 300.0) & (pressure <= 500.0), 0.02, 0.0)
    ice_radius = np.zeros((n, n_layer))
    ice_radius[8] = 50.0
    radius[0, 0] = 50.0
    level = np.repeat(PROFILE["level_pressure_hpa"][None, :], n, axis=0)
    atm = Atmosphere(
        level_pressure_hpa=level,
        pressure_hpa=np.repeat(pressure[None, :], n, axis=0),
        temperature=np.repeat(PROFILE["temperature"][None, :], n, axis=0),
        h2o_gkg=np.repeat(PROFILE["h2o_gkg"][None, :], n, axis=0),
        o3_ppmv=np.repeat(PROFILE["o3_ppmv"][None, :], n, axis=0),
        clouds=(
            Cloud("water", water, radius),
            Cloud("ice", ice, ice_radius),
        ),
        interface_pressure_pa=level * 100.0,
    )
    model = ForwardModel("abi_g16", coeff_path=root)
    surface = _surface_n(n)
    geometry = Geometry(np.full(n, 30.0), np.full(n, 180.0), np.zeros(n))
    alone = []
    for i in range(n):
        one = Atmosphere(
            level_pressure_hpa=atm.level_pressure_hpa[i : i + 1],
            pressure_hpa=atm.pressure_hpa[i : i + 1],
            temperature=atm.temperature[i : i + 1],
            h2o_gkg=atm.h2o_gkg[i : i + 1],
            o3_ppmv=atm.o3_ppmv[i : i + 1],
            clouds=tuple(
                Cloud(cloud.cloud_type, cloud.water_content[i : i + 1], cloud.effective_radius[i : i + 1])
                for cloud in atm.clouds
            ),
            interface_pressure_pa=atm.interface_pressure_pa[i : i + 1],
        )
        result = model.run(
            one,
            _surface_n(1),
            Geometry(np.array([30.0]), np.array([180.0]), np.array([0.0])),
            include_scattering=True,
            chunk_size=1,
        )
        alone.append(np.asarray(result.brightness_temperature).reshape(-1))
    reference = np.stack(alone)
    for chunk_size in (1, 7, 256):
        got = model.run(atm, surface, geometry, include_scattering=True, chunk_size=chunk_size)
        np.testing.assert_allclose(got.brightness_temperature, reference, atol=1e-6, rtol=0)


def test_cloud_optics_follow_the_per_column_legendre_rule():
    """Batch cloud optics match ``layer_cloud_optics`` column by column."""
    from pathlib import Path

    from crtm_py.batch_forward import _Cloud, cloud_optics_batch
    from crtm_py.cloudlut import layer_cloud_optics, load_cloudcoeff
    from crtm_py.spc import load_spccoeff

    root = _coeff_root()
    pressure = PROFILE["pressure_hpa"]
    n_layer = pressure.size
    k_liquid = int(np.argmin(np.abs(pressure - 800.0)))
    water = np.zeros((2, n_layer))
    water[0, k_liquid] = 1.0e-4
    radius = np.full((2, n_layer), 0.5)
    radius[0, 0] = 50.0
    radius[1] = 50.0
    ice = np.zeros((2, n_layer))
    ice[1] = np.where((pressure >= 300.0) & (pressure <= 500.0), 0.02, 0.0)
    ice_radius = np.zeros((2, n_layer))
    ice_radius[1] = 50.0
    spc = load_spccoeff(next(Path(root).rglob("abi_g16.SpcCoeff.bin")))
    wn = np.array([float(spc.wavenumber[spc.channel_index(b)]) for b in range(7, 17)])
    coeff = load_cloudcoeff(next(Path(root).rglob("CloudCoeff.bin")))
    tau, ssa, phase, n_leg = cloud_optics_batch(
        coeff,
        wn,
        (_Cloud("water", water, radius), _Cloud("ice", ice, ice_radius)),
        True,
    )
    for i in range(2):
        scalar = (
            Cloud("water", water[i], radius[i]),
            Cloud("ice", ice[i], ice_radius[i]),
        )
        for ic, wavenumber in enumerate(wn):
            optical, albedo, moments = layer_cloud_optics(
                coeff, float(wavenumber), scalar, n_layer, scattering=True
            )
            np.testing.assert_allclose(tau[i, ic], optical, atol=1e-8, rtol=1e-7)
            np.testing.assert_allclose(ssa[i, ic], albedo, atol=1e-8, rtol=1e-7)
            if moments is None:
                assert n_leg[i, ic] == 0
                continue
            assert moments.shape[0] == int(n_leg[i, ic]) + 1
            active = np.any(moments != 0.0, axis=0)
            np.testing.assert_allclose(
                phase[i, ic, active, : moments.shape[0]], moments[:, active].T, atol=1e-8, rtol=1e-7
            )
            if np.any(~active):
                np.testing.assert_allclose(phase[i, ic, ~active, 1 : moments.shape[0]], 0.0, atol=1e-12)
