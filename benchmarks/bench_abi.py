"""Time ABI G16 brightness temperature on repeated CRTM test columns.

Two baselines from before this batch path:

* Carr Fire wrfout, 199 by 199 columns, 79 layers, ABI channels 7–16,
  scattering on. On main that was about 139–169 ms per column per core.
  Eight cores took 11–14 minutes of wall clock for the whole field.
  ``cProfile`` put the time in ``ada_radiance``.
* The per-column path on this test profile was about 10 ms clear and
  64 ms with a liquid cloud (about 167 ms with ice over liquid). 39601
  liquid columns is on the order of 40 minutes on one Python thread.

``cloudy`` puts 0.05 kg m^-2 of liquid water in every column between
700 and 900 hPa, so every column enters ADA. ``mixed`` does that on
even columns only and leaves the odd columns clear. The cloud mask is
what makes ``mixed`` cheaper than ``cloudy``.

Run from the repository root, with coefficients available::

    OMP_NUM_THREADS=1 CRTM_PY_COEFF_DIR=/path/to/coeffs python benchmarks/bench_abi.py

``--full`` also times 199*199 columns. The default stops at 1000.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

from crtm_py.atmosphere import Atmosphere, Cloud
from crtm_py.forward import ForwardModel
from crtm_py.geometry import Geometry
from crtm_py.surface import Surface


def _profile():
    path = os.path.join(os.path.dirname(__file__), "..", "tests", "data", "abi_g16_clearsky_92.npz")
    data = np.load(path)
    return data


def _column(n, clouds_mode):
    data = _profile()
    level = np.repeat(data["level_pressure_hpa"][None, :], n, axis=0)
    pressure = np.repeat(data["pressure_hpa"][None, :], n, axis=0)
    temperature = np.repeat(data["temperature"][None, :], n, axis=0)
    h2o = np.repeat(data["h2o_gkg"][None, :], n, axis=0)
    o3 = np.repeat(data["o3_ppmv"][None, :], n, axis=0)
    clouds = ()
    if clouds_mode != "clear":
        water = np.where((data["pressure_hpa"] >= 700.0) & (data["pressure_hpa"] <= 900.0), 0.05, 0.0)
        water = np.repeat(water[None, :], n, axis=0)
        if clouds_mode == "mixed":
            water[1::2] = 0.0
        clouds = (Cloud("water", water, np.full(water.shape, 10.0)),)
    atmosphere = Atmosphere(
        level_pressure_hpa=level,
        pressure_hpa=pressure,
        temperature=temperature,
        h2o_gkg=h2o,
        o3_ppmv=o3,
        clouds=clouds,
        interface_pressure_pa=level * 100.0,
    )
    surface = Surface(
        land_type=np.ones(n, dtype=np.int64),
        wind_speed=np.full(n, 5.0),
        water_coverage=np.ones(n),
        land_coverage=np.zeros(n),
        ice_coverage=np.zeros(n),
        snow_coverage=np.zeros(n),
        land_temperature=np.full(n, 290.0),
        snow_temperature=np.full(n, 280.0),
        water_temperature=np.full(n, 290.0),
        ice_temperature=np.full(n, 280.0),
        soil_moisture_content=np.full(n, 0.05),
        vegetation_fraction=np.zeros(n),
        soil_temperature=np.full(n, 283.0),
        snow_depth=np.zeros(n),
    )
    geometry = Geometry(np.zeros(n), np.full(n, 180.0), np.zeros(n))
    return atmosphere, surface, geometry


def _time(model, n, clouds_mode, chunk_size):
    atmosphere, surface, geometry = _column(n, clouds_mode)
    scattering = clouds_mode != "clear"
    model.run(atmosphere, surface, geometry, include_scattering=scattering, chunk_size=chunk_size)
    start = time.perf_counter()
    model.run(atmosphere, surface, geometry, include_scattering=scattering, chunk_size=chunk_size)
    return time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="also time 39601 columns")
    parser.add_argument("--chunk-size", type=int, default=256)
    args = parser.parse_args()
    if not os.environ.get("CRTM_PY_COEFF_DIR"):
        raise SystemExit("set CRTM_PY_COEFF_DIR to a directory with abi_g16 coefficients")
    model = ForwardModel("abi_g16")
    sizes = [1, 100, 1000]
    if args.full:
        sizes.append(199 * 199)
    print(f"chunk_size={args.chunk_size}  OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', 'unset')}")
    print(f"{'columns':>8} {'clouds':>8} {'seconds':>10} {'ms/column':>10}")
    for clouds_mode in ("clear", "cloudy", "mixed"):
        for n in sizes:
            seconds = _time(model, n, clouds_mode, args.chunk_size)
            print(f"{n:8d} {clouds_mode:>8} {seconds:10.3f} {seconds / n * 1000:10.3f}", flush=True)


if __name__ == "__main__":
    main()
