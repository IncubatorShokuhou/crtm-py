"""US Standard layers inserted above the user top."""

from crtm_py.toa import extra_layer_count, prepend_standard_atmosphere
import numpy as np


def test_ecmwf_top_needs_eight_standard_layers():
    # The CRTM clear-sky test profile starts at 0.714 hPa.
    assert extra_layer_count(0.714) == 8
    assert extra_layer_count(0.005) == 0


def test_added_layers_meet_the_user_top():
    level = np.array([0.714, 1.0, 10.0])
    pressure = np.array([0.85, 5.0])
    temperature = np.array([256.0, 250.0])
    h2o = np.array([0.003, 0.004])
    o3 = np.array([3.0, 4.0])
    new_level, new_p, new_t, new_h, new_o, n = prepend_standard_atmosphere(
        level, pressure, temperature, h2o, o3
    )
    assert n == 8
    assert new_level[0] == 0.005
    assert new_level[8] == 0.714
    assert new_p.size == pressure.size + 8
    assert new_t[8] == 256.0
    assert new_o[8] == 3.0
