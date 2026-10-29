"""ASYMTX against one layer matrix dumped from CRTM v2.4.0.

The matrix is HH for ABI G16 channel 7, layer 90 of the liquid-cloud
profile. It is not symmetric. The eigenvalues are the values printed by
the Fortran routine on that same matrix.
"""

import numpy as np

from crtm_py.asymtx import asymtx

HH = np.array(
    [
        [158.35596613674937, -76.63110742033733, -37.73598270348818, -7.332815689988553, 0.0],
        [-2.031732315098851, 5.786306477943747, -2.138635361705827, -0.42081311622171114, 0.0],
        [-0.293062955934217, -0.6562723115214956, 1.3970021111001256, -0.36254748446267704, 0.0],
        [-0.016075906937431637, -0.1093285470888523, -0.45275254708363155, 0.5940210194128412, 0.0],
        [0.007421058418071631, -0.017970565646613132, -0.3390405246915009, -0.6329267337337521, 1.0],
    ]
)
EVAL = np.array([1.0, 159.43587260437255, 5.3032570020576335, 0.2108095065766149, 1.1833566321992517])


def test_layer_matrix_is_not_symmetric():
    assert np.max(np.abs(HH - HH.T)) > 10.0


def test_asymtx_matches_fortran_eigenvalues():
    values, vectors, ier = asymtx(HH)
    assert ier == 0
    np.testing.assert_allclose(values, EVAL, rtol=0.0, atol=1e-9)
    for value, vector in zip(values, vectors.T):
        residual = HH @ vector - value * vector
        assert np.max(np.abs(residual)) < 1e-8
