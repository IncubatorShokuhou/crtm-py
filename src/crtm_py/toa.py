"""US Standard layers CRTM inserts above a user profile.

The numbers are the AIRS 101-level climatology in ``CRTM_Model_Profiles``:
level pressure, temperature, water vapour (g/kg) and ozone (ppmv).
``CRTM_Atmosphere_AddLayers`` uses them when the user top is below 0.005 hPa.
"""

from __future__ import annotations

import numpy as np

TOA_PRESSURE_HPA = 0.005
_MIN_ABSORBER = 1.0e-76

_LEVEL_PRESSURE_HPA = np.array([
    0.005, 0.016065, 0.038383, 0.076879, 0.136954,
    0.224412, 0.345404, 0.506374, 0.714023, 0.975274,
    1.29724, 1.6872, 2.152573, 2.700897, 3.339812,
    4.077038, 4.920362, 5.877623, 6.956695, 8.16548,
    9.511889, 11.003835, 12.649223, 14.455936, 16.431833,
    18.584732, 20.922408, 23.452583, 26.182918, 29.121009,
    32.274378, 35.650467, 39.256633, 43.100144, 47.188171,
    51.527786, 56.125953, 60.98953, 66.125259, 71.539768,
    77.23956, 83.231016, 89.52039, 96.113803, 103.017244,
    110.236565, 117.777481, 125.645562, 133.84624, 142.384796,
    151.266366, 160.495939, 170.078348, 180.018279, 190.32026,
    200.988665, 212.027712, 223.441461, 235.233814, 247.408514,
    259.969142, 272.91912, 286.261706, 300, 314.136936,
    328.675286, 343.617659, 358.966503, 374.724098, 390.892566,
    407.473861, 424.469776, 441.881941, 459.711821, 477.960722,
    496.629785, 515.719989, 535.232153, 555.166935, 575.524832,
    596.306182, 617.511163, 639.139797, 661.191946, 683.667316,
    706.56546, 729.885772, 753.627494, 777.789716, 802.371376,
    827.371259, 852.788003, 878.620096, 904.86588, 931.523549,
    958.591154, 986.066601, 1013.947655, 1042.23194, 1070.91694,
    1100
], dtype=np.float64)
_LEVEL_TEMPERATURE = np.array([
    190.195, 203.667, 215.175, 226.809, 237.789,
    247.507, 256.483, 263.555, 268.955, 270.636,
    267.045, 261.561, 256.386, 251.67, 247.321,
    243.261, 239.551, 236.057, 232.759, 229.832,
    228.503, 227.233, 226.133, 225.245, 224.392,
    223.595, 222.847, 222.126, 221.425, 220.733,
    220.065, 219.42, 218.797, 218.196, 217.614,
    217.106, 216.7, 216.7, 216.7, 216.7,
    216.7, 216.7, 216.7, 216.7, 216.7,
    216.7, 216.7, 216.7, 216.7, 216.7,
    216.7, 216.7, 216.7, 216.7, 216.7,
    216.723, 216.757, 216.79, 218.296, 220.415,
    222.495, 224.553, 226.585, 228.58, 230.577,
    232.588, 234.564, 236.514, 238.474, 240.401,
    242.296, 244.201, 246.087, 247.943, 249.783,
    251.627, 253.442, 255.229, 257.021, 258.798,
    260.549, 262.275, 264.015, 265.73, 267.42,
    269.095, 270.776, 272.433, 274.067, 275.689,
    277.314, 278.917, 280.497, 282.066, 283.643,
    285.2, 286.736, 288.251, 289.746, 291.221,
    292.678
], dtype=np.float64)
_LEVEL_H2O_GKG = np.array([
    0.00088699704, 0.00152427987, 0.00201101378, 0.00240599752, 0.00272437587,
    0.00296236227, 0.00310396312, 0.00319368814, 0.00323610094, 0.00325981037,
    0.00325671227, 0.00323196162, 0.00319285497, 0.00313971111, 0.00310625749,
    0.00308100578, 0.00306248483, 0.00304448043, 0.00302081059, 0.00299803553,
    0.0029744045, 0.00295184624, 0.0029251134, 0.00289196774, 0.00286016092,
    0.00282977471, 0.00280069281, 0.00277267144, 0.00273864263, 0.00268478764,
    0.00264118787, 0.00259714199, 0.00254483821, 0.00250568261, 0.00247311126,
    0.00244672175, 0.00242271064, 0.00240621873, 0.0023923837, 0.00238458004,
    0.00238110201, 0.00238850647, 0.00239914434, 0.00242737549, 0.00245492999,
    0.00271901083, 0.00299417069, 0.00324509817, 0.00347714919, 0.00378040842,
    0.00495999293, 0.00611455674, 0.00757663188, 0.00942311177, 0.0112322375,
    0.0142466434, 0.0178602949, 0.0214039606, 0.0273154771, 0.0341753036,
    0.0409074844, 0.0542724979, 0.071713328, 0.088841782, 0.115945509,
    0.156046471, 0.195454032, 0.234255117, 0.27275076, 0.310602956,
    0.347828689, 0.406535547, 0.470306893, 0.533055556, 0.601901014,
    0.685096335, 0.766994753, 0.847628708, 0.965099457, 1.09450413,
    1.22196301, 1.34952968, 1.52006224, 1.68809431, 1.85367915,
    2.03383153, 2.26692831, 2.49671002, 2.72324444, 2.94770972,
    3.17163654, 3.39246541, 3.61025923, 3.83453885, 4.08727861,
    4.33660869, 4.5825977, 4.82530792, 5.06480362, 5.30114363,
    5.53438577
], dtype=np.float64)
_LEVEL_O3_PPMV = np.array([
    0.473303997, 0.274146902, 0.280243651, 0.510480276, 0.831073636,
    1.12571713, 1.5810462, 2.16173433, 2.8709199, 3.74371439,
    4.73885897, 5.61330664, 6.34570524, 7.09761422, 7.50525354,
    7.77591671, 7.81936233, 7.80536994, 7.56992926, 7.33375237,
    7.02220367, 6.72479815, 6.44285088, 6.17639446, 5.9207009,
    5.68739719, 5.47387081, 5.26813104, 5.03222902, 4.69211312,
    4.38133099, 4.07406867, 3.74952993, 3.39674005, 3.03656514,
    2.78144823, 2.5264243, 2.2352837, 1.96713487, 1.74480839,
    1.53398068, 1.34350343, 1.16379657, 1.01995634, 0.879563704,
    0.781925294, 0.689400605, 0.615902146, 0.556537854, 0.499365854,
    0.453753714, 0.409109075, 0.372418996, 0.345262678, 0.318655731,
    0.288188006, 0.255989671, 0.224414924, 0.195655516, 0.168400469,
    0.141652577, 0.123541576, 0.110965251, 0.0986141761, 0.0873600481,
    0.0774538294, 0.0677189026, 0.0591969666, 0.0563118453, 0.0534749481,
    0.0506850024, 0.0480182173, 0.0454157757, 0.0428550687, 0.0408112252,
    0.0398355325, 0.0388750495, 0.0379293958, 0.036905594, 0.0358639488,
    0.0348379664, 0.0338620749, 0.0336773243, 0.0334952826, 0.0333158922,
    0.0331308212, 0.0329213219, 0.0327148022, 0.032511201, 0.0321398618,
    0.0313748034, 0.0306203294, 0.0298762248, 0.0291576054, 0.0284997705,
    0.0278508104, 0.0272105464, 0.0265788165, 0.0259554534, 0.025340304,
    0.0247332178
], dtype=np.float64)


def extra_layer_count(top_pressure_hpa: float) -> int:
    """``Extra_Layers``: model levels strictly above the user top, plus the splice."""
    if top_pressure_hpa <= TOA_PRESSURE_HPA:
        return 0
    for index in range(1, _LEVEL_PRESSURE_HPA.size):
        if _LEVEL_PRESSURE_HPA[index] >= top_pressure_hpa:
            return index
    return 0


def prepend_standard_atmosphere(level_pressure_hpa, pressure_hpa, temperature, h2o_gkg, o3_ppmv):
    """Insert US Standard layers from 0.005 hPa down to the user top.

    Layer amounts are shifted so the lowest added layer meets a log-pressure
    extrapolation of the user's top two layers, which is ``Shift_Profile``.
    Returns the extended arrays and the number of layers inserted.
    """
    level = np.asarray(level_pressure_hpa, dtype=np.float64)
    pressure = np.asarray(pressure_hpa, dtype=np.float64)
    temp = np.asarray(temperature, dtype=np.float64)
    h2o = np.asarray(h2o_gkg, dtype=np.float64)
    o3 = np.asarray(o3_ppmv, dtype=np.float64)
    n = extra_layer_count(float(level[0]))
    if n < 1 or pressure.size < 2:
        return level, pressure, temp, h2o, o3, 0
    pl = _LEVEL_PRESSURE_HPA[: n + 1].copy()
    tl = _LEVEL_TEMPERATURE[: n + 1].copy()
    hl = _LEVEL_H2O_GKG[: n + 1].copy()
    ol = _LEVEL_O3_PPMV[: n + 1].copy()
    top = float(level[0])
    span = np.log(pl[n]) - np.log(pl[n - 1])
    lpoly = (np.log(top) - np.log(pl[n - 1])) / span
    pl[n] = top
    tl[n] = tl[n - 1] + (tl[n] - tl[n - 1]) * lpoly
    hl[n] = hl[n - 1] + (hl[n] - hl[n - 1]) * lpoly
    ol[n] = ol[n - 1] + (ol[n] - ol[n - 1]) * lpoly
    p_layer = (pl[1:] - pl[:-1]) / np.log(pl[1:] / pl[:-1])
    t_layer = 0.5 * (tl[:-1] + tl[1:])
    h_layer = 0.5 * (hl[:-1] + hl[1:])
    o_layer = 0.5 * (ol[:-1] + ol[1:])
    user_span = np.log(pressure[1]) - np.log(pressure[0])
    shift_lpoly = (np.log(p_layer[-1]) - np.log(pressure[0])) / user_span

    def _shift(layer, x0, x1):
        target = x0 + (x1 - x0) * shift_lpoly
        return layer + (target - layer[-1])

    t_layer = _shift(t_layer, temp[0], temp[1])
    h_layer = np.maximum(_shift(h_layer, h2o[0], h2o[1]), _MIN_ABSORBER)
    o_layer = np.maximum(_shift(o_layer, o3[0], o3[1]), _MIN_ABSORBER)
    return (
        np.concatenate([pl[:n], level]),
        np.concatenate([p_layer, pressure]),
        np.concatenate([t_layer, temp]),
        np.concatenate([h_layer, h2o]),
        np.concatenate([o_layer, o3]),
        n,
    )


def prepend_standard_atmosphere_batch(level_pressure_hpa, pressure_hpa, temperature, h2o_gkg, o3_ppmv):
    """Batch form of :func:`prepend_standard_atmosphere`.

    Inputs are ``(n_profile, n_level)`` and ``(n_profile, n_layer)``. Profiles
    that share a pressure grid share the added-layer geometry; the temperature
    and absorber shift is still per column.
    """
    level = np.asarray(level_pressure_hpa, dtype=np.float64)
    pressure = np.asarray(pressure_hpa, dtype=np.float64)
    temp = np.asarray(temperature, dtype=np.float64)
    h2o = np.asarray(h2o_gkg, dtype=np.float64)
    o3 = np.asarray(o3_ppmv, dtype=np.float64)
    if level.ndim == 1:
        level = level.reshape(1, -1)
        pressure = pressure.reshape(1, -1)
        temp = temp.reshape(1, -1)
        h2o = h2o.reshape(1, -1)
        o3 = o3.reshape(1, -1)
    n_add = extra_layer_count(float(level[0, 0]))
    if n_add < 1 or pressure.shape[1] < 2:
        return level, pressure, temp, h2o, o3, 0
    if not (np.allclose(level, level[0]) and np.allclose(pressure, pressure[0])):
        parts = [
            prepend_standard_atmosphere(level[i], pressure[i], temp[i], h2o[i], o3[i])
            for i in range(level.shape[0])
        ]
        return (
            np.stack([p[0] for p in parts]),
            np.stack([p[1] for p in parts]),
            np.stack([p[2] for p in parts]),
            np.stack([p[3] for p in parts]),
            np.stack([p[4] for p in parts]),
            parts[0][5],
        )
    one = prepend_standard_atmosphere(level[0], pressure[0], temp[0], h2o[0], o3[0])
    # Geometry is shared. Rebuild the shift from each column so the splice matches.
    pl = _LEVEL_PRESSURE_HPA[: n_add + 1].copy()
    tl = _LEVEL_TEMPERATURE[: n_add + 1].copy()
    hl = _LEVEL_H2O_GKG[: n_add + 1].copy()
    ol = _LEVEL_O3_PPMV[: n_add + 1].copy()
    top = float(level[0, 0])
    span = np.log(pl[n_add]) - np.log(pl[n_add - 1])
    lpoly = (np.log(top) - np.log(pl[n_add - 1])) / span
    pl[n_add] = top
    tl[n_add] = tl[n_add - 1] + (tl[n_add] - tl[n_add - 1]) * lpoly
    hl[n_add] = hl[n_add - 1] + (hl[n_add] - hl[n_add - 1]) * lpoly
    ol[n_add] = ol[n_add - 1] + (ol[n_add] - ol[n_add - 1]) * lpoly
    p_layer = (pl[1:] - pl[:-1]) / np.log(pl[1:] / pl[:-1])
    t_layer = 0.5 * (tl[:-1] + tl[1:])
    h_layer = 0.5 * (hl[:-1] + hl[1:])
    o_layer = 0.5 * (ol[:-1] + ol[1:])
    shift = (np.log(p_layer[-1]) - np.log(pressure[0, 0])) / (np.log(pressure[0, 1]) - np.log(pressure[0, 0]))

    def _shift(layer, column):
        target = column[:, 0] + (column[:, 1] - column[:, 0]) * shift
        return layer + (target - layer[-1])[:, None]

    t_add = _shift(t_layer, temp)
    h_add = np.maximum(_shift(h_layer, h2o), _MIN_ABSORBER)
    o_add = np.maximum(_shift(o_layer, o3), _MIN_ABSORBER)
    level_out = np.broadcast_to(np.concatenate([pl[:n_add], level[0]]), (level.shape[0], n_add + level.shape[1])).copy()
    pressure_out = np.broadcast_to(np.concatenate([p_layer, pressure[0]]), (level.shape[0], n_add + pressure.shape[1])).copy()
    return (
        level_out,
        pressure_out,
        np.concatenate([t_add, temp], axis=1),
        np.concatenate([h_add, h2o], axis=1),
        np.concatenate([o_add, o3], axis=1),
        n_add,
    )
