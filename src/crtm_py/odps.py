"""ODPS gas optical depth, CRTM v2.4.

The predictor formulas are ``ODPS_Compute_Predictor`` for groups 1 and 2.
Water-vapour line absorption uses compact OPTRAN when
``OSignificance == 1`` (``Add_OPTRAN_wloOD``). User profiles are averaged
onto the coefficient pressure grid with ``LayerAvg`` and the slant-path
optical depth is interpolated back and divided by the surface secant, which
is what ``ODPS_Compute_AtmAbsorption`` stores as nadir optical depth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from crtm_py.crtm_binary import FortranFile

# Profile_Utility_Parameters / Fundamental_Constants
_MW_H2O = 18.01528
_MW_DRY = 28.9648
_R0 = 8.314472
_G0 = 9.80665
_EPS = _MW_H2O / _MW_DRY
_R_DRY = _R0 / (_MW_DRY * 1.0e-3)
_C_TV = (1.0 / _EPS - 1.0) / 1000.0
_CC_KM = 0.001 * _R_DRY / _G0
_EARTH_RADIUS_KM = 6370.0
_SMALLDIFF = 1.0e-20
_MAX_OD = 20.0
_LIMIT_EXP = 20.0
_LIMIT_LOG = 4.8e8
_RECIPROCAL_GRAVITY = 1.0 / 980.665
_MIN_ABSORBER = 1.0e-76
_MAX_OPTRAN_ORDER = 10
_N_OPTRAN_PREDICTORS = 6

_N_PREDICTORS = {
    1: np.array([7, 18, 7, 11, 11, 14, 10, 11], dtype=np.int64),
    2: np.array([7, 15, 7, 11, 10], dtype=np.int64),
}
_ABSORBER_IDS = {1: (1, 3, 2, 4, 5, 6), 2: (1, 3, 2)}
_H2O, _O3, _CO2 = 0, 1, 2
_COMP_DRY, _COMP_WLO, _COMP_WCO, _COMP_OZO, _COMP_CO2 = 0, 1, 2, 3, 4


@dataclass
class OdpsCoeff:
    """One sensor's ODPS TauCoeff."""

    sensor_id: str
    group: int
    sensor_channel: np.ndarray
    component_id: np.ndarray
    absorber_id: np.ndarray
    ref_level_pressure: np.ndarray  # hPa, length n_layers+1, index 0 is the top
    ref_pressure: np.ndarray
    ref_temperature: np.ndarray
    ref_absorber: np.ndarray  # (n_layers, n_absorbers)
    min_absorber: np.ndarray
    max_absorber: np.ndarray
    n_predictors: np.ndarray  # (n_components, n_channels)
    pos_index: np.ndarray
    coefficients: np.ndarray  # single-precision values, stored float64
    o_significance: np.ndarray | None
    order: np.ndarray | None
    op_index: np.ndarray | None  # (7, n_channels), row 0 is the predictor count
    opos_index: np.ndarray | None
    oc: np.ndarray | None
    alpha: float
    alpha_c1: float
    alpha_c2: float
    o_component_index: int

    @property
    def n_layers(self) -> int:
        return int(self.ref_pressure.shape[0])

    def channel_index(self, band: int) -> int:
        hits = np.flatnonzero(self.sensor_channel == int(band))
        if hits.size != 1:
            raise KeyError(f"{self.sensor_id} has no ODPS channel {band}")
        return int(hits[0])


def load_odps_file(path) -> OdpsCoeff:
    """Read an ODPS ``TauCoeff.bin``."""
    ff = FortranFile(path)
    ff.read_i32s((2,))  # release, version
    ff.read_i32()  # algorithm
    dims = ff.read_i32s((7,))
    n_layers, n_comp, n_abs, n_ch, n_coeff, _n_op, n_oc = [int(v) for v in dims]
    group = int(ff.read_i32())
    sensor_rec = ff.read_record()
    sensor_id = sensor_rec[:20].decode("ascii").strip()
    channels = ff.read_i32s((n_ch,))
    component_id = ff.read_i32s((n_comp,))
    absorber_id = ff.read_i32s((n_abs,))
    n_level = n_layers + 1
    n_prof = n_level + 2 * n_layers + 3 * n_layers * n_abs
    ref = ff.read_f64((n_prof,))
    cursor = 0

    def take(n):
        nonlocal cursor
        piece = ref[cursor : cursor + n]
        cursor += n
        return piece

    ref_level = take(n_level)
    ref_p = take(n_layers)
    ref_t = take(n_layers)
    ref_a = take(n_layers * n_abs).reshape((n_layers, n_abs), order="F")
    min_a = take(n_layers * n_abs).reshape((n_layers, n_abs), order="F")
    max_a = take(n_layers * n_abs).reshape((n_layers, n_abs), order="F")
    pred = ff.read_i32s((2 * n_comp * n_ch,))
    n_predictors = pred[: n_comp * n_ch].reshape((n_comp, n_ch), order="F")
    pos_index = pred[n_comp * n_ch :].reshape((n_comp, n_ch), order="F")
    coefficients = ff.read_f32((n_coeff,)) if n_coeff else np.zeros(0)
    o_sig = order = op_index = opos = oc = None
    alpha = alpha_c1 = alpha_c2 = 0.0
    o_comp = -1
    if n_oc > 0:
        # OSignificance, Order, OP_Index(0:6, L), OPos_Index, OC, alphas, component
        blob = ff.read_record()
        n_op_rows = _N_OPTRAN_PREDICTORS + 1
        n_int = n_ch + n_ch + n_op_rows * n_ch + n_ch
        ints = np.frombuffer(blob[: n_int * 4], dtype=np.dtype(ff.endian + "i4")).astype(np.int64)
        n_double = n_oc + 3
        double_end = n_int * 4 + n_double * 8
        rest = np.frombuffer(blob[n_int * 4 : double_end], dtype=np.dtype(ff.endian + "f8"))
        o_sig = ints[:n_ch].copy()
        order = ints[n_ch : 2 * n_ch].copy()
        op_index = ints[2 * n_ch : 2 * n_ch + n_op_rows * n_ch].reshape((n_op_rows, n_ch), order="F")
        opos = ints[2 * n_ch + n_op_rows * n_ch :].copy()
        oc = rest[:n_oc].astype(np.float64).copy()
        alpha, alpha_c1, alpha_c2 = [float(v) for v in rest[n_oc : n_oc + 3]]
        o_comp = int(np.frombuffer(blob[double_end:], dtype=np.dtype(ff.endian + "i4"))[0])
    expected = _ABSORBER_IDS.get(group)
    if expected is None:
        raise ValueError(f"ODPS group {group} is not implemented")
    if tuple(int(v) for v in absorber_id) != expected:
        raise ValueError(f"absorber ids {absorber_id.tolist()} are not group {group}")
    return OdpsCoeff(
        sensor_id=sensor_id,
        group=group,
        sensor_channel=channels,
        component_id=component_id,
        absorber_id=absorber_id,
        ref_level_pressure=ref_level,
        ref_pressure=ref_p,
        ref_temperature=ref_t,
        ref_absorber=ref_a,
        min_absorber=min_a,
        max_absorber=max_a,
        n_predictors=n_predictors,
        pos_index=pos_index,
        coefficients=coefficients,
        o_significance=o_sig,
        order=order,
        op_index=op_index,
        opos_index=opos,
        oc=oc,
        alpha=alpha,
        alpha_c1=alpha_c1,
        alpha_c2=alpha_c2,
        o_component_index=o_comp,
    )


def layer_average(px1: np.ndarray, px2: np.ndarray):
    """``LayerAvg`` weights from user ``ln p`` (``px2``) onto ODPS ``ln p`` (``px1``).

    Returns
    -------
    weights : ndarray, shape (n_user, n_odps)
    index : ndarray, shape (2, n_odps)
        Inclusive 0-based source bounds.
    """
    px1 = np.asarray(px1, dtype=np.float64)
    px2 = np.asarray(px2, dtype=np.float64)
    kn1 = px1.size
    kn2 = px2.size
    pz = np.zeros((kn2, kn1), dtype=np.float64)
    index = np.zeros((2, kn1), dtype=np.int64)
    istart = 0
    for ki in range(kn1):
        z2 = px1[ki]
        z1 = 2.0 * z2 - px1[ki + 1] if ki == 0 else px1[ki - 1]
        z3 = 2.0 * z2 - z1 if ki == kn1 - 1 else px1[ki + 1]
        if z3 > px2[kn2 - 1]:
            z3 = px2[kn2 - 1]
        iskip = 0
        if z2 >= px2[kn2 - 1]:
            z3 = px2[kn2 - 1]
            z2 = px2[kn2 - 1]
            iskip = 1
        ic = 0
        j_unmatched = kn2 - 1
        for j in range(istart, kn2 - 1):
            if px2[j] > z3:
                # Fortran ``go to 1000`` keeps the current j. That index is
                # 1-based there, so the 0-based slot is this j.
                j_unmatched = j
                break
            if px2[j] <= z2 and px2[j + 1] > z1:
                itop = 0
                ibot = 0
                if z1 < z3:
                    y1 = z1
                    if px2[j] > z1:
                        y1 = px2[j]
                        itop = 1
                    y2 = z2
                    if px2[j + 1] < z2:
                        y2 = px2[j + 1]
                        ibot = 1
                else:
                    y1 = z2
                    if px2[j] > z2:
                        y1 = px2[j]
                        itop = 1
                    y2 = z1
                    if px2[j + 1] < z1:
                        y2 = px2[j + 1]
                        ibot = 1
                dy = y2 - y1
                dz = z1 - z2
                if abs(dz) < _SMALLDIFF:
                    raise ValueError("LayerAvg dz <= 0")
                dzd = 1.0 / dz
                zw1 = (z1 - y1) * dzd * dy
                zw2 = (z1 - y2) * dzd * dy
                dx = px2[j + 1] - px2[j]
                if abs(dx) < _SMALLDIFF:
                    raise ValueError("LayerAvg dx <= 0")
                d = (px2[j + 1] - z2) / dx
                if z1 < z3 and ibot == 0:
                    zw1 = zw1 + zw2 * d
                    zw2 = zw2 * (1.0 - d)
                elif z1 > z3 and itop == 0:
                    zw2 = zw2 + zw1 * (1.0 - d)
                    zw1 = zw1 * d
                pz[j, ki] += zw1
                pz[j + 1, ki] += zw2
                ic = 1
            if px2[j] < z3 and px2[j + 1] >= z2 and iskip == 0:
                itop = 0
                ibot = 0
                if z3 < z1:
                    y1 = z3
                    if px2[j] > z3:
                        y1 = px2[j]
                        itop = 1
                    y2 = z2
                    if px2[j + 1] < z2:
                        y2 = px2[j + 1]
                        ibot = 1
                else:
                    y1 = z2
                    if px2[j] > z2:
                        y1 = px2[j]
                        itop = 1
                    y2 = z3
                    if px2[j + 1] < z3:
                        y2 = px2[j + 1]
                        ibot = 1
                dy = y2 - y1
                dz = z3 - z2
                if dz == 0.0:
                    dz = 1.0
                if abs(dz) < _SMALLDIFF:
                    raise ValueError("LayerAvg dz <= 0")
                dzd = 1.0 / dz
                zw1 = (z3 - y1) * dzd * dy
                zw2 = (z3 - y2) * dzd * dy
                dx = px2[j + 1] - px2[j]
                if abs(dx) < _SMALLDIFF:
                    raise ValueError("LayerAvg dx <= 0")
                d = (px2[j + 1] - z2) / dx
                if z3 < z1 and ibot == 0:
                    zw1 = zw1 + zw2 * d
                    zw2 = zw2 * (1.0 - d)
                elif z3 > z1 and itop == 0:
                    zw2 = zw2 + zw1 * (1.0 - d)
                    zw1 = zw1 * d
                pz[j, ki] += zw1
                pz[j + 1, ki] += zw2
                ic = 1
        if ic == 0:
            pz[j_unmatched, ki] = 1.0
        nonzero = np.flatnonzero(pz[:, ki] != 0.0)
        if nonzero.size == 0:
            index[0, ki] = kn2 - 1
            index[1, ki] = kn2 - 1
        else:
            index[0, ki] = int(nonzero[0])
            index[1, ki] = int(nonzero[-1])
        istart = int(index[0, ki])
        sl = slice(index[0, ki], index[1, ki] + 1)
        total = pz[sl, ki].sum()
        if total != 0.0:
            pz[sl, ki] /= total
    return pz, index


def _interp_index(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    """``Compute_Interp_Index``. ``x`` and ``u`` ascend. Returns shape (2, nu)."""
    nx = x.size
    nu = u.size
    out = np.zeros((2, nu), dtype=np.int64)
    k1 = nu
    for iu in range(nu):
        if u[iu] < x[0]:
            out[:, iu] = 0
        else:
            k1 = iu
            break
    k2 = -1
    for iu in range(nu - 1, k1 - 1, -1):
        if u[iu] > x[nx - 1]:
            out[:, iu] = nx - 1
        else:
            k2 = iu
            break
    j = 0
    for iu in range(k1, k2 + 1):
        for ix in range(j, nx - 1):
            if x[ix] <= u[iu] <= x[ix + 1]:
                out[0, iu] = ix
                out[1, iu] = ix + 1
                j = ix
                break
    return out


def _interp_profile(index: np.ndarray, y: np.ndarray, x: np.ndarray, u: np.ndarray) -> np.ndarray:
    out = np.empty(u.shape, dtype=np.float64)
    for i in range(u.size):
        k1 = int(index[0, i])
        k2 = int(index[1, i])
        if k1 == k2:
            out[i] = y[k1]
        else:
            out[i] = y[k1] + (y[k2] - y[k1]) * (u[i] - x[k1]) / (x[k2] - x[k1])
    return out


def _geopotential_km(level_pressure, temperature, water_vapor):
    n_layers = temperature.size
    height = np.zeros(n_layers + 1, dtype=np.float64)
    for k in range(n_layers - 1, -1, -1):
        tv = temperature[k] * (1.0 + _C_TV * water_vapor[k])
        height[k] = height[k + 1] + _CC_KM * tv * np.log(level_pressure[k + 1] / level_pressure[k])
    return height


def _surface_secant_profile(coeff: OdpsCoeff, user_surface_hpa, zenith_deg, surface_altitude_km):
    """Height-dependent secant from ``Map_Input``, reference temperature and water."""
    z = _geopotential_km(coeff.ref_level_pressure, coeff.ref_temperature, coeff.ref_absorber[:, _H2O])
    n = coeff.n_layers
    ref_ln = np.log(coeff.ref_level_pressure)
    if coeff.ref_level_pressure[-1] >= user_surface_hpa:
        sfc_idx = n  # Fortran index of the layer whose bottom is still above the user surface
        fraction = 0.0
        user_ln = np.log(user_surface_hpa)
        for k in range(n, -1, -1):
            if coeff.ref_level_pressure[k] < user_surface_hpa:
                sfc_idx = k + 1
                fraction = (ref_ln[sfc_idx] - user_ln) / (ref_ln[sfc_idx] - ref_ln[k])
                break
        z_sfc = z[sfc_idx] + fraction * (z[sfc_idx - 1] - z[sfc_idx])
        z = z - z_sfc + surface_altitude_km
    else:
        z = z + _CC_KM * coeff.ref_temperature[-1] * np.log(1013.0 / coeff.ref_level_pressure[-1])
    sine = (_EARTH_RADIUS_KM + surface_altitude_km) * np.sin(np.deg2rad(zenith_deg))
    secant = np.empty(n, dtype=np.float64)
    for k in range(n):
        s = sine / (_EARTH_RADIUS_KM + z[k + 1])
        s = min(max(float(s), -0.999999), 0.999999)
        secant[k] = 1.0 / np.sqrt(1.0 - s * s)
    return secant


def _predictors(group, temperature, absorber, ref_level_p, ref_t, ref_a, secant):
    n_layers = temperature.size
    n_abs = absorber.shape[1]
    n_comp = _N_PREDICTORS[group].size
    max_pred = int(_N_PREDICTORS[group].max())
    x = np.zeros((n_layers, max_pred, n_comp), dtype=np.float64)
    tz = np.empty(n_layers)
    tzp = np.empty(n_layers)
    gazp = np.empty((n_layers, n_abs))
    gatzp = np.empty((n_layers, n_abs))
    tz_ref = tz_sum = tzp_ref = tzp_sum = 0.0
    gaz_ref = np.zeros(n_abs)
    gaz_sum = np.zeros(n_abs)
    gazp_ref = np.zeros(n_abs)
    gazp_sum = np.zeros(n_abs)
    gatzp_ref = np.zeros(n_abs)
    gatzp_sum = np.zeros(n_abs)
    for k in range(n_layers):
        # Fortran layer k>=2 weights by the lower interface, Ref_Level_Pressure(k).
        if k == 0:
            pdp = ref_level_p[0] * (ref_level_p[1] - ref_level_p[0])
        else:
            pdp = ref_level_p[k + 1] * (ref_level_p[k + 1] - ref_level_p[k])
        tz_ref += ref_t[k]
        tz_sum += temperature[k]
        tz[k] = tz_sum / tz_ref
        tzp_ref += pdp * ref_t[k]
        tzp_sum += pdp * temperature[k]
        tzp[k] = tzp_sum / tzp_ref
        for j in range(n_abs):
            gaz_ref[j] += ref_a[k, j]
            gaz_sum[j] += absorber[k, j]
            gazp_ref[j] += pdp * ref_a[k, j]
            gazp_sum[j] += pdp * absorber[k, j]
            gazp[k, j] = gazp_sum[j] / gazp_ref[j]
            gatzp_ref[j] += pdp * ref_a[k, j] * ref_t[k]
            gatzp_sum[j] += pdp * absorber[k, j] * temperature[k]
            gatzp[k, j] = gatzp_sum[j] / gatzp_ref[j]
    for k in range(n_layers):
        dt = temperature[k] - ref_t[k]
        t = temperature[k] / ref_t[k]
        h2o = absorber[k, _H2O] / ref_a[k, _H2O]
        o3 = absorber[k, _O3] / ref_a[k, _O3]
        co2 = absorber[k, _CO2] / ref_a[k, _CO2]
        t2 = t * t
        dt2 = dt * abs(dt)
        h2o_a = secant[k] * h2o
        h2o_r = np.sqrt(h2o_a)
        h2o_s = h2o_a * h2o_a
        h2o_r4 = np.sqrt(h2o_r)
        h2o_ratio = h2o / gatzp[k, _H2O]
        o3_a = secant[k] * o3
        o3_r = np.sqrt(o3_a)
        dry = x[k, :, _COMP_DRY]
        dry[0] = secant[k]
        dry[1] = secant[k] * t
        dry[2] = secant[k] * t2
        dry[3] = t
        dry[4] = secant[k] * secant[k]
        dry[5] = t2
        dry[6] = tz[k]
        wco = x[k, :, _COMP_WCO]
        wco[0] = h2o_a / t
        wco[1] = h2o_a / t * h2o
        wco[2] = h2o_a / t2 * h2o / t2
        wco[3] = h2o_a / t2
        wco[4] = h2o_a / t2 * h2o
        wco[5] = h2o_a / t2**2
        wco[6] = h2o_a
        ozo = x[k, :, _COMP_OZO]
        ozo[0] = o3_a
        ozo[1] = o3_a * dt
        ozo[2] = o3_a * o3 * gazp[k, _O3]
        ozo[3] = o3_a * o3_a
        ozo[4] = o3_a * gazp[k, _O3]
        ozo[5] = o3_a * np.sqrt(secant[k] * gazp[k, _O3])
        ozo[6] = o3_r * dt
        ozo[7] = o3_r
        ozo[8] = o3_r * o3 / gazp[k, _O3]
        ozo[9] = secant[k] * gazp[k, _O3]
        ozo[10] = (secant[k] * gazp[k, _O3]) ** 2
        co2p = x[k, :, _COMP_CO2]
        co2p[0] = secant[k] * t
        co2p[1] = secant[k] * t2
        co2p[2] = t
        co2p[3] = t2
        co2p[4] = secant[k]
        co2p[5] = secant[k] * co2
        co2p[6] = secant[k] * tzp[k]
        co2p[7] = (secant[k] * gazp[k, _CO2]) ** 2
        co2p[8] = tzp[k] ** 3
        co2p[9] = secant[k] * tzp[k] * np.sqrt(t)
        wlo = x[k, :, _COMP_WLO]
        wlo[0] = h2o_a
        wlo[1] = h2o_a * dt
        wlo[2] = h2o_s
        wlo[3] = h2o_a * dt2
        wlo[4] = h2o_r4
        wlo[5] = h2o_s * h2o_a
        wlo[6] = h2o_r
        wlo[7] = h2o_r * dt
        wlo[8] = h2o_s * h2o_s
        wlo[9] = h2o_ratio
        wlo[10] = h2o_r * h2o_ratio
        wlo[11] = (secant[k] * gazp[k, _H2O]) ** 2
        wlo[12] = secant[k] * gazp[k, _H2O]
        wlo[13] = secant[k]
        wlo[14] = secant[k] * co2
        if group == 1:
            raise NotImplementedError("ODPS group 1 trace-gas predictors are not used by ABI G16")
    return x


def _optran_predictors(temperature, vapor, level_p, pressure, secant, alpha, alpha_c1, alpha_c2):
    n_layers = temperature.size
    ox = np.zeros((n_layers, 14), dtype=np.float64)
    ap = np.zeros((n_layers, _MAX_OPTRAN_ORDER), dtype=np.float64)
    da = np.zeros(n_layers, dtype=np.float64)
    for k in range(n_layers):
        t2 = temperature[k] * temperature[k]
        p2 = pressure[k] * pressure[k]
        ox[k, 0] = temperature[k]
        ox[k, 1] = pressure[k]
        ox[k, 2] = t2
        ox[k, 3] = p2
        ox[k, 4] = temperature[k] * pressure[k]
        ox[k, 5] = t2 * pressure[k]
        ox[k, 6] = temperature[k] * p2
        ox[k, 7] = t2 * p2
        ox[k, 8] = pressure[k] ** 0.25
        ox[k, 9] = vapor[k]
        ox[k, 10] = vapor[k] / t2
        ox[k, 13] = secant[k]
    int_vapor = 0.0
    s_t = 0.0
    s_p = 0.0
    xl_t_prev = 0.0
    xl_p_prev = 0.0
    for k in range(n_layers):
        dpon_g = _RECIPROCAL_GRAVITY * (level_p[k + 1] - level_p[k])
        d_abs = dpon_g * vapor[k] * secant[k]
        new_vapor = int_vapor + d_abs
        ave = 0.5 * (int_vapor + new_vapor)
        da[k] = d_abs
        s_t += temperature[k] * d_abs
        s_p += pressure[k] * d_abs
        inverse = 1.0 / new_vapor if new_vapor > _MIN_ABSORBER else 0.0
        xl_t = 0.5 * s_t * inverse
        xl_p = 0.5 * s_p * inverse
        ox[k, 11] = xl_t + xl_t_prev
        ox[k, 12] = xl_p + xl_p_prev
        ap1 = np.log((ave - alpha_c2) / alpha_c1) / alpha
        ap[k, 0] = ap1
        for i in range(1, _MAX_OPTRAN_ORDER):
            ap[k, i] = ap[k, i - 1] * ap1
        int_vapor = new_vapor
        xl_t_prev = xl_t
        xl_p_prev = xl_p
    return ox, ap, da


def _optran_od(coeff: OdpsCoeff, channel: int, ox, ap, da) -> np.ndarray:
    assert coeff.op_index is not None and coeff.oc is not None
    assert coeff.order is not None and coeff.opos_index is not None
    np_ = int(coeff.op_index[0, channel])
    od = np.zeros(coeff.n_layers, dtype=np.float64)
    if np_ <= 0:
        return od
    n_orders = int(coeff.order[channel])
    js = int(coeff.opos_index[channel])  # 1-based
    b = np.zeros((coeff.n_layers, np_ + 1), dtype=np.float64)
    for i in range(np_ + 1):
        jj = js + i * (n_orders + 1)
        poly = coeff.oc[jj - 1 : jj - 1 + n_orders + 1]
        acc = np.full(coeff.n_layers, poly[0])
        for j in range(1, n_orders + 1):
            acc = acc + poly[j] * ap[:, j - 1]
        b[:, i] = acc
    ln_chi = b[:, 0].copy()
    for i in range(1, np_ + 1):
        ii = int(coeff.op_index[i, channel])  # 1-based OX index
        ln_chi = ln_chi + b[:, i] * ox[:, ii - 1]
    chi = np.empty_like(ln_chi)
    hi = ln_chi > _LIMIT_EXP
    lo = ln_chi < -_LIMIT_EXP
    mid = ~hi & ~lo
    chi[hi] = _LIMIT_LOG
    chi[lo] = 0.0
    chi[mid] = np.exp(ln_chi[mid])
    return chi * da


def gaseous_nadir_od(coeff: OdpsCoeff, level_pressure_hpa, pressure_hpa, temperature, h2o_gkg, o3_ppmv, zenith_deg, bands):
    """Nadir gas optical depth on the user layers.

    Shapes are one profile: level pressure has ``n_layers + 1`` values,
    top-down. Missing CO2 (and any other absorber the group trains) is the
    reference profile, which is what ``Map_Input`` does.
    """
    level_p = np.asarray(level_pressure_hpa, dtype=np.float64)
    pressure = np.asarray(pressure_hpa, dtype=np.float64)
    temp_u = np.asarray(temperature, dtype=np.float64)
    h2o_u = np.asarray(h2o_gkg, dtype=np.float64)
    o3_u = np.asarray(o3_ppmv, dtype=np.float64)
    n_user = pressure.size
    ref_ln_p = np.log(coeff.ref_pressure)
    user_ln_p = np.log(pressure)
    ref_ln_level = np.log(coeff.ref_level_pressure)
    user_ln_level = np.log(level_p)
    weights, _bounds = layer_average(ref_ln_p, user_ln_p)
    temp = weights.T @ temp_u
    absorber = np.zeros((coeff.n_layers, coeff.ref_absorber.shape[1]), dtype=np.float64)
    user_abs = [_H2O, _O3]
    user_val = [h2o_u, o3_u]
    for j in range(absorber.shape[1]):
        if j in user_abs:
            absorber[:, j] = weights.T @ user_val[user_abs.index(j)]
        else:
            absorber[:, j] = coeff.ref_absorber[:, j]
        absorber[:, j] = np.maximum(absorber[:, j], np.finfo(np.float64).eps)
        absorber[:, j] = np.clip(absorber[:, j], coeff.min_absorber[:, j], coeff.max_absorber[:, j])
    secant = _surface_secant_profile(coeff, float(level_p[-1]), float(zenith_deg), 0.0)
    x = _predictors(coeff.group, temp, absorber, coeff.ref_level_pressure, coeff.ref_temperature, coeff.ref_absorber, secant)
    ox = ap = da = None
    if coeff.oc is not None:
        ox, ap, da = _optran_predictors(
            temp, absorber[:, _H2O], coeff.ref_level_pressure, coeff.ref_pressure, secant,
            coeff.alpha, coeff.alpha_c1, coeff.alpha_c2,
        )
    surface_secant = 1.0 / np.cos(np.deg2rad(float(zenith_deg)))
    out = np.zeros((n_user, len(bands)), dtype=np.float64)
    for ic, band in enumerate(bands):
        ch = coeff.channel_index(int(band))
        od = np.zeros(coeff.n_layers, dtype=np.float64)
        for j in range(coeff.n_predictors.shape[0]):
            use_optran = (
                coeff.o_significance is not None
                and j + 1 == coeff.o_component_index
                and int(coeff.o_significance[ch]) == 1
            )
            if use_optran:
                od += _optran_od(coeff, ch, ox, ap, da)
                continue
            np_ = int(coeff.n_predictors[j, ch])
            if np_ <= 0:
                continue
            j0 = int(coeff.pos_index[j, ch])
            for i in range(np_):
                start = j0 - 1 + i * coeff.n_layers
                od += coeff.coefficients[start : start + coeff.n_layers] * x[:, i, j]
        od = np.clip(od, 0.0, _MAX_OD)
        path = np.zeros(coeff.n_layers + 1, dtype=np.float64)
        path[1:] = np.cumsum(od)
        index = _interp_index(ref_ln_level, user_ln_level)
        user_path = _interp_profile(index, path, ref_ln_level, user_ln_level)
        out[:, ic] = np.diff(user_path) / surface_secant
    return out
