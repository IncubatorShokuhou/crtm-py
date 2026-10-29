"""Column-batched CRTM forward path.

Profiles and channels stay in NumPy arrays. Layer sums that CRTM defines
from the top downward use ``cumsum``. The ADA eigenstep uses
``numpy.linalg.eig`` on a stack of the same real nonsymmetric matrices
``ASYMTX`` solves one at a time. Those eigenvalues are real; the solver
returns them unordered and with its own vector scaling. ``eig`` matches
the values to about 1e-12 once both results are sorted, and the vectors
match up to scale. The adding algebra is unchanged if a column is scaled
or the columns are reordered together with their eigenvalues, which is
checked against the Fortran brightness temperatures.

Cloud lookup and that eigenstep run only where there is cloud. Layers
with water content at or below the CRTM threshold are left out of the
``CloudCoeff`` interpolation. Cells whose single-scatter albedo is at or
below the ADA threshold take the diagonal emission operator. Clear
columns never enter ``_ada_cloudy``.
"""

from __future__ import annotations

import numpy as np

from crtm_py.cloudlut import _DENSITY_INDEX, _LEGENDRE_OFFSET, _WATER_CONTENT_THRESHOLD, n_streams_for_radius
from crtm_py.constants import EMIS_ICE, EMIS_SNOW
from crtm_py.lagrange import find_random_index, interp_1d, lagrange_weights
from crtm_py.odps import (
    _CC_KM,
    _CO2,
    _COMP_CO2,
    _COMP_DRY,
    _COMP_OZO,
    _COMP_WCO,
    _COMP_WLO,
    _C_TV,
    _EARTH_RADIUS_KM,
    _H2O,
    _LIMIT_EXP,
    _LIMIT_LOG,
    _MAX_OD,
    _MAX_OPTRAN_ORDER,
    _MIN_ABSORBER,
    _O3,
    _geopotential_km,
    _interp_index,
    layer_average,
)
from crtm_py.toa import prepend_standard_atmosphere_batch

_PHASE_FLOOR = 1.0e-7
_DELTA_OD = 1.0e-8
_MAX_ALBEDO = 0.999999
_ALBEDO_THRESHOLD = 1.0e-10
_ANGLE_MATCH = 1.0e-4
# Cap one ``eig`` call. A chunk of cloudy layers is still one batch of matrices.
_EIG_BATCH = 8192
_PACK: dict[tuple[int, tuple[int, ...]], dict] = {}


def _node_weights(nodes, xi):
    """Four ``Interp_1D`` weights for each query, last axis length 4."""
    w_left, w_right, lp_left, lp_right = lagrange_weights(nodes, xi) if xi.ndim == 0 else _weights_vec(nodes, xi)
    if xi.ndim == 0:
        w = np.zeros(4, dtype=np.float64)
        w[0] = w_left * lp_left[0]
        w[1] = w_left * lp_left[1] + w_right * lp_right[0]
        w[2] = w_left * lp_left[2] + w_right * lp_right[1]
        w[3] = w_right * lp_right[2]
        return w
    return _weights_vec(nodes, xi)


def _weights_vec(nodes, xi):
    """``nodes`` (..., 4), ``xi`` (...,)."""
    x = nodes
    dxi_l = xi[..., None] - x[..., :3]
    dx1 = x[..., 0] - x[..., 1]
    dx2 = x[..., 0] - x[..., 2]
    dx3 = x[..., 1] - x[..., 2]
    lp_l = np.stack(
        [
            dxi_l[..., 1] * dxi_l[..., 2] / (dx1 * dx2),
            dxi_l[..., 0] * dxi_l[..., 2] / (-dx1 * dx3),
            dxi_l[..., 0] * dxi_l[..., 1] / (dx2 * dx3),
        ],
        axis=-1,
    )
    xr = x[..., 1:]
    dxi_r = xi[..., None] - xr[..., :3]
    rdx1 = xr[..., 0] - xr[..., 1]
    rdx2 = xr[..., 0] - xr[..., 2]
    rdx3 = xr[..., 1] - xr[..., 2]
    lp_r = np.stack(
        [
            dxi_r[..., 1] * dxi_r[..., 2] / (rdx1 * rdx2),
            dxi_r[..., 0] * dxi_r[..., 2] / (-rdx1 * rdx3),
            dxi_r[..., 0] * dxi_r[..., 1] / (rdx2 * rdx3),
        ],
        axis=-1,
    )
    w_right = np.where(xi < x[..., 1], 0.0, np.where(xi > x[..., 2], 1.0, dxi_l[..., 1] / (-dx3)))
    w_left = 1.0 - w_right
    w = np.empty(xi.shape + (4,), dtype=np.float64)
    w[..., 0] = w_left * lp_l[..., 0]
    w[..., 1] = w_left * lp_l[..., 1] + w_right * lp_r[..., 0]
    w[..., 2] = w_left * lp_l[..., 2] + w_right * lp_r[..., 1]
    w[..., 3] = w_right * lp_r[..., 2]
    return w


def _stencil_index(grid, value):
    value = np.clip(np.asarray(value, dtype=np.float64), grid[0], grid[-1])
    k0 = np.searchsorted(grid, value, side="left")
    k0 = np.clip(k0, 0, grid.size - 1)
    i1 = np.minimum(np.maximum(1, k0 + 1 - 2), grid.size - 3) - 1
    return i1.astype(np.int64), value


def _gather_points(table, freq_i, rad_i, freq_w, rad_w):
    """Four-point mix on a flat list of radii.

    ``table`` is (n_freq, n_reff). ``freq_i`` is (C,), ``rad_i`` is (M,).
    The result is (C, M).
    """
    off = np.arange(4)
    fi = freq_i[:, None] + off
    ri = rad_i[:, None] + off
    vals = table[fi[:, None, :, None], ri[None, :, None, :]]
    return np.einsum("ca,mb,cmab->cm", freq_w, rad_w, vals, optimize=True)


def _pack_poly(coeff, channel_ids):
    key = (id(coeff), tuple(int(c) for c in channel_ids))
    cached = _PACK.get(key)
    if cached is not None:
        return cached
    n_layer = coeff.n_layers
    n_ch = len(channel_ids)
    n_comp = coeff.n_predictors.shape[0]
    blocks = []
    counts = []
    for comp in range(n_comp):
        counts_c = coeff.n_predictors[comp, channel_ids]
        n_pred = int(counts_c.max()) if counts_c.size else 0
        block = np.zeros((n_ch, max(n_pred, 1), n_layer), dtype=np.float64)
        for ic, ch in enumerate(channel_ids):
            np_ = int(coeff.n_predictors[comp, ch])
            if np_ <= 0:
                continue
            j0 = int(coeff.pos_index[comp, ch])
            for i in range(np_):
                start = j0 - 1 + i * n_layer
                block[ic, i] = coeff.coefficients[start : start + n_layer]
        blocks.append(block)
        counts.append(counts_c.copy())
    _PACK[key] = {"blocks": blocks, "counts": counts, "channels": np.asarray(channel_ids)}
    return _PACK[key]


def _pdp(ref_level_p):
    pdp = np.empty(ref_level_p.size - 1, dtype=np.float64)
    pdp[0] = ref_level_p[0] * (ref_level_p[1] - ref_level_p[0])
    pdp[1:] = ref_level_p[2:] * (ref_level_p[2:] - ref_level_p[1:-1])
    return pdp


def _predictors_batch(temperature, absorber, ref_level_p, ref_t, ref_a, secant):
    """Predictors for group 2. ``temperature`` is (n, layer)."""
    n_layer = temperature.shape[-1]
    pdp = _pdp(ref_level_p)
    tz = np.cumsum(temperature, axis=-1) / np.cumsum(ref_t)
    tzp = np.cumsum(pdp * temperature, axis=-1) / np.cumsum(pdp * ref_t)
    pdp_a = pdp.reshape((1, n_layer, 1))
    gazp = np.cumsum(pdp_a * absorber, axis=1) / np.cumsum(pdp[:, None] * ref_a, axis=0)
    gatzp = np.cumsum(pdp_a * absorber * temperature[..., None], axis=1) / np.cumsum(
        (pdp * ref_t)[:, None] * ref_a, axis=0
    )
    secant = np.broadcast_to(secant, temperature.shape)
    dt = temperature - ref_t
    t = temperature / ref_t
    h2o = absorber[..., _H2O] / ref_a[:, _H2O]
    o3 = absorber[..., _O3] / ref_a[:, _O3]
    co2 = absorber[..., _CO2] / ref_a[:, _CO2]
    t2 = t * t
    dt2 = dt * np.abs(dt)
    h2o_a = secant * h2o
    h2o_r = np.sqrt(h2o_a)
    h2o_s = h2o_a * h2o_a
    h2o_r4 = np.sqrt(h2o_r)
    h2o_ratio = h2o / gatzp[..., _H2O]
    o3_a = secant * o3
    o3_r = np.sqrt(o3_a)
    x = np.zeros(temperature.shape + (15, 5), dtype=np.float64)
    dry = x[..., _COMP_DRY]
    dry[..., 0] = secant
    dry[..., 1] = secant * t
    dry[..., 2] = secant * t2
    dry[..., 3] = t
    dry[..., 4] = secant * secant
    dry[..., 5] = t2
    dry[..., 6] = tz
    wco = x[..., _COMP_WCO]
    wco[..., 0] = h2o_a / t
    wco[..., 1] = h2o_a / t * h2o
    wco[..., 2] = h2o_a / t2 * h2o / t2
    wco[..., 3] = h2o_a / t2
    wco[..., 4] = h2o_a / t2 * h2o
    wco[..., 5] = h2o_a / t2**2
    wco[..., 6] = h2o_a
    ozo = x[..., _COMP_OZO]
    ozo[..., 0] = o3_a
    ozo[..., 1] = o3_a * dt
    ozo[..., 2] = o3_a * o3 * gazp[..., _O3]
    ozo[..., 3] = o3_a * o3_a
    ozo[..., 4] = o3_a * gazp[..., _O3]
    ozo[..., 5] = o3_a * np.sqrt(secant * gazp[..., _O3])
    ozo[..., 6] = o3_r * dt
    ozo[..., 7] = o3_r
    ozo[..., 8] = o3_r * o3 / gazp[..., _O3]
    ozo[..., 9] = secant * gazp[..., _O3]
    ozo[..., 10] = (secant * gazp[..., _O3]) ** 2
    co2p = x[..., _COMP_CO2]
    co2p[..., 0] = secant * t
    co2p[..., 1] = secant * t2
    co2p[..., 2] = t
    co2p[..., 3] = t2
    co2p[..., 4] = secant
    co2p[..., 5] = secant * co2
    co2p[..., 6] = secant * tzp
    co2p[..., 7] = (secant * gazp[..., _CO2]) ** 2
    co2p[..., 8] = tzp**3
    co2p[..., 9] = secant * tzp * np.sqrt(t)
    wlo = x[..., _COMP_WLO]
    wlo[..., 0] = h2o_a
    wlo[..., 1] = h2o_a * dt
    wlo[..., 2] = h2o_s
    wlo[..., 3] = h2o_a * dt2
    wlo[..., 4] = h2o_r4
    wlo[..., 5] = h2o_s * h2o_a
    wlo[..., 6] = h2o_r
    wlo[..., 7] = h2o_r * dt
    wlo[..., 8] = h2o_s * h2o_s
    wlo[..., 9] = h2o_ratio
    wlo[..., 10] = h2o_r * h2o_ratio
    wlo[..., 11] = (secant * gazp[..., _H2O]) ** 2
    wlo[..., 12] = secant * gazp[..., _H2O]
    wlo[..., 13] = secant
    wlo[..., 14] = secant * co2
    return x


def _optran_state(temperature, vapor, level_p, pressure, secant, alpha, alpha_c1, alpha_c2):
    from crtm_py.odps import _RECIPROCAL_GRAVITY

    dpon = _RECIPROCAL_GRAVITY * np.diff(level_p)
    d_abs = dpon * vapor * secant
    new_vapor = np.cumsum(d_abs, axis=-1)
    previous = np.concatenate([np.zeros((d_abs.shape[0], 1)), new_vapor[:, :-1]], axis=1)
    ave = 0.5 * (previous + new_vapor)
    s_t = np.cumsum(temperature * d_abs, axis=-1)
    s_p = np.cumsum(pressure * d_abs, axis=-1)
    inverse = np.divide(1.0, new_vapor, out=np.zeros_like(new_vapor), where=new_vapor > _MIN_ABSORBER)
    xl_t = 0.5 * s_t * inverse
    xl_p = 0.5 * s_p * inverse
    xl_t_prev = np.concatenate([np.zeros((d_abs.shape[0], 1)), xl_t[:, :-1]], axis=1)
    xl_p_prev = np.concatenate([np.zeros((d_abs.shape[0], 1)), xl_p[:, :-1]], axis=1)
    t2 = temperature * temperature
    p = np.broadcast_to(pressure, temperature.shape)
    p2 = p * p
    ox = np.empty(temperature.shape + (14,), dtype=np.float64)
    ox[..., 0] = temperature
    ox[..., 1] = p
    ox[..., 2] = t2
    ox[..., 3] = p2
    ox[..., 4] = temperature * p
    ox[..., 5] = t2 * p
    ox[..., 6] = temperature * p2
    ox[..., 7] = t2 * p2
    ox[..., 8] = p**0.25
    ox[..., 9] = vapor
    ox[..., 10] = vapor / t2
    ox[..., 11] = xl_t + xl_t_prev
    ox[..., 12] = xl_p + xl_p_prev
    ox[..., 13] = secant
    ap1 = np.log((ave - alpha_c2) / alpha_c1) / alpha
    ap = np.empty(temperature.shape + (_MAX_OPTRAN_ORDER,), dtype=np.float64)
    ap[..., 0] = ap1
    for i in range(1, _MAX_OPTRAN_ORDER):
        ap[..., i] = ap[..., i - 1] * ap1
    return ox, ap, d_abs


def _optran_od_batch(coeff, channel, ox, ap, da):
    np_ = int(coeff.op_index[0, channel])
    if np_ <= 0:
        return np.zeros(ox.shape[:2], dtype=np.float64)
    n_orders = int(coeff.order[channel])
    js = int(coeff.opos_index[channel])
    b = np.empty(ox.shape[:2] + (np_ + 1,), dtype=np.float64)
    for i in range(np_ + 1):
        jj = js + i * (n_orders + 1)
        poly = coeff.oc[jj - 1 : jj - 1 + n_orders + 1]
        acc = np.full(ox.shape[:2], poly[0], dtype=np.float64)
        for j in range(1, n_orders + 1):
            acc = acc + poly[j] * ap[..., j - 1]
        b[..., i] = acc
    ln_chi = b[..., 0].copy()
    for i in range(1, np_ + 1):
        ii = int(coeff.op_index[i, channel])
        ln_chi = ln_chi + b[..., i] * ox[..., ii - 1]
    chi = np.empty_like(ln_chi)
    hi = ln_chi > _LIMIT_EXP
    lo = ln_chi < -_LIMIT_EXP
    chi[hi] = _LIMIT_LOG
    chi[lo] = 0.0
    mid = ~hi & ~lo
    chi[mid] = np.exp(ln_chi[mid])
    return chi * da


def _secant_batch(coeff, surface_hpa, zenith_deg):
    if not hasattr(coeff, "_ref_height_km"):
        coeff._ref_height_km = _geopotential_km(
            coeff.ref_level_pressure, coeff.ref_temperature, coeff.ref_absorber[:, _H2O]
        )
    z = coeff._ref_height_km.copy()
    n = coeff.n_layers
    ref_ln = np.log(coeff.ref_level_pressure)
    surface = float(surface_hpa)
    if coeff.ref_level_pressure[-1] >= surface:
        user_ln = np.log(surface)
        sfc_idx = n
        fraction = 0.0
        for k in range(n, -1, -1):
            if coeff.ref_level_pressure[k] < surface:
                sfc_idx = k + 1
                fraction = (ref_ln[sfc_idx] - user_ln) / (ref_ln[sfc_idx] - ref_ln[k])
                break
        z = z - (z[sfc_idx] + fraction * (z[sfc_idx - 1] - z[sfc_idx]))
    else:
        z = z + _CC_KM * coeff.ref_temperature[-1] * np.log(1013.0 / coeff.ref_level_pressure[-1])
    sine = _EARTH_RADIUS_KM * np.sin(np.deg2rad(zenith_deg))
    s = sine[:, None] / (_EARTH_RADIUS_KM + z[1:])
    s = np.clip(s, -0.999999, 0.999999)
    return 1.0 / np.sqrt(1.0 - s * s)


def gas_optical_depth(coeff, level_p, pressure, temperature, h2o, o3, zenith_deg, bands):
    """Nadir gas optical depth, shape ``(n_profile, n_channel, n_layer)``."""
    channel_ids = [coeff.channel_index(int(b)) for b in bands]
    if not np.allclose(pressure, pressure[0]) or not np.allclose(level_p, level_p[0]):
        parts = [
            gas_optical_depth(
                coeff,
                level_p[i : i + 1],
                pressure[i : i + 1],
                temperature[i : i + 1],
                h2o[i : i + 1],
                o3[i : i + 1],
                zenith_deg[i : i + 1],
                bands,
            )
            for i in range(pressure.shape[0])
        ]
        return np.concatenate(parts, axis=0)
    weights, _bounds = layer_average(np.log(coeff.ref_pressure), np.log(pressure[0]))
    temp = np.einsum("uo,nu->no", weights, temperature, optimize=True)
    absorber = np.empty(temp.shape + (coeff.ref_absorber.shape[1],), dtype=np.float64)
    user = {_H2O: h2o, _O3: o3}
    for j in range(absorber.shape[-1]):
        if j in user:
            column = np.einsum("uo,nu->no", weights, user[j], optimize=True)
        else:
            column = np.broadcast_to(coeff.ref_absorber[:, j], temp.shape).copy()
        column = np.maximum(column, np.finfo(np.float64).eps)
        absorber[..., j] = np.clip(column, coeff.min_absorber[:, j], coeff.max_absorber[:, j])
    secant = _secant_batch(coeff, level_p[0, -1], zenith_deg)
    predictors = _predictors_batch(
        temp, absorber, coeff.ref_level_pressure, coeff.ref_temperature, coeff.ref_absorber, secant
    )
    ox = ap = da = None
    if coeff.oc is not None:
        ox, ap, da = _optran_state(
            temp,
            absorber[..., _H2O],
            coeff.ref_level_pressure,
            coeff.ref_pressure,
            secant,
            coeff.alpha,
            coeff.alpha_c1,
            coeff.alpha_c2,
        )
    packed = _pack_poly(coeff, channel_ids)
    n_prof = temperature.shape[0]
    od = np.zeros((n_prof, len(bands), coeff.n_layers), dtype=np.float64)
    for comp, block in enumerate(packed["blocks"]):
        optran = np.zeros(len(bands), dtype=bool)
        if coeff.o_significance is not None and comp + 1 == coeff.o_component_index:
            optran = coeff.o_significance[channel_ids] == 1
        if np.any(~optran):
            poly = np.einsum("nlp,cpl->ncl", predictors[..., : block.shape[1], comp], block, optimize=True)
            od[:, ~optran, :] += poly[:, ~optran, :]
        for ic, ch in enumerate(channel_ids):
            if optran[ic]:
                od[:, ic, :] += _optran_od_batch(coeff, ch, ox, ap, da)
    od = np.clip(od, 0.0, _MAX_OD)
    path = np.zeros(od.shape[:2] + (coeff.n_layers + 1,), dtype=np.float64)
    path[..., 1:] = np.cumsum(od, axis=-1)
    ref_ln_level = np.log(coeff.ref_level_pressure)
    user_ln_level = np.log(level_p[0])
    index = _interp_index(ref_ln_level, user_ln_level)
    k1 = index[0]
    k2 = index[1]
    y1 = path[..., k1]
    y2 = path[..., k2]
    denom = ref_ln_level[k2] - ref_ln_level[k1]
    fac = np.divide(user_ln_level - ref_ln_level[k1], denom, out=np.zeros_like(user_ln_level), where=k1 != k2)
    user_path = y1 + (y2 - y1) * fac
    surface_secant = 1.0 / np.cos(np.deg2rad(zenith_deg))
    return np.diff(user_path, axis=-1) / surface_secant[:, None, None]


def _cloud_fields_points(cloud_coeff, wavenumber, radius, cloud_type, n_leg):
    """Mass extinction, albedo and phase at cloudy cells. ``radius`` is (M,).

    Returns ``ke`` and ``w`` shaped (n_channel, M). ``phase`` is
    (n_channel, M, n_leg + 1), or ``None`` when scattering is off.
    """
    den = _DENSITY_INDEX[cloud_type]
    freq = cloud_coeff.frequency_ir
    reff = cloud_coeff.reff_ir
    fi, freq_x = _stencil_index(freq, wavenumber)
    ri, rad_x = _stencil_index(reff, radius)
    freq_w = _weights_vec(freq[fi[:, None] + np.arange(4)], freq_x)
    rad_w = _weights_vec(reff[ri[:, None] + np.arange(4)], rad_x)
    ke = np.maximum(_gather_points(cloud_coeff.ke_ir[:, :, den], fi, ri, freq_w, rad_w), 0.0)
    w = np.clip(_gather_points(cloud_coeff.w_ir[:, :, den], fi, ri, freq_w, rad_w), 0.0, 1.0)
    if n_leg <= 0:
        return ke * (1.0 - w), np.zeros_like(w), None
    offset = _LEGENDRE_OFFSET[n_leg]
    phase = np.zeros(ke.shape + (n_leg + 1,), dtype=np.float64)
    phase[..., 0] = 0.5
    for ell in range(1, n_leg + 1):
        phase[..., ell] = _gather_points(
            cloud_coeff.pcoeff_ir[:, :, den, ell + offset], fi, ri, freq_w, rad_w
        )
    return ke, w, phase


def cloud_optics_batch(cloud_coeff, wavenumber, clouds, scattering):
    """Cloud nadir optical depth and phase.

    ``wavenumber`` is (n_channel,). Each cloud has ``water`` and ``radius``
    shaped (n_profile, n_layer). Returns tau (n, c, layer), ssa, phase.
    """
    n_ch = wavenumber.size
    sample = clouds[0].water
    n_prof, n_layer = sample.shape
    tau = np.zeros((n_prof, n_ch, n_layer), dtype=np.float64)
    bs = np.zeros_like(tau)
    phase_acc = None
    n_leg = 0
    if scattering:
        max_re = np.zeros(n_prof, dtype=np.float64)
        for cloud in clouds:
            wet = cloud.water > _WATER_CONTENT_THRESHOLD
            max_re = np.maximum(max_re, np.max(np.where(wet, cloud.radius, 0.0), axis=1))
        # One stream count per channel, from the largest particle in the chunk.
        n_leg_ch = np.array(
            [n_streams_for_radius(float(max_re.max()), float(wn)) if max_re.max() > 0 else 0 for wn in wavenumber]
        )
        n_leg = int(n_leg_ch.max()) if n_leg_ch.size else 0
        if n_leg:
            phase_acc = np.zeros((n_prof, n_ch, n_layer, n_leg + 1), dtype=np.float64)
    for cloud in clouds:
        wet = cloud.water > _WATER_CONTENT_THRESHOLD
        if not np.any(wet):
            continue
        pi, li = np.nonzero(wet)
        radius = np.maximum(cloud.radius[pi, li], cloud_coeff.reff_ir[0])
        ke, w, phase = _cloud_fields_points(
            cloud_coeff, wavenumber, radius, cloud.cloud_type, n_leg if scattering else 0
        )
        column = cloud.water[pi, li]
        tau[pi, :, li] += (ke * column).T
        if phase is not None:
            scatter = (ke * column) * w
            bs[pi, :, li] += scatter.T
            phase_acc[pi, :, li, 1:] += np.moveaxis(phase[..., 1:] * scatter[..., None], 0, 1)
    ssa = np.zeros_like(tau)
    if phase_acc is not None:
        use = (bs > 1.0e-10) & (tau > 0.0)
        albedo = np.divide(bs, tau, out=np.zeros_like(bs), where=use)
        phase_acc[..., 1:] = np.divide(
            phase_acc[..., 1:], bs[..., None], out=np.zeros_like(phase_acc[..., 1:]), where=use[..., None]
        )
        phase_acc[..., 0] = 0.5
        delta = phase_acc[..., n_leg]
        factor = np.maximum(1.0 - delta * albedo, 1.0e-6)
        tau = np.where(use, factor * tau, tau)
        ssa = np.where(use, np.clip((1.0 - delta) * albedo / factor, 0.0, _MAX_ALBEDO), 0.0)
    return tau, ssa, phase_acc, n_leg


def _planck(spc, channel_ids, temperature):
    c1 = spc.band_c1[channel_ids]
    c2 = spc.band_c2[channel_ids]
    p1 = spc.planck_c1[channel_ids]
    p2 = spc.planck_c2[channel_ids]
    teff = c1[None, :, None] + c2[None, :, None] * temperature[:, None, :]
    return p1[None, :, None] / np.expm1(p2[None, :, None] / teff)


def _emissivity(coeff, channel_ids, zenith, wind, land_type, water_cov, land_cov, ice_cov, snow_cov, user):
    wn = coeff.spc.wavenumber[channel_ids]
    n = zenith.size
    n_ch = wn.size
    parts = np.zeros((n, n_ch, 4), dtype=np.float64)
    if coeff.water is not None:
        parts[..., 0] = _water_emis(coeff.water, wn, zenith, wind)
    else:
        parts[..., 0] = 0.98
    if coeff.land is not None:
        parts[..., 1] = _land_emis(coeff.land, wn, land_type)
    else:
        parts[..., 1] = 0.97
    parts[..., 2] = EMIS_ICE
    parts[..., 3] = EMIS_SNOW
    if user is not None:
        user = np.asarray(user, dtype=np.float64)
        if user.ndim == 0:
            parts[...] = user
        elif user.ndim == 1 and user.size == n:
            parts[...] = user[:, None, None]
        else:
            parts[...] = user
    weights = np.stack([np.maximum(water_cov, 0), np.maximum(land_cov, 0), np.maximum(ice_cov, 0), np.maximum(snow_cov, 0)], axis=1)
    total = weights.sum(axis=1)
    weights = np.where(total[:, None] > 0, weights / np.maximum(total[:, None], 1e-30), 0.0)
    weights[total <= 0, 1] = 1.0
    return np.einsum("ncp,np->nc", parts, weights, optimize=True)


def _bracket(grid, value):
    x = np.asarray(value, dtype=np.float64)
    idx = np.searchsorted(grid, x, side="right") - 1
    idx = np.clip(idx, 0, grid.size - 2)
    span = grid[idx + 1] - grid[idx]
    w2 = (x - grid[idx]) / span
    return idx, 1.0 - w2, w2


def _water_emis(coeff, wavenumber, zenith, wind):
    ia, wa1, wa2 = _bracket(coeff.angle, np.abs(zenith))
    jf, wf1, wf2 = _bracket(coeff.frequency, wavenumber)
    iw, ww1, ww2 = _bracket(coeff.wind_speed, wind)
    table = coeff.emissivity
    out = np.zeros((zenith.size, wavenumber.size), dtype=np.float64)
    for da, wa in ((0, wa1), (1, wa2)):
        for df, wf in ((0, wf1), (1, wf2)):
            for dw, ww in ((0, ww1), (1, ww2)):
                out += wa[:, None] * wf[None, :] * ww[:, None] * table[ia[:, None] + da, jf[None, :] + df, iw[:, None] + dw]
    return np.clip(out, 0.0, 1.0)


def _land_emis(coeff, wavenumber, land_type):
    n_type = coeff.reflectance.shape[1]
    itype = np.clip(land_type.astype(np.int64), 1, n_type) - 1
    fi, freq = _stencil_index(coeff.frequency, wavenumber)
    nodes = coeff.frequency[fi[:, None] + np.arange(4)]
    weights = _weights_vec(nodes, freq)
    out = np.zeros((land_type.size, wavenumber.size), dtype=np.float64)
    off = np.arange(4)
    freq_idx = fi[:, None] + off
    for t in np.unique(itype):
        mask = itype == t
        refl = coeff.reflectance[freq_idx, t]
        value = np.einsum("cf,cf->c", weights, refl, optimize=True)
        out[mask] = 1.0 - value
    return np.clip(out, 0.0, 1.0)


def _emission(tau, planck, skin, emis, cosmic, mu):
    rad = emis * skin
    refl = 1.0 - emis
    mu = mu[:, None]
    for k in range(tau.shape[-1] - 1, -1, -1):
        trans = np.exp(-tau[..., k] / mu)
        src = planck[..., k] * (1.0 - trans)
        rad = src + trans * (refl * src + rad)
        refl = trans * refl * trans
    return rad + refl * cosmic


def _legendre_batch(mu, n_terms):
    p = np.empty(mu.shape[:-1] + (n_terms, mu.shape[-1]), dtype=np.float64)
    p[..., 0, :] = 1.0
    if n_terms == 1:
        return p
    p[..., 1, :] = mu
    for ell in range(1, n_terms - 1):
        p[..., ell + 1, :] = ((2 * ell + 1) * mu * p[..., ell, :] - ell * p[..., ell - 1, :]) / (ell + 1)
    return p


def _normalize_phase_batch(pff, pbb, weights, n_streams):
    n_z = pff.shape[-1]
    sum_fac = np.zeros(pff.shape[:-2] + (n_z + 1,), dtype=np.float64)
    for i in range(n_streams):
        factor = np.sum((pff[..., i, i:] + pbb[..., i, i:]) * weights[i:], axis=-1)
        factor = np.where(factor == 0.0, 1.0, factor)
        scale = (1.0 - sum_fac[..., i]) / factor
        pff[..., i, i:] *= scale[..., None]
        pbb[..., i, i:] *= scale[..., None]
        if i + 1 < n_z:
            pff[..., i + 1 :, i] = pff[..., i, i + 1 :]
            pbb[..., i + 1 :, i] = pbb[..., i, i + 1 :]
            acc = np.sum((pff[..., i + 1, : i + 1] + pbb[..., i + 1, : i + 1]) * weights[: i + 1], axis=-1)
            sum_fac[..., i + 1] = acc
    if n_streams < n_z:
        factor = sum_fac[..., n_z - 1]
        factor = np.where(factor == 0.0, 1.0, factor)
        pff[..., :, n_z - 1] /= factor[..., None]
        pbb[..., :, n_z - 1] /= factor[..., None]
        pff[..., n_z - 1, :-1] = pff[..., :-1, n_z - 1]
        pbb[..., n_z - 1, :-1] = pbb[..., :-1, n_z - 1]
    return pff, pbb


def _phase_flat(phase, mu, weights, n_streams):
    """Forward and backward phase for a flat list of scattering cells.

    ``phase`` is (m, n_leg + 1). ``mu`` is (m, n_angles). The last phase
    coefficient is the delta-truncation term and is not summed.
    """
    n_terms = phase.shape[-1] - 1
    pleg = _legendre_batch(mu, n_terms)
    coeff = phase[:, :n_terms]
    pff = np.einsum("ml,mli,mlj->mij", coeff, pleg, pleg, optimize=True)
    sign = np.where(np.arange(n_terms) % 2 == 0, 1.0, -1.0)
    pbb = np.einsum("ml,l,mli,mlj->mij", coeff, sign, pleg, pleg, optimize=True)
    pff = np.where(pff >= 0.0, pff, _PHASE_FLOOR)
    pbb = np.where(pbb >= 0.0, pbb, _PHASE_FLOOR)
    return _normalize_phase_batch(pff, pbb, weights, n_streams)


def _eig_stack(hh):
    """Real eigenpairs of a stack of nonsymmetric matrices.

    ``ASYMTX`` and ``numpy.linalg.eig`` agree on the eigenvalues of these
    ADA matrices to ~1e-12 after sorting. Eigenvectors agree up to scale.
    Columns here are scaled to unit length. Order and sign do not change
    the reflection and transmission that the adding step builds.
    """
    ev, vec = np.linalg.eig(hh)
    ev = np.real(ev)
    vec = np.real(vec)
    norm = np.linalg.norm(vec, axis=-2, keepdims=True)
    vec = vec / np.maximum(norm, 1e-30)
    return ev, vec


def _amom_batch(ssa, tau, mu, weights, pff, pbb, planck, n_streams):
    """Layer operators. Leading axes are the batch; the last two are angles."""
    albedo = np.minimum(ssa, _MAX_ALBEDO)
    n_z = mu.shape[-1]
    tiny = tau < _DELTA_OD
    c = albedo[..., None] / mu
    # Single-scatter branch, used for very thin layers.
    refl_t = c[..., None] * pbb * weights
    trans_t = c[..., None] * pff * weights
    diag = 1.0 - tau[..., None] / mu
    trans_t = trans_t + np.eye(n_z) * diag[..., None]
    thermal_t = np.sum(refl_t + trans_t, axis=-1)
    src_t = (1.0 - thermal_t) * planck[..., None]
    # Eigenvalue branch.
    pm = c[..., None] * pbb * weights
    pp = c[..., None] * pff * weights
    pp = pp - np.eye(n_z) / mu[..., None]
    ppm = pp - pm
    ppp = pp + pm
    hh = ppm @ ppp
    eigval, eigvec = _eig_stack(hh)
    gamma = np.sqrt(np.maximum(eigval, 0.0))
    eigvf = np.linalg.solve(ppm, eigvec * gamma[..., None, :])
    gp = 0.5 * (eigvec + eigvf)
    gm = 0.5 * (eigvec - eigvf)
    exp_x = np.exp(-np.clip(gamma * tau[..., None], 0.0, 80.0))
    a1 = gp * exp_x[..., None, :]
    a4 = gm * exp_x[..., None, :]
    a2 = np.linalg.solve(gm, a1)
    a3 = gp @ a2
    a5 = a1 @ a2
    a6 = a4 @ a2
    gm_a5 = gm - a5
    # ``solve`` does not broadcast a bare (n_z, n_z) identity onto a
    # rank-3 stack. The batch may be (m, n_z, n_z) once cloudy cells are flat.
    identity = np.broadcast_to(np.eye(n_z), gm_a5.shape)
    inv = np.linalg.solve(gm_a5, identity)
    trans = (a4 - a3) @ inv
    refl = (gp - a6) @ inv
    thermal = np.sum(trans[..., :n_streams] + refl[..., :n_streams], axis=-1)
    if n_z == n_streams + 1:
        thermal = thermal.copy()
        thermal[..., -1] = thermal[..., -1] + trans[..., -1, -1]
    src = (1.0 - thermal) * planck[..., None]
    tiny_b = tiny[..., None, None]
    trans = np.where(tiny_b, trans_t, trans)
    refl = np.where(tiny_b, refl_t, refl)
    src = np.where(tiny[..., None], src_t, src)
    return trans, refl, src


def _scattering_operators(tau, ssa, phase, planck, mu, weights, n_streams):
    """Reflection and transmission where the albedo exceeds the ADA threshold.

    The stack is sorted by layer so the adding loop can take one slice per
    level. ``None`` means every cell is emission-only.
    """
    pi, ci, ki = np.nonzero(ssa > _ALBEDO_THRESHOLD)
    if pi.size == 0:
        return None
    order = np.argsort(ki, kind="stable")
    pi, ci, ki = pi[order], ci[order], ki[order]
    n_z = mu.shape[-1]
    m = pi.size
    trans = np.empty((m, n_z, n_z), dtype=np.float64)
    refl = np.empty_like(trans)
    src = np.empty((m, n_z), dtype=np.float64)
    for start in range(0, m, _EIG_BATCH):
        stop = min(start + _EIG_BATCH, m)
        sel = slice(start, stop)
        rows = pi[sel]
        cols = ci[sel]
        lays = ki[sel]
        pff, pbb = _phase_flat(phase[rows, cols, lays], mu[rows], weights, n_streams)
        trans_s, refl_s, src_s = _amom_batch(
            ssa[rows, cols, lays],
            tau[rows, cols, lays],
            mu[rows],
            weights,
            pff,
            pbb,
            planck[rows, cols, lays],
            n_streams,
        )
        trans[sel] = trans_s
        refl[sel] = refl_s
        src[sel] = src_s
    return pi, ci, ki, trans, refl, src


def _layer_spans(layer_index):
    """Half-open slices of a layer-sorted index, keyed by layer number."""
    if layer_index.size == 0:
        return {}
    changes = np.flatnonzero(np.diff(layer_index)) + 1
    bounds = np.concatenate(([0], changes, [layer_index.size]))
    return {int(layer_index[bounds[i]]): (int(bounds[i]), int(bounds[i + 1])) for i in range(bounds.size - 1)}


def _ada_cloudy(tau, ssa, phase, planck, skin, emis, cosmic, mu_sensor, n_leg):
    n_streams = max((n_leg + 2) // 2, 1)
    mu_q, wt_q = _quadrature(n_streams)
    mu = np.concatenate([np.broadcast_to(mu_q, (mu_sensor.size, n_streams)), mu_sensor[:, None]], axis=1)
    weights = np.append(wt_q, 0.0)
    n_z = n_streams + 1
    # A sensor angle that lands on a node is rare for these Gauss points.
    # Columns that do match are integrated on the scalar path by the caller.
    level_rad = np.broadcast_to(emis[..., None] * skin[..., None], emis.shape + (n_z,)).copy()
    level_refl = np.zeros(emis.shape + (n_z, n_z), dtype=np.float64)
    idx = np.arange(n_z)
    level_refl[..., idx, idx] = (1.0 - emis)[..., None]
    ops = _scattering_operators(tau, ssa, phase, planck, mu, weights, n_streams)
    if ops is None:
        spans = {}
        op_pi = op_ci = op_trans = op_refl = op_src = None
    else:
        op_pi, op_ci, op_ki, op_trans, op_refl, op_src = ops
        spans = _layer_spans(op_ki)
    for k in range(tau.shape[-1] - 1, -1, -1):
        trans_b = np.exp(-tau[..., k, None] / mu[:, None, :])
        src_b = planck[..., k, None] * (1.0 - trans_b)
        trans = np.zeros(tau.shape[:-1] + (n_z, n_z), dtype=np.float64)
        trans[..., idx, idx] = trans_b
        refl = np.zeros_like(trans)
        src = src_b
        span = spans.get(k)
        if span is not None:
            a, b = span
            rows = op_pi[a:b]
            cols = op_ci[a:b]
            trans[rows, cols] = op_trans[a:b]
            refl[rows, cols] = op_refl[a:b]
            src[rows, cols] = op_src[a:b]
        below_r = level_refl
        below_u = level_rad
        temporal = -below_r @ refl
        temporal[..., idx, idx] += 1.0
        inv_gt = trans @ np.linalg.inv(temporal)
        refl_down = np.einsum("...ij,...j->...i", below_r, src, optimize=True)
        level_rad = src + np.einsum("...ij,...j->...i", inv_gt, refl_down + below_u, optimize=True)
        level_refl = refl + inv_gt @ (below_r @ trans)
    upward = level_rad[..., -1] + np.sum(level_refl[..., -1, :], axis=-1) * cosmic
    return upward


_QUAD: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def _quadrature(n_streams):
    cached = _QUAD.get(n_streams)
    if cached is None:
        nodes, weights = np.polynomial.legendre.leggauss(n_streams)
        mu = 0.5 * (nodes + 1.0)
        wt = 0.5 * weights
        order = np.argsort(mu)
        cached = (mu[order], wt[order])
        _QUAD[n_streams] = cached
    return cached


def _angles_match_node(mu_sensor, n_streams):
    mu_q, _wt = _quadrature(n_streams)
    return np.min(np.abs(mu_sensor[:, None] - mu_q), axis=1) < _ANGLE_MATCH


def forward_chunk(coeff, level_p, pressure, temperature, h2o, o3, clouds, zenith, surface, bands, scattering):
    """Brightness temperature for one chunk, shape ``(n_profile, n_channel)``."""
    level_p, pressure, temperature, h2o, o3, n_added = prepend_standard_atmosphere_batch(
        level_p, pressure, temperature, h2o, o3
    )
    if n_added and clouds:
        pad = np.zeros((level_p.shape[0], n_added), dtype=np.float64)
        clouds = [
            _Cloud(
                cloud.cloud_type,
                np.concatenate([pad, np.where(cloud.water > _WATER_CONTENT_THRESHOLD, cloud.water, 0.0)], axis=1),
                np.concatenate([pad, cloud.radius], axis=1),
            )
            for cloud in clouds
        ]
    elif clouds:
        clouds = [
            _Cloud(
                cloud.cloud_type,
                np.where(cloud.water > _WATER_CONTENT_THRESHOLD, cloud.water, 0.0),
                cloud.radius,
            )
            for cloud in clouds
        ]
    gas = gas_optical_depth(coeff.odps, level_p, pressure, temperature, h2o, o3, zenith, bands)
    channel_ids = [coeff.spc.channel_index(int(b)) for b in bands]
    wn = coeff.spc.wavenumber[np.array(channel_ids)]
    tau = gas
    ssa = np.zeros_like(tau)
    phase = None
    n_leg = 0
    if clouds and coeff.cloud is not None:
        cloud_tau, cloud_ssa, phase, n_leg = cloud_optics_batch(coeff.cloud, wn, clouds, scattering)
        if scattering and phase is not None:
            total = tau + cloud_tau
            ssa = np.divide(cloud_ssa * cloud_tau, total, out=np.zeros_like(total), where=total > 0)
            tau = total
        else:
            tau = tau + cloud_tau
            phase = None
            n_leg = 0
    planck = _planck(coeff.spc, channel_ids, temperature)
    skin_t = _skin_temperature(surface)
    skin = _planck(coeff.spc, channel_ids, np.broadcast_to(skin_t[:, None], (skin_t.size, 1)))[..., 0]
    emis = _emissivity(
        coeff,
        channel_ids,
        zenith,
        surface["wind"],
        surface["land_type"],
        surface["water"],
        surface["land"],
        surface["ice"],
        surface["snow"],
        surface["emissivity"],
    )
    cosmic = coeff.spc.cosmic_background[channel_ids]
    mu = np.cos(np.deg2rad(zenith))
    tb_rad = np.empty((tau.shape[0], len(bands)), dtype=np.float64)
    clear = np.ones(tau.shape[0], dtype=bool) if phase is None else ~np.any(ssa > _ALBEDO_THRESHOLD, axis=(1, 2))
    if np.any(clear):
        tb_rad[clear] = _emission(tau[clear], planck[clear], skin[clear], emis[clear], cosmic, mu[clear])
    if phase is not None and np.any(~clear):
        idx = np.flatnonzero(~clear)
        # Split by stream count. ABI infrared with the usual radii is one group.
        max_re = _chunk_max_radius(clouds, idx)
        groups: dict[int, list[int]] = {}
        for local, prof in enumerate(idx):
            count = n_streams_for_radius(float(max_re[local]), float(wn.min())) if max_re[local] > 0 else n_leg
            groups.setdefault(int(count), []).append(prof)
        for count, profs in groups.items():
            sel = np.array(profs, dtype=np.int64)
            matched = _angles_match_node(mu[sel], max((count + 2) // 2, 1))
            ordinary = sel[~matched]
            if ordinary.size:
                tb_rad[ordinary] = _ada_cloudy(
                    tau[ordinary],
                    ssa[ordinary],
                    phase[ordinary],
                    planck[ordinary],
                    skin[ordinary],
                    emis[ordinary],
                    cosmic,
                    mu[ordinary],
                    count,
                )
            if np.any(matched):
                from crtm_py.ada import ada_radiance

                for prof in sel[matched]:
                    for ic, band in enumerate(bands):
                        streams = phase.shape[-1] - 1
                        tb_rad[prof, ic] = ada_radiance(
                            tau[prof, ic],
                            ssa[prof, ic],
                            phase[prof, ic].T,
                            planck[prof, ic],
                            skin[prof, ic],
                            emis[prof, ic],
                            cosmic[ic],
                            mu[prof],
                            streams=streams,
                        )
    return _temperature_batch(coeff.spc, channel_ids, tb_rad), tb_rad


class _Cloud:
    def __init__(self, cloud_type, water, radius):
        self.cloud_type = cloud_type
        self.water = water
        self.radius = radius


def _chunk_max_radius(clouds, idx):
    max_re = np.zeros(idx.size, dtype=np.float64)
    for cloud in clouds:
        wet = cloud.water[idx] > _WATER_CONTENT_THRESHOLD
        max_re = np.maximum(max_re, np.max(np.where(wet, cloud.radius[idx], 0.0), axis=1))
    return max_re


def _skin_temperature(surface):
    weights = np.stack(
        [
            np.maximum(surface["water"], 0),
            np.maximum(surface["land"], 0),
            np.maximum(surface["ice"], 0),
            np.maximum(surface["snow"], 0),
        ],
        axis=1,
    )
    temps = np.stack(
        [surface["water_t"], surface["land_t"], surface["ice_t"], surface["snow_t"]],
        axis=1,
    )
    total = weights.sum(axis=1)
    weights = np.where(total[:, None] > 0, weights / np.maximum(total[:, None], 1e-30), 0.0)
    weights[total <= 0, 1] = 1.0
    return np.sum(weights * temps, axis=1)


def _temperature_batch(spc, channel_ids, radiance):
    """Invert channel radiance. ``radiance`` is (n, channel)."""
    p1 = spc.planck_c1[channel_ids]
    p2 = spc.planck_c2[channel_ids]
    b1 = spc.band_c1[channel_ids]
    b2 = spc.band_c2[channel_ids]
    teff = p2 / np.log(p1 / radiance + 1.0)
    return (teff - b1) / b2
