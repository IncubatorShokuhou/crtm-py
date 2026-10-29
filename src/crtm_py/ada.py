"""Adding-doubling radiative transfer from ``ADA_Module``.

Clear columns use one specular stream, the same branch CRTM takes when
scattering is off (``n_Streams = 0``, ``n_Angles = 1``). Cloudy columns
use the matrix-operator layer and the adding loop. ``ASYMTX`` assumes a
symmetric matrix. The phase-function operator assembled here is not
symmetric, and forcing it to be symmetric increased the window-channel
error, so the eigenvalue step uses ``numpy.linalg.eig`` and keeps the
real part. Quadrature is Gauss-Legendre mapped onto ``(0, 1)``, which is
the double-Gauss rule CRTM uses for a hemisphere. The sensor zenith is
an extra stream when it does not match a quadrature angle.
"""

from __future__ import annotations

import numpy as np

_PHASE_FLOOR = 1.0e-7
_DELTA_OD = 1.0e-8
_MAX_ALBEDO = 0.999999
_ALBEDO_THRESHOLD = 1.0e-10
_ANGLE_MATCH = 1.0e-4


def _legendre(mu: np.ndarray, n_terms: int) -> np.ndarray:
    p = np.empty((n_terms, mu.size), dtype=np.float64)
    p[0] = 1.0
    if n_terms == 1:
        return p
    p[1] = mu
    for ell in range(1, n_terms - 1):
        p[ell + 1] = ((2 * ell + 1) * mu * p[ell] - ell * p[ell - 1]) / (ell + 1)
    return p


def _double_gauss(n_streams: int):
    nodes, weights = np.polynomial.legendre.leggauss(n_streams)
    mu = 0.5 * (nodes + 1.0)
    wt = 0.5 * weights
    # CRTM/DISORT double-Gauss stores the largest cosine first.
    order = np.argsort(mu)[::-1]
    return mu[order], wt[order]


def _normalize_phase(pff, pbb, weights, n_streams):
    n_z = pff.shape[0]
    sum_fac = np.zeros(n_z + 1)
    for i in range(n_streams):
        factor = 0.0
        for j in range(i, n_z):
            factor += (pff[i, j] + pbb[i, j]) * weights[j]
        if factor == 0.0:
            factor = 1.0
        scale = (1.0 - sum_fac[i]) / factor
        for j in range(i, n_z):
            pff[i, j] *= scale
            pbb[i, j] *= scale
        if i + 1 < n_z:
            for j in range(i + 1, n_z):
                pff[j, i] = pff[i, j]
                pbb[j, i] = pbb[i, j]
            acc = 0.0
            for j in range(i + 1):
                acc += (pff[i + 1, j] + pbb[i + 1, j]) * weights[j]
            sum_fac[i + 1] = acc
    if n_streams < n_z:
        factor = sum_fac[n_z - 1]
        if factor == 0.0:
            factor = 1.0
        for j in range(n_z):
            pff[j, n_z - 1] /= factor
            pbb[j, n_z - 1] /= factor
            if j < n_z - 1:
                pff[n_z - 1, j] = pff[j, n_z - 1]
                pbb[n_z - 1, j] = pbb[j, n_z - 1]


def _phase_matrices(phase, mu, weights, n_streams):
    """Forward and backward phase matrices for one layer. ``phase[0:n_terms]``."""
    n_terms = phase.size - 1  # last entry is the delta term, not part of the sum
    n_z = mu.size
    pleg = _legendre(mu, n_terms)
    pff = np.zeros((n_z, n_z))
    pbb = np.zeros((n_z, n_z))
    for i in range(n_z):
        for j in range(n_z):
            acc_f = 0.0
            acc_b = 0.0
            for ell in range(n_terms):
                term = phase[ell] * pleg[ell, i] * pleg[ell, j]
                acc_f += term
                acc_b += term if ell % 2 == 0 else -term
            pff[i, j] = acc_f if acc_f >= 0.0 else _PHASE_FLOOR
            pbb[i, j] = acc_b if acc_b >= 0.0 else _PHASE_FLOOR
    _normalize_phase(pff, pbb, weights, n_streams)
    return pff, pbb


def _amom(ssa, tau, mu, weights, pff, pbb, planck, n_streams):
    n_z = mu.size
    albedo = min(ssa, _MAX_ALBEDO)
    if tau < _DELTA_OD:
        trans = np.zeros((n_z, n_z))
        refl = np.zeros((n_z, n_z))
        source = np.zeros(n_z)
        for i in range(n_z):
            c = tau * albedo / mu[i]
            thermal = 0.0
            for j in range(n_z):
                refl[i, j] = c * pbb[i, j] * weights[j]
                trans[i, j] = c * pff[i, j] * weights[j]
                if i == j:
                    trans[i, i] += 1.0 - tau / mu[i]
                thermal += refl[i, j] + trans[i, j]
            source[i] = (1.0 - thermal) * planck
        return trans, refl, source, source
    pm = np.zeros((n_z, n_z))
    pp = np.zeros((n_z, n_z))
    for i in range(n_z):
        c = albedo / mu[i]
        for j in range(n_z):
            pm[i, j] = c * pbb[i, j] * weights[j]
            pp[i, j] = c * pff[i, j] * weights[j]
        pp[i, i] -= 1.0 / mu[i]
    ppm = pp - pm
    ppp = pp + pm
    hh = ppm @ ppp
    eigval, eigvec = np.linalg.eig(hh)
    eigval = np.real(eigval)
    eigvec = np.real(eigvec)
    eigval = np.clip(eigval, 0.0, None)
    gamma = np.sqrt(eigval)
    eigvf = np.linalg.lstsq(ppm, eigvec * gamma, rcond=None)[0]
    gp = 0.5 * (eigvec + eigvf)
    gm = 0.5 * (eigvec - eigvf)
    exp_x = np.exp(-np.clip(gamma * tau, 0.0, 80.0))
    a1 = gp * exp_x
    a4 = gm * exp_x
    a2 = np.linalg.lstsq(gm, a1, rcond=None)[0]
    a3 = gp @ a2
    a5 = a1 @ a2
    a6 = a4 @ a2
    gm_a5 = gm - a5
    inv = np.linalg.pinv(gm_a5)
    trans = (a4 - a3) @ inv
    refl = (gp - a6) @ inv
    source = np.zeros(n_z)
    for i in range(n_z):
        thermal = 0.0
        for j in range(n_streams):
            thermal += trans[i, j] + refl[i, j]
        if i == n_z - 1 and n_z == n_streams + 1:
            thermal += trans[n_z - 1, n_z - 1]
        source[i] = (1.0 - thermal) * planck
    return trans, refl, source, source.copy()


def ada_radiance(tau, ssa, phase, planck_layer, planck_surface, emissivity, cosmic, mu_sensor, *, streams: int):
    """Upward radiance at the sensor zenith.

    ``tau`` and ``ssa`` are nadir quantities. ``phase`` is ``None`` or
    shape ``(n_legendre + 1, n_layers)``. ``mu_sensor`` is the cosine of
    the sensor zenith angle.
    """
    n_layers = tau.size
    scattering = phase is not None and np.any(ssa > _ALBEDO_THRESHOLD)
    if not scattering:
        mu = np.array([mu_sensor])
        weights = np.array([1.0])
        n_streams = 0
        n_z = 1
    else:
        # CRTM stores n_full = n_mie + 2 and uses n_full/2 hemispheric streams.
        n_streams = max((streams + 2) // 2, 1)
        mu, weights = _double_gauss(n_streams)
        if np.min(np.abs(mu - mu_sensor)) < _ANGLE_MATCH:
            n_z = n_streams
        else:
            mu = np.append(mu, mu_sensor)
            weights = np.append(weights, 0.0)
            n_z = n_streams + 1
    level_rad = np.zeros((n_z, n_layers + 1))
    level_refl = np.zeros((n_z, n_z, n_layers + 1))
    level_refl[:, :, n_layers] = 0.0
    for i in range(n_z):
        level_refl[i, i, n_layers] = 1.0 - emissivity
    level_rad[:, n_layers] = emissivity * planck_surface
    for k in range(n_layers - 1, -1, -1):
        if ssa[k] > _ALBEDO_THRESHOLD and phase is not None:
            pff, pbb = _phase_matrices(phase[:, k], mu, weights, n_streams)
            trans, refl, src_up, src_down = _amom(
                ssa[k], tau[k], mu, weights, pff, pbb, planck_layer[k], n_streams
            )
            temporal = -level_refl[:, :, k + 1] @ refl
            temporal.flat[:: n_z + 1] += 1.0
            inv_gamma = np.linalg.inv(temporal)
            inv_gt = trans @ inv_gamma
            refl_down = level_refl[:, :, k + 1] @ src_down
            level_rad[:, k] = src_up + inv_gt @ (refl_down + level_rad[:, k + 1])
            level_refl[:, :, k] = refl + inv_gt @ (level_refl[:, :, k + 1] @ trans)
        else:
            trans = np.exp(-tau[k] / mu)
            src = planck_layer[k] * (1.0 - trans)
            for i in range(n_z):
                reflected = np.dot(level_refl[i, :, k + 1], src)
                level_rad[i, k] = src[i] + trans[i] * (reflected + level_rad[i, k + 1])
            for i in range(n_z):
                for j in range(n_z):
                    level_refl[i, j, k] = trans[i] * level_refl[i, j, k + 1] * trans[j]
    sensor = n_z - 1
    upward = level_rad[sensor, 0]
    upward += level_refl[sensor, :, 0].sum() * cosmic
    return float(upward)
