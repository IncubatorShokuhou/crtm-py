"""CloudCoeff lookup used by ``Get_Cloud_Opt_IR``.

Effective radius in the ABI-era table is microns (2–100 µm). The comment
in ``Get_Cloud_Opt_IR`` that says millimetres does not match this table or
the Mie-parameter formula in ``CRTM_Compute_nStreams``, which treats the
radius as microns.

Mass extinction ``ke`` is m^2 kg^-1. With scattering off, CRTM replaces
``ke`` by ``ke * (1 - w)`` so the emission solver sees absorption only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from crtm_py.crtm_binary import FortranFile
from crtm_py.lagrange import find_random_index, interp_2d, lagrange_weights

_WATER_CONTENT_THRESHOLD = 1.0e-6
_DENSITY_INDEX = {
    "water": 0,
    "rain": 0,
    "snow": 1,
    "graupel": 2,
    "ice": 3,
    "hail": 3,
}
_LEGENDRE_OFFSET = {2: 0, 4: 0, 6: 5, 8: 12, 16: 21}


@dataclass
class CloudCoeff:
    frequency_ir: np.ndarray
    reff_ir: np.ndarray
    ke_ir: np.ndarray  # (n_freq, n_reff, n_density)
    w_ir: np.ndarray
    pcoeff_ir: np.ndarray  # (n_freq, n_reff, n_density, n_legendre)


def load_cloudcoeff(path) -> CloudCoeff:
    ff = FortranFile(path)
    ff.read_i32s((2,))
    n_mw_f, n_mw_r, n_ir_f, n_ir_r, _n_t, n_den, n_leg, _n_phase = [int(v) for v in ff.read_i32s((8,))]
    vec = ff.read_f64()
    frequency = vec[n_mw_f : n_mw_f + n_ir_f].copy()
    reff_at = n_mw_f + n_ir_f + n_mw_r
    reff = vec[reff_at : reff_at + n_ir_r].copy()
    ff.read_record()  # microwave liquid
    ff.read_record()  # microwave solid
    ir = ff.read_f64()
    n_den_store = n_den + 1  # Fortran 0:n_Densities
    n_leg_store = n_leg + 1
    n_ke = n_ir_f * n_ir_r * n_den_store
    ke = ir[:n_ke].reshape((n_ir_f, n_ir_r, n_den_store), order="F")
    w = ir[n_ke : 2 * n_ke].reshape((n_ir_f, n_ir_r, n_den_store), order="F")
    # g is the next block; phase coefficients follow it
    p0 = 3 * n_ke
    pcoeff = ir[p0:].reshape((n_ir_f, n_ir_r, n_den_store, n_leg_store), order="F")
    return CloudCoeff(frequency_ir=frequency, reff_ir=reff, ke_ir=ke, w_ir=w, pcoeff_ir=pcoeff)


def cloud_bulk(coeff: CloudCoeff, wavenumber, cloud_type: str, radius_um: float, *, scattering: bool, n_legendre: int):
    """Return ``ke``, single-scatter albedo and phase coefficients.

    Phase coefficients follow ``CRTM_AtmOptics_Combine``: index 0 is 0.5
    and index ``n_legendre`` is the delta-truncation term. ``ke`` is
    extinction when ``scattering`` is true and absorption otherwise.
    """
    if cloud_type not in _DENSITY_INDEX:
        raise KeyError(cloud_type)
    k = _DENSITY_INDEX[cloud_type]
    i1, freq = find_random_index(coeff.frequency_ir, wavenumber)
    j1, radius = find_random_index(coeff.reff_ir, radius_um)
    freq_w = lagrange_weights(coeff.frequency_ir[i1 : i1 + 4], freq)
    radius_w = lagrange_weights(coeff.reff_ir[j1 : j1 + 4], radius)
    ke = interp_2d(coeff.ke_ir[i1 : i1 + 4, j1 : j1 + 4, k], freq_w, radius_w)
    w = interp_2d(coeff.w_ir[i1 : i1 + 4, j1 : j1 + 4, k], freq_w, radius_w)
    if ke < 0.0:
        ke = 0.0
    w = float(np.clip(w, 0.0, 1.0))
    if not scattering:
        return ke * (1.0 - w), 0.0, None
    offset = _LEGENDRE_OFFSET[n_legendre]
    phase = np.zeros(n_legendre + 1, dtype=np.float64)
    phase[0] = 0.5
    for ell in range(1, n_legendre + 1):
        phase[ell] = interp_2d(
            coeff.pcoeff_ir[i1 : i1 + 4, j1 : j1 + 4, k, ell + offset], freq_w, radius_w
        )
    return ke, w, phase


def n_streams_for_radius(max_radius_um: float, wavenumber: float) -> int:
    """``CRTM_Compute_nStreams`` before the ``+ 2`` stored on the RT solution."""
    mie = 2.0 * np.pi * max_radius_um * wavenumber / 10000.0
    if mie < 0.01:
        return 2
    if mie < 1.0:
        return 4
    return 6


def layer_cloud_optics(coeff: CloudCoeff, wavenumber, clouds, temperature_shape, *, scattering: bool):
    """Accumulate cloud optical depth, albedo and phase on user layers.

    ``clouds`` is a sequence of objects with ``cloud_type``,
    ``water_content`` and ``effective_radius``. Returns nadir extinction
    (or absorption), single-scatter albedo and phase ``(n_leg+1, n_layers)``.
    """
    n_layers = int(temperature_shape)
    tau = np.zeros(n_layers, dtype=np.float64)
    bs = np.zeros(n_layers, dtype=np.float64)
    if not clouds:
        return tau, bs, None
    max_re = 0.0
    for cloud in clouds:
        if np.any(cloud.water_content > _WATER_CONTENT_THRESHOLD):
            max_re = max(max_re, float(np.max(cloud.effective_radius)))
    n_leg = n_streams_for_radius(max_re, wavenumber) if scattering and max_re > 0.0 else 0
    phase_acc = np.zeros((n_leg + 1, n_layers), dtype=np.float64) if n_leg else None
    for cloud in clouds:
        for k in range(n_layers):
            rho = float(cloud.water_content[k])
            if rho <= _WATER_CONTENT_THRESHOLD:
                continue
            re = float(cloud.effective_radius[k])
            if re <= 0.0:
                re = float(coeff.reff_ir[0])
            ke, w, phase = cloud_bulk(
                coeff, wavenumber, cloud.cloud_type, re, scattering=scattering and n_leg > 0, n_legendre=max(n_leg, 2)
            )
            tau[k] += ke * rho
            if phase_acc is not None and phase is not None and w > 0.0:
                scatter = rho * ke * w
                bs[k] += scatter
                phase_acc[1:, k] += phase[1 : n_leg + 1] * scatter
    ssa = np.zeros(n_layers, dtype=np.float64)
    if phase_acc is not None:
        for k in range(n_layers):
            if bs[k] > 1.0e-10 and tau[k] > 0.0:
                w = bs[k] / tau[k]
                phase_acc[1:, k] /= bs[k]
                phase_acc[0, k] = 0.5
                delta = phase_acc[n_leg, k] if n_leg < phase_acc.shape[0] else 0.0
                # index n_leg is the truncation term; the phase sum uses 0..n_leg-1
                factor = 1.0 - delta * w
                if factor <= 0.0:
                    factor = 1.0e-6
                tau[k] = factor * tau[k]
                ssa[k] = (1.0 - delta) * w / factor
                ssa[k] = min(max(ssa[k], 0.0), 0.999999)
    return tau, ssa, phase_acc
