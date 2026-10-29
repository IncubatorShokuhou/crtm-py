"""Effective radius from UPP ``EFFR`` and the column water UPP passes to CRTM.

``effective_radius`` follows ``FUNCTION EFFR`` in
``CALRAD_WCLOUD_newcrtm.f`` (``dtc_post_v4.1.0``), including the final
conversion from diameter to radius. Two behaviors of that function are
kept on purpose:

* WSM6 ice uses the number concentration where the WSM6 source uses
  particle mass, so the radius sits on the 250 µm cap whenever ice is
  present.
* Thompson graupel divides by ``params.F`` ``RG``, and ``RG`` equals
  gravity (9.81), not the graupel mixing ratio.

The inlined Ferrier / Zhao assignments in ``CALRAD`` are not all equal
to ``EFFR``. ``cloud_profiles_from_upp`` follows the subroutine, which
is what ``CRTM_Forward`` actually receives. ``effective_radius`` remains
available for the function itself.

Radii are microns. Water contents passed onward are kg m^-2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from crtm_py.constants import D608, G, PI, RD

_G_RATIO = np.array(
    [6, 24, 60, 120, 210, 336, 504, 720, 990, 1320, 1716, 2184, 2730, 3360, 4080, 4896],
    dtype=np.float64,
)

_MIN_QC = 1.0e-7
_MIN_QR = 1.0e-7
_MIN_QI = 1.0e-8
_MIN_QS = 1.0e-8
_MIN_QG = 1.0e-7

_WSM6_CNP = 3.0e8
_WSM6_RHOR = 1000.0
_WSM6_RHOS = 100.0
_WSM6_RHOG = 500.0
_WSM6_DIMAX = 500.0e-6
_WSM6_DICON = 11.9
_WSM6_N0S = 2.0e6
_WSM6_N0R = 8.0e6
_WSM6_N0G = 4.0e6
_WSM6_QMIN = 1.0e-15

_LIN_RHOI = 100.0
_LIN_RHOR = 1000.0
_LIN_RHOS = 100.0
_LIN_RHOG = 400.0
_LIN_CNP = 3.0e8
_LIN_N0R = 8.0e6
_LIN_N0S = 3.0e6
_LIN_N0G = 4.0e6

_BETA_CRG = 3.0
_BETA_S = 2.4
_GAMMA_CRG = 6.0
_GAMMA_S = 2.981134

_NTHOM_RHOR = 1000.0
_NTHOM_RHOS = 100.0
_NTHOM_RHOG = 500.0
_NTHOM_RHOI = 890.0
_NTHOM_GON_MIN = 1.0e4
_NTHOM_GON_MAX = 3.0e6
_NTHOM_NT_C = 100.0e6
_NTHOM_BM_S = 2.0
_NTHOM_SA = np.array(
    [5.065339, -0.062659, -3.032362, 0.029469, -0.000285, 0.31255, 0.000204, 0.003199, 0.0, -0.015952],
    dtype=np.float64,
)
_NTHOM_SB = np.array(
    [0.476221, -0.015896, 0.165977, 0.007468, -0.000141, 0.060366, 0.000079, 0.000594, 0.0, -0.003577],
    dtype=np.float64,
)

_GFDL_RHOR = 1000.0
_GFDL_RHOS = 100.0
_GFDL_RHOG = 400.0
_GFDL_CCN = 1.0e8
_GFDL_BETA = 1.22
_GFDL_TICE = 273.16
_GFDL_GAMMAR = 17.837789
_GFDL_GAMMAS = 8.2850630
_GFDL_GAMMAG = 11.631769
_GFDL_ALPHAR = 0.8
_GFDL_ALPHAS = 0.25
_GFDL_ALPHAG = 0.5
_GFDL_N0R = 8.0e6
_GFDL_N0S = 3.0e6
_GFDL_N0G = 4.0e6
_GFDL_REWMIN, _GFDL_REWMAX = 5.0, 10.0
_GFDL_REIMIN, _GFDL_REIMAX = 10.0, 150.0
_GFDL_RERMIN, _GFDL_RERMAX = 0.0, 10000.0
_GFDL_RESMIN, _GFDL_RESMAX = 0.0, 10000.0
_GFDL_REGMIN, _GFDL_REGMAX = 0.0, 10000.0


def gammln(xx):
    """``GAMMLN``: ``ln(Gamma(xx))`` for ``xx > 0``, Lanczos form from UPP."""
    cof = np.array(
        [
            76.18009172947146,
            -86.50532032941677,
            24.01409824083091,
            -1.231739572450155,
            0.1208650973866179e-2,
            -0.5395239384953e-5,
        ],
        dtype=np.float64,
    )
    x = np.asarray(xx, dtype=np.float64)
    y = np.array(x, dtype=np.float64, copy=True)
    tmp = (x + 0.5) * np.log(x + 5.5) - (x + 5.5)
    ser = np.full(np.shape(x), 1.000000000190015, dtype=np.float64)
    for j in range(6):
        y = y + 1.0
        ser = ser + cof[j] / y
    return tmp + np.log(2.5066282746310005 * ser / x)


def wgamma(y):
    """``WGAMMA``: ``exp(GAMMLN(y))``."""
    return np.exp(gammln(y))


def air_density(pressure_pa, temperature, specific_humidity):
    """Moist-air density, ``pmid / (Rd T (1 + 0.608 q))``."""
    p = np.asarray(pressure_pa, dtype=np.float64)
    t = np.asarray(temperature, dtype=np.float64)
    q = np.asarray(specific_humidity, dtype=np.float64)
    return p / (RD * t * (1.0 + D608 * q))


def _nthom_ab(tc0, bm):
    tc2 = tc0 * tc0
    tc3 = tc2 * tc0
    bm2 = bm * bm
    bm3 = bm2 * bm
    sa, sb = _NTHOM_SA, _NTHOM_SB
    loga = (
        sa[0]
        + sa[1] * tc0
        + sa[2] * bm
        + sa[3] * tc0 * bm
        + sa[4] * tc2
        + sa[5] * bm2
        + sa[6] * tc2 * bm
        + sa[7] * tc0 * bm2
        + sa[8] * tc3
        + sa[9] * bm3
    )
    b = (
        sb[0]
        + sb[1] * tc0
        + sb[2] * bm
        + sb[3] * tc0 * bm
        + sb[4] * tc2
        + sb[5] * bm2
        + sb[6] * tc2 * bm
        + sb[7] * tc0 * bm2
        + sb[8] * tc3
        + sb[9] * bm3
    )
    return 10.0**loga, b


def _wsm6(species, rho, qqw, qqi, qqr, qqs, qqg):
    out = np.zeros(np.shape(rho), dtype=np.float64)
    if species == "C":
        mask = qqw > _MIN_QC
        out = np.where(
            mask,
            1.0e6 * ((6.0 * rho * qqw) / (PI * _WSM6_RHOR * _WSM6_CNP)) ** (1.0 / 3.0),
            out,
        )
    elif species == "R":
        mask = qqr > _MIN_QR
        out = np.where(
            mask,
            1.0e6
            * ((6.0 * rho * qqr) / (PI * _WSM6_RHOR * _WSM6_N0R * _GAMMA_CRG)) ** (1.0 / (1.0 + _BETA_CRG)),
            out,
        )
    elif species == "G":
        mask = qqg > _MIN_QG
        out = np.where(
            mask,
            1.0e6
            * ((6.0 * rho * qqg) / (PI * _WSM6_RHOG * _WSM6_N0G * _GAMMA_CRG)) ** (1.0 / (1.0 + _BETA_CRG)),
            out,
        )
    elif species == "S":
        mask = qqs > _MIN_QS
        out = np.where(
            mask,
            1.0e6
            * ((6.0 * rho * qqs) / (PI * _WSM6_RHOS * _WSM6_N0S * _GAMMA_S)) ** (1.0 / (1.0 + _BETA_S)),
            out,
        )
    elif species == "I":
        mask = qqi > _MIN_QI
        xmitemp = rho * np.maximum(qqi, _WSM6_QMIN)
        xmitemp = np.sqrt(np.sqrt(xmitemp * xmitemp * xmitemp))
        xmi = np.minimum(np.maximum(5.38e7 * xmitemp, 1.0e3), 1.0e6)
        diam = np.maximum(np.minimum(_WSM6_DICON * np.sqrt(xmi), _WSM6_DIMAX), 1.0e-25)
        out = np.where(mask, 1.0e6 * diam, out)
    else:
        raise ValueError(f"WSM6 has no species {species!r}")
    return out


def _lin(species, rho, qqw, qqi, qqr, qqs, qqg):
    out = np.zeros(np.shape(rho), dtype=np.float64)
    specs = {
        "C": (qqw, _MIN_QC, _LIN_RHOR, _LIN_CNP, None),
        "I": (qqi, _MIN_QI, _LIN_RHOI, _LIN_CNP, None),
        "R": (qqr, _MIN_QR, _LIN_RHOR, _LIN_N0R, "crg"),
        "S": (qqs, _MIN_QS, _LIN_RHOS, _LIN_N0S, "s"),
        "G": (qqg, _MIN_QG, _LIN_RHOG, _LIN_N0G, "crg"),
    }
    if species not in specs:
        raise ValueError(f"Lin has no species {species!r}")
    q, qmin, rhox, n0, kind = specs[species]
    mask = q > qmin
    if kind is None:
        val = 1.0e6 * ((6.0 * rho * q) / (PI * rhox * n0)) ** (1.0 / 3.0)
    elif kind == "crg":
        val = 1.0e6 * ((6.0 * rho * q) / (PI * rhox * n0 * _GAMMA_CRG)) ** (1.0 / (1.0 + _BETA_CRG))
    else:
        val = 1.0e6 * ((6.0 * rho * q) / (PI * rhox * n0 * _GAMMA_S)) ** (1.0 / (1.0 + _BETA_S))
    return np.where(mask, val, out)


def _thompson(species, rho, temperature, qqw, qqi, qqr, qqs, qqg, qqnr, qqni):
    out = np.zeros(np.shape(rho), dtype=np.float64)
    bm_r = 3.0
    mu_r = 0.0
    obmr = 1.0 / bm_r
    am_r = PI * _NTHOM_RHOR / 6.0
    crg1 = float(wgamma(bm_r + 1.0))
    crg2 = float(wgamma(mu_r + 1.0))
    crg3 = float(wgamma(bm_r + mu_r + 1.0))
    org2 = 1.0 / crg2

    bm_i = 3.0
    mu_i = 0.0
    am_i = PI * _NTHOM_RHOI / 6.0
    obmi = 1.0 / bm_i
    cig1 = float(wgamma(mu_i + 1.0))
    cig2 = float(wgamma(bm_i + mu_i + 1.0))
    oig1 = 1.0 / cig1

    am_s = 0.069
    oams = 1.0 / am_s
    cse1 = _NTHOM_BM_S + 1.0

    bm_g = 3.0
    mu_g = 0.0
    obmg = 1.0 / bm_g
    am_g = PI * _NTHOM_RHOG / 6.0
    cgg1 = float(wgamma(bm_g + 1.0))
    cgg2 = float(wgamma(mu_g + 1.0))
    cgg3 = float(wgamma(bm_g + mu_g + 1.0))
    oge1 = 1.0 / (bm_g + 1.0)
    ogg1 = 1.0 / cgg1
    ogg2 = 1.0 / cgg2

    if species == "C":
        mask = qqw >= _MIN_QC
        rc = np.maximum(1.0e-12, qqw * rho)
        ncc2 = np.maximum(1.0e-6, _NTHOM_NT_C * rho)
        nu = np.where(ncc2 < 10.0e6, 15, np.minimum(15, np.rint(1000.0e6 / ncc2).astype(np.int64) + 2))
        nu = nu.astype(np.int64)
        lamc = (ncc2 / rc) ** obmr * (am_r * _G_RATIO[nu]) ** obmr
        diam = np.maximum(5.01e-6, np.minimum((3.0 + nu) / lamc, 50.0e-6))
        out = np.where(mask, 1.0e6 * diam, out)
    elif species == "R":
        mask = qqr > _MIN_QR
        rr = np.maximum(1.0e-12, qqr * rho)
        ncr2 = np.maximum(1.0e-6, qqnr * rho)
        lamr = (ncr2 / rr) ** obmr * (am_r * crg3 * org2) ** obmr
        diam = np.maximum(50.01e-6, np.minimum((3.0 + mu_r) / lamr, 1999.0e-6))
        out = np.where(mask, 1.0e6 * diam, out)
    elif species == "I":
        mask = qqi >= _MIN_QI
        ri = np.maximum(1.0e-12, qqi * rho)
        nci2 = np.maximum(1.0e-6, qqni * rho)
        lami = (nci2 / ri) ** obmi * (am_i * cig2 * oig1) ** obmi
        diam = np.maximum(10.01e-6, np.minimum((3.0 + mu_i) / lami, 250.0e-6))
        out = np.where(mask, 1.0e6 * diam, out)
    elif species == "S":
        mask = qqs >= _MIN_QS
        rs = qqs * rho
        tc0 = np.minimum(-0.1, temperature - 273.15)
        smob = rs * oams
        if abs(_NTHOM_BM_S - 2.0) < 1.0e-3:
            smo2 = smob
        else:
            a_m, b_m = _nthom_ab(tc0, _NTHOM_BM_S)
            smo2 = (smob / a_m) ** (1.0 / b_m)
        a_th, b_th = _nthom_ab(tc0, cse1)
        smoc = a_th * smo2**b_th
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(smob > 0.0, smoc / smob, 0.0)
        diam = np.maximum(50.0e-6, np.minimum(ratio, 1999.0e-6))
        out = np.where(mask, 1.0e6 * diam, out)
    elif species == "G":
        mask = qqg >= _MIN_QG
        with np.errstate(divide="ignore", invalid="ignore"):
            no_exp = 200.0 / np.maximum(qqg, 1.0e-30)
        no_exp = np.clip(no_exp, _NTHOM_GON_MIN, _NTHOM_GON_MAX)
        # Divisor is params RG (= G), matching the v4.1 source symbol.
        lm_exp = (no_exp * am_g * cgg1 / G) ** oge1
        lamg = lm_exp * (cgg3 * ogg2 * ogg1) ** obmg
        out = np.where(mask, 1.0e6 * (3.0 + mu_g) / lamg, out)
    else:
        raise ValueError(f"Thompson has no species {species!r}")
    return out


def _gfdl(species, temperature, qqw, qqi, qqr, qqs, qqg):
    out = np.zeros(np.shape(temperature), dtype=np.float64)
    if species == "C":
        mask = qqw > _MIN_QC
        with np.errstate(divide="ignore", invalid="ignore"):
            raw = np.exp((1.0 / 3.0) * np.log((3.0 * qqw) / (4.0 * PI * _GFDL_RHOR * _GFDL_CCN))) * 1.0e6
        raw = np.clip(raw, _GFDL_REWMIN, _GFDL_REWMAX) * 2.0
        out = np.where(mask, raw, out)
    elif species == "I":
        mask = qqi > _MIN_QI
        tc = temperature - _GFDL_TICE
        with np.errstate(divide="ignore", invalid="ignore"):
            logq = np.log(1.0e3 * np.maximum(qqi, 1.0e-30))
        raw = np.where(
            tc < -50.0,
            _GFDL_BETA / 9.917 * np.exp((1.0 - 0.891) * logq) * 1.0e3,
            np.where(
                tc < -40.0,
                _GFDL_BETA / 9.337 * np.exp((1.0 - 0.920) * logq) * 1.0e3,
                np.where(
                    tc < -30.0,
                    _GFDL_BETA / 9.208 * np.exp((1.0 - 0.945) * logq) * 1.0e3,
                    _GFDL_BETA / 9.387 * np.exp((1.0 - 0.969) * logq) * 1.0e3,
                ),
            ),
        )
        out = np.where(mask, np.clip(raw, _GFDL_REIMIN, _GFDL_REIMAX) * 2.0, out)
    elif species == "R":
        mask = qqr > _MIN_QR
        with np.errstate(divide="ignore", invalid="ignore"):
            lambdar = np.exp(0.25 * np.log(PI * _GFDL_RHOR * _GFDL_N0R / np.maximum(qqr, 1.0e-30)))
        raw = 0.5 * np.exp(np.log(_GFDL_GAMMAR / 6.0) / _GFDL_ALPHAR) / lambdar * 1.0e6
        out = np.where(mask, np.clip(raw, _GFDL_RERMIN, _GFDL_RERMAX) * 2.0, out)
    elif species == "S":
        mask = qqs > _MIN_QS
        with np.errstate(divide="ignore", invalid="ignore"):
            lambdas = np.exp(0.25 * np.log(PI * _GFDL_RHOS * _GFDL_N0S / np.maximum(qqs, 1.0e-30)))
        raw = 0.5 * np.exp(np.log(_GFDL_GAMMAS / 6.0) / _GFDL_ALPHAS) / lambdas * 1.0e6
        out = np.where(mask, np.clip(raw, _GFDL_RESMIN, _GFDL_RESMAX) * 2.0, out)
    elif species == "G":
        mask = qqg > _MIN_QG
        with np.errstate(divide="ignore", invalid="ignore"):
            lambdag = np.exp(0.25 * np.log(PI * _GFDL_RHOG * _GFDL_N0G / np.maximum(qqg, 1.0e-30)))
        raw = 0.5 * np.exp(np.log(_GFDL_GAMMAG / 6.0) / _GFDL_ALPHAG) / lambdag * 1.0e6
        out = np.where(mask, np.clip(raw, _GFDL_REGMIN, _GFDL_REGMAX) * 2.0, out)
    else:
        raise ValueError(f"GFDL has no species {species!r}")
    return out


def _ferrier_effr(species, rho, qqr, qqs, f_rimef, nlice, nrain):
    """``EFFR`` branches for Ferrier mp_opt 5/85/95. Returns a diameter in µm.

    ``CALRAD`` itself does not call ``EFFR`` for these schemes. See
    ``cloud_profiles_from_upp``.
    """
    out = np.zeros(np.shape(rho), dtype=np.float64)
    if species == "C":
        out = np.full(np.shape(rho), 2.0 * 10.0, dtype=np.float64)
    elif species == "I":
        out = np.full(np.shape(rho), 2.0 * 25.0, dtype=np.float64)
    elif species == "R":
        mask = (qqr > _MIN_QR) & (nrain > 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            raw = 2.0 * 1.0e6 * 1.5 * (rho * qqr / (PI * 1000.0 * nrain)) ** (1.0 / 3.0)
        out = np.where(mask, raw, out)
    elif species == "S":
        mask = (f_rimef <= 5.0) & (nlice > 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            raw = 2.0 * 1.0e6 * 1.5 * (rho * qqs / (PI * 100.0 * nlice)) ** (1.0 / 3.0)
        out = np.where(mask, raw, out)
    elif species == "G":
        mask = (f_rimef > 5.0) & (f_rimef <= 20.0) & (nlice > 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            raw = 2.0 * 1.0e6 * 1.5 * (rho * qqs / (PI * 400.0 * nlice)) ** (1.0 / 3.0)
        out = np.where(mask, raw, out)
    elif species == "H":
        mask = (f_rimef > 20.0) & (nlice > 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            raw = 2.0 * 1.0e6 * 1.5 * (rho * qqs / (PI * 900.0 * nlice)) ** (1.0 / 3.0)
        out = np.where(mask, raw, out)
    else:
        raise ValueError(f"Ferrier EFFR has no species {species!r}")
    return out


def effective_radius(
    pressure_pa,
    temperature,
    specific_humidity,
    cloud_water,
    cloud_ice,
    rain,
    rime_factor,
    n_large_ice,
    n_rain,
    snow,
    graupel,
    n_rain_thompson,
    n_ice_thompson,
    mp_opt,
    species,
):
    """Effective radius in microns, half the diameter ``EFFR`` computes internally.

    Parameters
    ----------
    pressure_pa, temperature, specific_humidity : array_like
        Mid-layer pressure (Pa), temperature (K) and specific humidity
        (kg kg^-1). Broadcast together.
    cloud_water, cloud_ice, rain, snow, graupel : array_like
        Mixing ratios (kg kg^-1), UPP ``qqw, qqi, qqr, qqs, qqg``.
    rime_factor, n_large_ice, n_rain : array_like
        ``f_rimef``, ``nlice``, ``nrain``.
    n_rain_thompson, n_ice_thompson : array_like
        ``qqnr``, ``qqni``. Thompson only; other schemes ignore them.
    mp_opt : int
        WRF ``mp_physics``. Supported: 2 Lin, 5/85/95 Ferrier, 6 WSM6,
        8 Thompson, 11 GFDL. Any other option returns zeros, as ``EFFR``
        does (including WSM6-adjacent option 28).
    species : str
        ``C`` cloud water, ``I`` ice, ``R`` rain, ``S`` snow, ``G``
        graupel, ``H`` hail. Case insensitive.

    Returns
    -------
    ndarray
        Effective radius in microns.
    """
    species = species.strip().upper()
    arrays = [
        pressure_pa,
        temperature,
        specific_humidity,
        cloud_water,
        cloud_ice,
        rain,
        rime_factor,
        n_large_ice,
        n_rain,
        snow,
        graupel,
        n_rain_thompson,
        n_ice_thompson,
    ]
    (
        pressure_pa,
        temperature,
        specific_humidity,
        qqw,
        qqi,
        qqr,
        f_rimef,
        nlice,
        nrain,
        qqs,
        qqg,
        qqnr,
        qqni,
    ) = np.broadcast_arrays(*[np.asarray(a, dtype=np.float64) for a in arrays])
    rho = air_density(pressure_pa, temperature, specific_humidity)
    mp = int(mp_opt)
    if mp == 6:
        diam = _wsm6(species, rho, qqw, qqi, qqr, qqs, qqg)
    elif mp == 2:
        diam = _lin(species, rho, qqw, qqi, qqr, qqs, qqg)
    elif mp == 8:
        diam = _thompson(species, rho, temperature, qqw, qqi, qqr, qqs, qqg, qqnr, qqni)
    elif mp == 11:
        diam = _gfdl(species, temperature, qqw, qqi, qqr, qqs, qqg)
    elif mp in (5, 85, 95):
        diam = _ferrier_effr(species, rho, qqr, qqs, f_rimef, nlice, nrain)
    else:
        diam = np.zeros(rho.shape, dtype=np.float64)
    return 0.5 * diam


def column_water_content(mixing_ratio, interface_pressure_pa):
    """Grid-mean hydrometeor content, ``max(0, q * dp / g)`` kg m^-2.

    ``interface_pressure_pa`` is top-down, shape ``(..., n_layers + 1)``,
    in pascals. ``mixing_ratio`` is shape ``(..., n_layers)``. This is the
    ``dpovg`` factor in ``CALRAD``.
    """
    q = np.asarray(mixing_ratio, dtype=np.float64)
    pint = np.asarray(interface_pressure_pa, dtype=np.float64)
    dp = np.diff(pint, axis=-1)
    return np.maximum(0.0, q * dp / G)


def species_for_microphysics(imp_physics: int) -> tuple[str, ...]:
    """Cloud-type list ``CALRAD`` allocates for ``imp_physics``.

    An empty tuple means UPP sets ``n_clouds = 0`` and skips condensate.
    Order is the CRTM cloud index order in that routine.
    """
    mp = int(imp_physics)
    if mp in (98, 99):
        return ("water", "ice")
    if mp in (5, 85, 95):
        return ("water", "ice", "rain", "snow", "graupel", "hail")
    if mp in (8, 6, 2, 28, 11):
        return ("water", "ice", "rain", "snow", "graupel")
    return ()


@dataclass
class Hydrometeors:
    """Mixing ratios and number concentrations on the model grid.

    Names follow ``CALRAD``. Mixing ratios are kg kg^-1. Omitted arrays
    are treated as zero. Shapes match the temperature field passed to
    :func:`crtm_py.atmosphere.atmosphere_from_upp`.
    """

    qqw: np.ndarray | None = None
    qqi: np.ndarray | None = None
    qqr: np.ndarray | None = None
    qqs: np.ndarray | None = None
    qqg: np.ndarray | None = None
    f_rimef: np.ndarray | None = None
    nlice: np.ndarray | None = None
    nrain: np.ndarray | None = None
    qqnr: np.ndarray | None = None
    qqni: np.ndarray | None = None
