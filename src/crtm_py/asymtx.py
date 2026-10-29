"""Real asymmetric eigensolver used by CRTM ADA.

``ASYMTX`` (adapted in CRTM from DISORT / EISPACK ``EIGRF``) solves

    A v = λ v

for a real matrix that is not symmetric, under the prior that every
eigenvalue is real. The header in ``CRTM_Utility`` states that explicitly.
It is not a symmetric eigensolver. The returned eigenvalues are unordered.
Each eigenvector is scaled to unit Euclidean length.
"""

from __future__ import annotations

import math

import numpy as np

_C1 = 0.4375
_C2 = 0.5
_C3 = 0.75
_C4 = 0.95
_C5 = 16.0
_C6 = 256.0
_TOL = 1.0e-12


def _sign(magnitude: float, signature: float) -> float:
    return math.copysign(abs(magnitude), signature)


def asymtx(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Return ``(eigenvalues, eigenvectors, ier)`` for one layer matrix.

    ``eigenvectors[:, j]`` belongs to ``eigenvalues[j]``, matching
    ``ASYMTX``. ``ier`` is the Fortran ``IER``: 0 on convergence.
    """
    source = np.asarray(matrix, dtype=np.float64)
    m = source.shape[0]
    if source.shape != (m, m):
        raise ValueError("ASYMTX expects a square matrix")
    if m < 1:
        raise ValueError("ASYMTX expects a positive order")
    a = np.zeros((m + 1, m + 1), dtype=np.float64)
    a[1:, 1:] = source
    ev = np.zeros((m + 1, m + 1), dtype=np.float64)
    val = np.zeros(m + 1, dtype=np.float64)
    wk = np.zeros(4 * m + 8, dtype=np.float64)
    ier = _asymtx_body(a, ev, val, wk, m)
    return val[1:].copy(), ev[1:, 1:].copy(), ier


def _asymtx_body(a, ev, val, wk, m: int) -> int:
    ier = 0
    if m == 1:
        val[1] = a[1, 1]
        ev[1, 1] = 1.0
        return 0
    if m == 2:
        discri = (a[1, 1] - a[2, 2]) ** 2 + 4.0 * a[1, 2] * a[2, 1]
        if discri < 0.0:
            return 1
        sgn = 1.0 if a[1, 1] >= a[2, 2] else -1.0
        val[1] = 0.5 * (a[1, 1] + a[2, 2] + sgn * math.sqrt(discri))
        val[2] = 0.5 * (a[1, 1] + a[2, 2] - sgn * math.sqrt(discri))
        ev[1, 1] = 1.0
        ev[2, 2] = 1.0
        if a[1, 1] == a[2, 2] and (a[2, 1] == 0.0 or a[1, 2] == 0.0):
            rnorm = abs(a[1, 1]) + abs(a[1, 2]) + abs(a[2, 1]) + abs(a[2, 2])
            w = _TOL * rnorm
            ev[2, 1] = a[2, 1] / w
            ev[1, 2] = -a[1, 2] / w
        else:
            ev[2, 1] = a[2, 1] / (val[1] - a[2, 2])
            ev[1, 2] = a[1, 2] / (val[2] - a[1, 1])
        _normalize(ev, m)
        return 0

    for i in range(1, m + 1):
        val[i] = 0.0
        ev[i, 1:] = 0.0
        ev[i, i] = 1.0

    rnorm = 0.0
    low = 1
    high = m

    while True:
        kkk = high
        isolated = False
        for j in range(kkk, 0, -1):
            row = 0.0
            for i in range(1, high + 1):
                if i != j:
                    row += abs(a[j, i])
            if row == 0.0:
                wk[high] = j
                if j != high:
                    for i in range(1, high + 1):
                        a[i, j], a[i, high] = a[i, high], a[i, j]
                    for i in range(low, m + 1):
                        a[j, i], a[high, i] = a[high, i], a[j, i]
                high -= 1
                isolated = True
                break
        if not isolated:
            break

    while True:
        lll = low
        isolated = False
        for j in range(lll, high + 1):
            col = 0.0
            for i in range(low, high + 1):
                if i != j:
                    col += abs(a[i, j])
            if col == 0.0:
                wk[low] = j
                if j != low:
                    for i in range(1, high + 1):
                        a[i, j], a[i, low] = a[i, low], a[i, j]
                    for i in range(low, m + 1):
                        a[j, i], a[low, i] = a[low, i], a[j, i]
                low += 1
                isolated = True
                break
        if not isolated:
            break

    for i in range(low, high + 1):
        wk[i] = 1.0

    while True:
        noconv = False
        for i in range(low, high + 1):
            col = 0.0
            row = 0.0
            for j in range(low, high + 1):
                if j != i:
                    col += abs(a[j, i])
                    row += abs(a[i, j])
            factor = 1.0
            g = row / _C5
            hsum = col + row
            while col < g:
                factor *= _C5
                col *= _C6
            g = row * _C5
            while col >= g:
                factor /= _C5
                col /= _C6
            if (col + row) / factor < _C4 * hsum:
                wk[i] *= factor
                noconv = True
                for j in range(low, m + 1):
                    a[i, j] /= factor
                for j in range(1, high + 1):
                    a[j, i] *= factor
        if not noconv:
            break

    if high - 1 >= low + 1:
        for n in range(low + 1, high):
            h = 0.0
            wk[n + m] = 0.0
            scale = 0.0
            for i in range(n, high + 1):
                scale += abs(a[i, n - 1])
            if scale != 0.0:
                for i in range(high, n - 1, -1):
                    wk[i + m] = a[i, n - 1] / scale
                    h += wk[i + m] ** 2
                g = -_sign(math.sqrt(h), wk[n + m])
                h = h - wk[n + m] * g
                wk[n + m] = wk[n + m] - g
                for j in range(n, m + 1):
                    f = 0.0
                    for i in range(high, n - 1, -1):
                        f += wk[i + m] * a[i, j]
                    for i in range(n, high + 1):
                        a[i, j] -= wk[i + m] * f / h
                for i in range(1, high + 1):
                    f = 0.0
                    for j in range(high, n - 1, -1):
                        f += wk[j + m] * a[i, j]
                    for j in range(n, high + 1):
                        a[i, j] -= wk[j + m] * f / h
                wk[n + m] = scale * wk[n + m]
                a[n, n - 1] = scale * g

        for n in range(high - 2, low - 1, -1):
            n1 = n + 1
            n2 = n + 2
            f = a[n1, n]
            if f != 0.0:
                f = f * wk[n1 + m]
                for i in range(n2, high + 1):
                    wk[i + m] = a[i, n]
                if n1 <= high:
                    for j in range(1, m + 1):
                        g = 0.0
                        for i in range(n1, high + 1):
                            g += wk[i + m] * ev[i, j]
                        g = g / f
                        for i in range(n1, high + 1):
                            ev[i, j] += g * wk[i + m]

    n = 1
    for i in range(1, m + 1):
        for j in range(n, m + 1):
            rnorm += abs(a[i, j])
        n = i
        if i < low or i > high:
            val[i] = a[i, i]
    n = high
    shift = 0.0
    lb = low

    while n >= low:
        inner = 0
        n1 = n - 1
        n2 = n - 2
        while True:
            for i in range(low, n + 1):
                lb = n + low - i
                if lb == low:
                    break
                s = abs(a[lb - 1, lb - 1]) + abs(a[lb, lb])
                if s == 0.0:
                    s = rnorm
                if abs(a[lb, lb - 1]) <= _TOL * s:
                    break
            x = a[n, n]
            if lb == n:
                a[n, n] = x + shift
                val[n] = a[n, n]
                n = n1
                break
            y = a[n1, n1]
            w = a[n, n1] * a[n1, n]
            if lb == n1:
                p = (y - x) * _C2
                q = p**2 + w
                z = math.sqrt(abs(q))
                a[n, n] = x + shift
                x = a[n, n]
                a[n1, n1] = y + shift
                z = p + _sign(z, p)
                val[n1] = x + z
                val[n] = val[n1]
                if z != 0.0:
                    val[n] = x - w / z
                x = a[n, n1]
                r = math.sqrt(x * x + z * z)
                p = x / r
                q = z / r
                for j in range(n1, m + 1):
                    z = a[n1, j]
                    a[n1, j] = q * z + p * a[n, j]
                    a[n, j] = q * a[n, j] - p * z
                for i in range(1, n + 1):
                    z = a[i, n1]
                    a[i, n1] = q * z + p * a[i, n]
                    a[i, n] = q * a[i, n] - p * z
                for i in range(low, high + 1):
                    z = ev[i, n1]
                    ev[i, n1] = q * z + p * ev[i, n]
                    ev[i, n] = q * ev[i, n] - p * z
                n = n2
                break
            if inner == 30:
                _normalize(ev, m)
                return n
            if inner == 10 or inner == 20:
                shift += x
                for i in range(low, n + 1):
                    a[i, i] -= x
                s = abs(a[n, n1]) + abs(a[n1, n2])
                x = _C3 * s
                y = x
                w = -_C1 * s**2
            inner += 1
            i_qr = lb
            for j in range(lb, n2 + 1):
                i_qr = n2 + lb - j
                z = a[i_qr, i_qr]
                r = x - z
                s = y - z
                p = (r * s - w) / a[i_qr + 1, i_qr] + a[i_qr, i_qr + 1]
                q = a[i_qr + 1, i_qr + 1] - z - r - s
                r = a[i_qr + 2, i_qr + 1]
                s = abs(p) + abs(q) + abs(r)
                p /= s
                q /= s
                r /= s
                if i_qr == lb:
                    break
                uu = abs(a[i_qr, i_qr - 1]) * (abs(q) + abs(r))
                vv = abs(p) * (abs(a[i_qr - 1, i_qr - 1]) + abs(z) + abs(a[i_qr + 1, i_qr + 1]))
                if uu <= _TOL * vv:
                    break
            a[i_qr + 2, i_qr] = 0.0
            for j in range(i_qr + 3, n + 1):
                a[j, j - 2] = 0.0
                a[j, j - 3] = 0.0
            ka = i_qr
            while ka <= n1:
                notlas = ka != n1
                if ka == i_qr:
                    s = _sign(math.sqrt(p * p + q * q + r * r), p)
                    if lb != i_qr:
                        a[ka, ka - 1] = -a[ka, ka - 1]
                else:
                    p = a[ka, ka - 1]
                    q = a[ka + 1, ka - 1]
                    r = 0.0
                    if notlas:
                        r = a[ka + 2, ka - 1]
                    x = abs(p) + abs(q) + abs(r)
                    if x == 0.0:
                        ka += 1
                        continue
                    p /= x
                    q /= x
                    r /= x
                    s = _sign(math.sqrt(p * p + q * q + r * r), p)
                    a[ka, ka - 1] = -s * x
                p = p + s
                x = p / s
                y = q / s
                z = r / s
                q = q / p
                r = r / p
                for j in range(ka, m + 1):
                    p = a[ka, j] + q * a[ka + 1, j]
                    if notlas:
                        p = p + r * a[ka + 2, j]
                        a[ka + 2, j] -= p * z
                    a[ka + 1, j] -= p * y
                    a[ka, j] -= p * x
                for ii in range(1, min(n, ka + 3) + 1):
                    p = x * a[ii, ka] + y * a[ii, ka + 1]
                    if notlas:
                        p = p + z * a[ii, ka + 2]
                        a[ii, ka + 2] -= p * r
                    a[ii, ka + 1] -= p * q
                    a[ii, ka] -= p
                for ii in range(low, high + 1):
                    p = x * ev[ii, ka] + y * ev[ii, ka + 1]
                    if notlas:
                        p = p + z * ev[ii, ka + 2]
                        ev[ii, ka + 2] -= p * r
                    ev[ii, ka + 1] -= p * q
                    ev[ii, ka] -= p
                ka += 1

    if rnorm != 0.0:
        for n in range(m, 0, -1):
            n2 = n
            a[n, n] = 1.0
            for i in range(n - 1, 0, -1):
                w = a[i, i] - val[n]
                if w == 0.0:
                    w = _TOL * rnorm
                r = a[i, n]
                for j in range(n2, n):
                    r += a[i, j] * a[j, n]
                a[i, n] = -r / w
                n2 = i
        for i in range(1, m + 1):
            if i < low or i > high:
                for j in range(i, m + 1):
                    ev[i, j] = a[i, j]
        if high != 0:
            for j in range(m, low - 1, -1):
                for i in range(low, high + 1):
                    z = 0.0
                    for n in range(low, min(j, high) + 1):
                        z += ev[i, n] * a[n, j]
                    ev[i, j] = z

    for i in range(low, high + 1):
        for j in range(1, m + 1):
            ev[i, j] *= wk[i]

    for i in range(low - 1, 0, -1):
        j = int(wk[i])
        if i != j:
            for n in range(1, m + 1):
                ev[i, n], ev[j, n] = ev[j, n], ev[i, n]
    for i in range(high + 1, m + 1):
        j = int(wk[i])
        if i != j:
            for n in range(1, m + 1):
                ev[i, n], ev[j, n] = ev[j, n], ev[i, n]
    _normalize(ev, m)
    return ier


def _normalize(ev, m: int) -> None:
    for j in range(1, m + 1):
        norm = math.sqrt(float(np.dot(ev[1:, j], ev[1:, j])))
        if norm != 0.0:
            ev[1:, j] /= norm
