"""Projected halo shape: iterative mass-weighted second moments inside an ellipse of semi-major axis R about the map centre.
shapes(L) -> eps = 1 - b/a (n, len(RK)) for maps L (n, H, H) of log10 Sigma (linear Sigma is the weight); 20 kpc pixels.
Start circular; update axis ratio q = sqrt(l_min / l_max) and position angle from the moment tensor; NIT iterations max.
"""
import numpy as np
RK = np.array([40, 60, 100, 150, 200, 300, 400, 600, 900]); NIT, NDRAW = 12, 4


def shapes(L, chunk=256, px=20.0, lin=False):
    """L (n, H, H) log10 Sigma (or linear Sigma if lin) -> eps (n, len(RK))"""
    H = L.shape[-1]; Y, X = np.indices((H, H)) - (H - 1) / 2; X, Y = X * px, Y * px
    out = np.empty((len(L), len(RK)))
    for b in range(0, len(L), chunk):
        S = L[b:b + chunk].astype(np.float64) if lin else 10 ** L[b:b + chunk].astype(np.float64); n = len(S)
        for j, R in enumerate(RK):
            q, ph = np.ones(n), np.zeros(n)
            for _ in range(NIT):
                c, s = np.cos(ph)[:, None, None], np.sin(ph)[:, None, None]
                u, v = c * X + s * Y, -s * X + c * Y
                w = S * ((u ** 2 + (v / q[:, None, None]) ** 2) < R ** 2)
                m = w.sum((1, 2)); Ixx, Iyy, Ixy = ((w * a).sum((1, 2)) / m for a in (X * X, Y * Y, X * Y))
                tr, det = Ixx + Iyy, Ixx * Iyy - Ixy ** 2; d = np.sqrt(np.maximum(tr ** 2 / 4 - det, 0))
                l1, l2 = tr / 2 + d, np.maximum(tr / 2 - d, 1e-12)
                q_new, ph = np.sqrt(l2 / l1), 0.5 * np.arctan2(2 * Ixy, Ixx - Iyy)
                if np.max(np.abs(q_new - q)) < 1e-4: q = q_new; break
                q = q_new
            out[b:b + n, j] = 1 - q
    return out


