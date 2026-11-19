"""Per-pixel loss weights from the local agreement of substructure between CDM and target.

For each pair in data/pairs.npz (log10 maps, target already re-centred):
  1. high-pass both maps: h = m - Gaussian(m, HP px)  -> subhaloes / small-scale structure
  2. local correlation in a Gaussian window (WIN px, ~5 px across):
       rho = G(hc*ht) / sqrt(G(hc^2) G(ht^2))
  3. structure fraction f = s^2 / (s^2 + s0^2), s^2 = G(hc^2) + G(ht^2), s0^2 = its median over the
     map: f ~ 1 where either map has a subhalo, ~ 0 in smooth regions (core, empty field)
  4. g(rho) = W_MIN for rho <= 0, rising linearly to 1 + A at rho = 1
     w = (1 - f) * 1 + f * g(rho)
So: both maps show the same subhalo -> up to 1+A; only one does, or it moved (dipole) -> W_MIN;
smooth regions and the outer EDGE px keep weight 1. Weights only enter the training loss (the model input is unchanged).
Output: data/agree_weights.npy (P,100,100) float16, aligned with pairs.npz rows.
usage: python3 make_agreement_weights.py
"""
import os
import numpy as np
from scipy.ndimage import gaussian_filter

HERE = os.path.dirname(os.path.abspath(__file__))
HP, WIN, A, W_MIN, EDGE = 3.0, 2.0, 1.0, 0.1, 5


def agreement_weight(c, t):
    hc = c - gaussian_filter(c, HP, mode="reflect")
    ht = t - gaussian_filter(t, HP, mode="reflect")
    G = lambda v: gaussian_filter(v, WIN, mode="reflect")
    vc, vt = G(hc * hc), G(ht * ht)
    rho = G(hc * ht) / np.sqrt(vc * vt + 1e-12)
    s2 = vc + vt
    f = s2 / (s2 + np.median(s2[EDGE:-EDGE, EDGE:-EDGE]))
    # the stamp-edge fall-off is common to both maps and would read as 'agreement': weight 1 there
    f[:EDGE] = f[-EDGE:] = 0; f[:, :EDGE] = f[:, -EDGE:] = 0
    g = np.where(rho <= 0, W_MIN, W_MIN + (1 + A - W_MIN) * rho)
    return (1 - f) + f * g, rho, f


if __name__ == "__main__":
    D = np.load(f"{HERE}/data/pairs.npz", allow_pickle=True)
    x, y, xi = D["x"].astype(np.float64), D["y"], D["xi"]
    W = np.empty(y.shape, np.float16)
    for p in range(len(y)):
        W[p] = agreement_weight(x[xi[p]], y[p].astype(np.float64))[0]
    np.save(f"{HERE}/data/agree_weights.npy", W)
    sig = D["sigma"]
    for s in (0.0, 0.1, 0.3, 1.0):
        w = W[sig == s].astype(np.float32)
        print(f"sigma {s}: mean weight {w.mean():.3f}; fraction of pixels < 0.5: {(w < 0.5).mean():.3f}; > 1.5: {(w > 1.5).mean():.3f}")
    print("wrote data/agree_weights.npy", W.shape)
