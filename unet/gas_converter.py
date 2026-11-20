"""Simple gas converter (no flow, no DM/stars conditioning): SIDM gas = CDM gas + mean change + calibrated random noise.
The mean converter beat the flow for the DM main halo; the gas change is smooth, weakly tied to the DM, and its core scatter
is run-to-run randomness, so a mean + noise model may be enough. Fitted on the train split of data/pairs_gas.npz
(log10 max(Sigma_gas, 1e11), 96x96 crop), Delta = log SIDM - log CDM:
  mean   mu(R; sigma, z) + beta(R; sigma) (log M200 - 14.3): ring-mean Delta per pair, least squares per ring and sigma
         (z as categorical offsets, linear in log M200)
  radial each pair's ring-mean residual profile is drawn from its train distribution: per sigma, the PCA of the residual
         profiles (rings 0-45) with NPC Gaussian components
  pixel  residual about the ring mean: Gaussian random field (white noise smoothed with a Gaussian of L px, L from the
         lag-1 autocorrelation) scaled to the train rms(R; sigma)
Draws for the test split -> runs/gas_conv/draws.npz (same format as gas_flow_draws.py; dm = "conv").
Being multiplicative on the CDM gas it is scale-free by construction (cf. gas_scale_check.py).
usage: python gas_converter.py [--ndraw 19]
"""
import os, argparse
import numpy as np
from scipy.ndimage import gaussian_filter
import dm_common as C, two_scale as T2
ap = argparse.ArgumentParser(); ap.add_argument("--ndraw", type=int, default=19); ap.add_argument("--npc", type=int, default=4)
ap.add_argument("--lin", action="store_true", help="add a per-ring linear term in the pixel Delta DM and Delta stars (about their ring means); test draws use the chain (DM-flow draw d, stars-chain draw d) -> runs/gas_convlin")
a = ap.parse_args(); CL = lambda v: np.clip(v, -1, 1)
rng = np.random.default_rng(3); NR = 46; RB = T2.RB; RBc0 = np.minimum(RB, NR - 1); cnt = np.bincount(RB.ravel(), minlength=48)[:NR]; IN = RB < NR
ringm = lambda d: np.stack([np.bincount(RB[IN], v[IN], NR) for v in d]) / cnt
D = np.load(f"{C.HERE}/data/pairs_gas.npz", allow_pickle=True)
xi, sig, split, zz, lm, ok = D["xi"], D["sigma"], D["split"], D["redshift"], D["log10_M200"], D["ok"] & (np.abs(D["shift"]).max(1) <= 2)
tr = np.where(ok & (sig > 0) & (split[xi] == "train"))[0]; ZS = np.unique(zz)
X = D["x"]; Y = D["y"]
P = {}
for s in C.SIG:
    idx = tr[sig[tr] == s]; dl = (Y[idx][:, 2:98, 2:98] - X[xi[idx]][:, 2:98, 2:98]).astype(np.float64); rp = ringm(dl)
    Z = np.stack([zz[xi[idx]] == z for z in ZS], 1).astype(float); A = np.column_stack([Z, lm[xi[idx]] - 14.3])
    coef = np.linalg.lstsq(A, rp, rcond=None)[0]                      # (nz + 1, NR)
    res = rp - A @ coef; U, Sv, Vt = np.linalg.svd(res - res.mean(0), full_matrices=False)
    pc = Vt[:a.npc]; pcsd = Sv[:a.npc] / np.sqrt(len(res))
    pix = dl - rp[:, np.minimum(RB, NR - 1)]  # residual about the ring mean
    pix[:, ~IN] = 0; beta = np.zeros((NR, 2))
    if a.lin:
        cy = D["cy"][idx][:, :, 2:98, 2:98].astype(np.float64); cy[:, 1] = CL(cy[:, 1])
        q = np.stack([c - ringm(c)[:, RBc0] for c in (cy[:, 0], cy[:, 1])], -1)   # (n, 96, 96, 2) about ring means
        for r in range(NR):
            w = RB == r; Q = q[:, w].reshape(-1, 2); beta[r] = np.linalg.lstsq(Q, pix[:, w].ravel(), rcond=None)[0]
        pix = pix - np.einsum("nijc,ijc->nij", q, beta[RBc0])
        pix[:, ~IN] = 0
    rms = np.sqrt(ringm(pix ** 2).mean(0))
    c1 = np.mean([np.corrcoef(p[IN & (np.roll(RB, 1, 1) < NR)].ravel(), np.roll(p, 1, 1)[IN & (np.roll(RB, 1, 1) < NR)].ravel())[0, 1] for p in pix[:300]])
    L = np.sqrt(-1 / (4 * np.log(max(c1, 1e-3))))
    if a.lin: print(f"  SIDM{s:g} beta(DM, stars) at R=0/100/300 kpc: {np.round(beta[[0, 5, 15]], 3).tolist()}")
    P[s] = dict(beta=beta, coef=coef, pc=pc, pcsd=pcsd, rms=rms, L=L, res_mean=res.mean(0))
    print(f"SIDM{s:g}: {len(idx)} train pairs; mean core Delta (R<40 kpc) {coef[:len(ZS), :2].mean():+.3f}; per-pair radial residual sd (core) "
          f"{res[:, :2].mean(1).std():.3f}, PC sd {np.round(pcsd, 3)}; pixel rms core/100kpc/500kpc {rms[0]:.3f}/{rms[5]:.3f}/{rms[25]:.3f}; lag-1 corr {c1:.2f} -> L {L:.2f} px", flush=True)

S, _ = C.load(); te = S["test"]; k = te["k"]; xg = X[xi[k]][:, 2:98, 2:98].astype(np.float32)
out = np.empty((a.ndraw, len(k), C.N, C.N), np.float16); RBc = np.minimum(RB, NR - 1)
if a.lin:
    dmh = np.load(f"{C.HERE}/runs/dm_flow_hyb/draws.npz")["draws"]; sth = np.load(f"{C.HERE}/runs/stars_hyb/draws.npz")["draws"]
    xst = D["cx"][xi[k], 1, 2:98, 2:98].astype(np.float32); cdm = te["c"]
for j in range(len(k)):
    s = round(float(te["sigma"][j]), 3); p = P[s]; zi = np.argmin(np.abs(ZS - zz[xi[k[j]]]))
    mu = p["coef"][zi] + p["coef"][-1] * (lm[xi[k[j]]] - 14.3) + p["res_mean"]
    for d in range(a.ndraw):
        prof = mu + (rng.standard_normal(len(p["pcsd"])) * p["pcsd"]) @ p["pc"]
        f = gaussian_filter(rng.standard_normal((C.N, C.N)), p["L"]); f /= f.std()
        lt = 0
        if a.lin:
            qd = [v - ringm(v[None])[0][RBc] for v in (dmh[d % len(dmh), j].astype(np.float64) - cdm[j], CL(sth[d % len(sth), j].astype(np.float64) - xst[j]))]
            lt = qd[0] * p["beta"][RBc, 0] + qd[1] * p["beta"][RBc, 1]
        out[d, j] = np.maximum(xg[j] + prof[RBc] + lt + f * p["rms"][RBc], 11.0)  # hard floor
od = f"{C.HERE}/runs/gas_conv{'lin' if a.lin else ''}"; os.makedirs(od, exist_ok=True)
np.savez(f"{od}/draws.npz", draws=out, k=k, name="gas_converter", epoch=-1, dm="convlin" if a.lin else "conv"); print("wrote", od + "/draws.npz")
