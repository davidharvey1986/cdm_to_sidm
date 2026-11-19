"""Post-hoc shift + stretch of the hybrid flow draws, per sigma and radius, fitted on the validation split.

draw  = mu + e                                  (mu = two-scale converter mean, e = flow residual sample)
cal   = mu + tbar(sigma, R) + a(sigma, R) * (e - ebar(sigma, R))
  tbar = mean of the true residual t = s - mu in the ring, ebar = mean of e in the ring (over maps, pixels, draws),
  a    = rms(t - tbar) / rms(e - ebar) in the ring.
So in every 1-px ring the calibrated draws have the true mean and the true one-draw spread about mu; the spatial
structure of each draw is kept (one number per ring per sigma, applied to every map).
Fitted on the val split (held out from the flow weights; used only for checkpoint choice), applied to the test split.

in : runs/dm_flow_hyb/draws_val.npz, runs/dm_flow_hyb/draws.npz (test)   (dm_flow_draws.py --hybrid [--split val])
out: runs/dm_flow_hyb_cal/draws.npz, runs/dm_flow_hyb_cal/calibration.json, figures/dm_flow_calibration.png
usage: python calibrate_dm_flow.py [--rmax 48]

RESULT (19 Nov 2026): NEGATIVE, not adopted; flow_hyb stays uncalibrated.
  Fixed SIDM0.1 inner scatter (0.065 vs 0.062 true, was 0.074), but on the test split the core overshoots at SIDM0.3/1
  (core mean -0.048 / -0.093 vs true -0.041 / -0.081; uncalibrated -0.039 / -0.083), and the ~3% stretch fitted at
  all R over-disperses the test draws (PIT hump at SIDM0.1/0.3, SIDM1 peaks 12.3 -> 13.1).
  Cause: val and test disagree on the core offset by ~0.01 dex, more than the bias being corrected, so the test-split
  core "bias" is split-to-split (cluster sample) noise. Bootstrap bands (~0.002) understate this: maps of one cluster
  (projections, sigma) are not independent.
"""
import os, json, argparse
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import dm_common as C, two_scale as T2
ap = argparse.ArgumentParser(); ap.add_argument("--rmax", type=int, default=48, help="last ring index; outer pixels share it")
ap.add_argument("--nboot", type=int, default=50); a_ = ap.parse_args()
S, _ = C.load(); RI = np.minimum(T2.R.astype(int), a_.rmax); NR = a_.rmax + 1
cnt = np.bincount(RI.ravel(), minlength=NR)


def ring_stats(x):
    """per-map ring sums of x and x^2: x (..., n, 96, 96) -> (n, NR) sums, summed over any leading draw axis"""
    x = x.reshape(-1, *x.shape[-3:]) if x.ndim == 4 else x[None]
    s1 = np.zeros((x.shape[1], NR)); s2 = np.zeros((x.shape[1], NR))
    for d in range(x.shape[0]):
        xd = x[d].reshape(x.shape[1], -1).astype(np.float64)
        for j in range(x.shape[1]):
            s1[j] += np.bincount(RI.ravel(), xd[j], NR); s2[j] += np.bincount(RI.ravel(), xd[j] ** 2, NR)
    return s1, s2, x.shape[0]


def fit(t1, t2, e1, e2, nd, idx):
    n = len(idx) * cnt
    tb = t1[idx].sum(0) / n; eb = e1[idx].sum(0) / (n * nd)
    tv = t2[idx].sum(0) / n - tb ** 2; ev = e2[idx].sum(0) / (n * nd) - eb ** 2
    return tb, eb, np.sqrt(tv / ev)


V = S["val"]; dv = np.load(f"{C.HERE}/runs/dm_flow_hyb/draws_val.npz"); assert (dv["k"] == V["k"]).all()
rng = np.random.default_rng(0); cal = {}
for s in C.SIG:
    m = np.where(V["sigma"] == s)[0]; mu = V["mu"][m]
    t1, t2, _ = ring_stats(V["s"][m] - mu)
    e1, e2, nd = ring_stats(dv["draws"][:, m].astype(np.float32) - mu[None])
    tb, eb, a = fit(t1, t2, e1, e2, nd, np.arange(len(m)))
    bs = np.array([np.concatenate(fit(t1, t2, e1, e2, nd, rng.integers(0, len(m), len(m)))) for _ in range(a_.nboot)])
    cal[str(s)] = dict(n=len(m), tbar=tb.tolist(), ebar=eb.tolist(), a=a.tolist(), b=(tb - eb).tolist(),
                       b_err=(bs[:, :NR] - bs[:, NR:2 * NR]).std(0).tolist(),
                       a_err=bs[:, 2 * NR:].std(0).tolist())
    print(f"sigma {s}: {len(m)} val maps; core (R<1.5px) b = {tb[0] - eb[0]:+.4f}, a = {a[0]:.3f}; "
          f"R=1-2px b = {tb[1] - eb[1]:+.4f}, a = {a[1]:.3f}; R>10px median a = {np.median(a[10:]):.3f}")

T = S["test"]; dt = np.load(f"{C.HERE}/runs/dm_flow_hyb/draws.npz"); assert (dt["k"] == T["k"]).all()
out = np.empty_like(dt["draws"])
for s in C.SIG:
    m = np.where(T["sigma"] == s)[0]; c = cal[str(s)]
    tb, eb, a = (np.asarray(c[k], np.float32)[RI] for k in ("tbar", "ebar", "a"))
    e = dt["draws"][:, m].astype(np.float32) - T["mu"][m][None]
    out[:, m] = (T["mu"][m][None] + tb + a * (e - eb)).astype(np.float16)
od = f"{C.HERE}/runs/dm_flow_hyb_cal"; os.makedirs(od, exist_ok=True)
np.savez(f"{od}/draws.npz", draws=out, k=T["k"], name="flowA_dm_hyb+cal", T=float(dt["T"]), hybrid=True, split="test")
json.dump(dict(rmax=a_.rmax, px_kpc=20, fitted_on="val", sigma=cal), open(f"{od}/calibration.json", "w"), indent=1)
print("wrote", od)

rk = (np.arange(NR) + 0.5) * 20; COL = {0.1: "#3a6fb0", 0.3: "#2a9d8f", 1.0: "#d1495b"}
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
for s in C.SIG:
    c = cal[str(s)]
    for i, (k, ke) in enumerate((("b", "b_err"), ("a", "a_err"))):
        v, ve = np.array(c[k]), np.array(c[ke])
        ax[i].fill_between(rk, v - ve, v + ve, color=COL[s], alpha=0.25, lw=0)
        ax[i].plot(rk, v, color=COL[s], lw=2, label=f"SIDM{s:g}")
ax[0].axhline(0, color="0.5", lw=0.8, ls=":"); ax[1].axhline(1, color="0.5", lw=0.8, ls=":")
ax[0].set_ylabel("shift b = mean(true) − mean(draws) [dex]"); ax[1].set_ylabel("stretch a = rms(true) / rms(draws)")
for x in ax: x.set_xscale("log"); x.set_xlabel("R [kpc]"); x.legend()
ax[0].set_title("shift (band = bootstrap 1σ over val maps)"); ax[1].set_title("stretch")
fig.suptitle("Hybrid flow calibration, fitted on the validation split", fontsize=13)
fig.savefig(f"{C.HERE}/figures/dm_flow_calibration.png", dpi=80); print("wrote figures/dm_flow_calibration.png")
