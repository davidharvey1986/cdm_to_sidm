"""Averaged residual over all BAHAMAS clusters (|shift| <= 2; darkskies-trained converter, so every BAHAMAS cluster is
out of sample): mean maps of true - CDM and true - converted per sigma, azimuthal mean residual with standard errors,
and per-pixel rms residual vs radius (before / after). -> figures/resid_avg_bahamas.png"""
import os, json
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import two_scale as T2
HERE = os.path.dirname(os.path.abspath(__file__))
D = np.load(f"{HERE}/data/pairs.npz", allow_pickle=True)
x, y, xi, sig, sh, zz = (D[k] for k in ("x", "y", "xi", "sigma", "shift", "redshift"))
import sys; TAG = sys.argv[1] if len(sys.argv) > 1 else "bahamas"
J = json.load(open(f"{HERE}/runs/two_scale_{TAG}_nocnn/template.json")); SIG = (0.1, 0.3, 1.0)
spl = D["split"]
r = np.hypot(*np.indices((96, 96)) - 47.5); rb = r.astype(int); cnt = np.bincount(rb.ravel()); R = np.arange(len(cnt)) * 20 + 10
prof = lambda m: np.bincount(rb.ravel(), m.ravel()) / cnt
res = {}
for s in SIG:
    k = np.where((np.abs(sh).max(1) <= 2) & (sig == s) & ((spl[xi] == "test") if TAG == "bahamas" else True))[0]; R0s, R1s, P0, P1, Q0, Q1 = 0, 0, [], [], 0, 0
    for b in range(0, len(k), 200):
        kk = k[b:b + 200]; c = x[xi[kk], 2:98, 2:98].astype(float); t = y[kk, 2:98, 2:98].astype(float)
        Lc = T2.smooth(c); Sc = c - Lc; n = len(kk)
        cv = c + T2.predict_large(Lc, np.full(n, s), zz[xi[kk]].astype(float), J["template"]) + T2.apply_small(Sc, np.full(n, s), J["small"]) - Sc
        a0, a1 = t - c, t - cv; R0s = R0s + a0.sum(0); R1s = R1s + a1.sum(0); Q0 = Q0 + (a0 ** 2).sum(0); Q1 = Q1 + (a1 ** 2).sum(0)
        P0 += [prof(m) for m in a0]; P1 += [prof(m) for m in a1]
    n = len(k); res[s] = dict(n=n, m0=R0s / n, m1=R1s / n, p0=np.array(P0), p1=np.array(P1), rms0=np.sqrt(prof(Q0 / n)), rms1=np.sqrt(prof(Q1 / n)))
    print(s, n, "centre r<3: %.3f -> %.3f" % (res[s]["m0"][r < 3].mean(), res[s]["m1"][r < 3].mean()), flush=True)
fig, ax = plt.subplots(3, 4, figsize=(22, 15), constrained_layout=True)
kw = dict(origin="lower", interpolation="nearest", cmap="RdBu_r", vmin=-0.05, vmax=0.05, extent=[-0.96, 0.96, -0.96, 0.96])
for i, s in enumerate(SIG):
    q = res[s]; a = ax[i]
    for j, (m, t) in enumerate([(q["m0"], "mean true − CDM  (no conversion)"), (q["m1"], "mean true − converted")]):
        im = a[j].imshow(m, **kw); a[j].set_xticks([]); a[j].set_yticks([])
        a[j].set_title(f"{t}\ncentre (r < 3 px) {m[r < 3].mean():+.3f} dex", fontsize=11)
    fig.colorbar(im, ax=a[1], shrink=0.8, label="dex"); a[0].set_ylabel(f"SIDM{s:g}  ({q['n']} maps)", fontsize=14)
    for p, cl, lab in ((q["p0"], "0.5", "true − CDM"), (q["p1"], "#c0392b", "true − converted")):
        mu, se = p.mean(0), p.std(0) / np.sqrt(len(p))
        a[2].plot(R, mu, color=cl, lw=2, label=lab); a[2].fill_between(R, mu - se, mu + se, color=cl, alpha=0.25, lw=0)
    a[2].axhline(0, color="k", lw=0.6); a[2].set_xscale("log"); a[2].set_ylim(-0.12, 0.03); a[2].set_xlabel("R [kpc]"); a[2].set_ylabel("mean residual [dex]")
    a[3].plot(R, q["rms0"], color="0.5", lw=2, label="true − CDM"); a[3].plot(R, q["rms1"], color="#c0392b", lw=2, label="true − converted")
    a[3].set_xscale("log"); a[3].set_ylim(0, 0.3); a[3].set_xlabel("R [kpc]"); a[3].set_ylabel("per-pixel rms residual [dex]")
    if i == 0:
        a[2].set_title("azimuthal mean residual (±1 s.e.)", fontsize=11); a[3].set_title("per-pixel rms residual", fontsize=11)
        a[2].legend(fontsize=10); a[3].legend(fontsize=10)
fig.suptitle(f"BAHAMAS: residual averaged over clusters, before and after conversion ({TAG}-trained converter{', BAHAMAS test clusters only' if TAG == 'bahamas' else ''})", fontsize=15)
out = f"{HERE}/figures/resid_avg_bahamas_{TAG}.png"; fig.savefig(out, dpi=80); print(out)
