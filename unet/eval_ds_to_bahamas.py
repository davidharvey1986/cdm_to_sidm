"""Cross-suite test, darkskies -> BAHAMAS: models trained only on darkskies (sigma 0.01-0.2), scored on the
BAHAMAS test split (sigma 0.1 inside the darkskies range, 0.3 and 1 extrapolated).

Methods (all predict delta = log10 SIDM - log10 CDM on the central 96x96 px):
  zero        no change
  powerlaw    radial delta profile from darkskies train medians, delta(r, sigma) = A(r) sigma^p
              (p fitted on the core r < 2 px, A(r) per radius bin); no mass/z dependence
  ds_mean     flow A trained on darkskies (--ds), mean of --n samples;  ds_sample: one sample
  bah_mean    flow A trained on BAHAMAS (--bah), the in-suite ceiling;   bah_sample: one sample
Outputs runs/ds_to_bahamas/{summary.json, results.png}
usage: python3 eval_ds_to_bahamas.py [--ds flowA_ds] [--bah flowA] [--n 8] [--split test]
"""
import os, json, argparse
import numpy as np, torch, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from flows import Flow, cond_vec, mark_initialized

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--ds", default="flowA_ds"); ap.add_argument("--bah", default="flowA")
ap.add_argument("--n", type=int, default=8); ap.add_argument("--split", default="test")
ap.add_argument("--out", default="ds_to_bahamas")
a = ap.parse_args()
dev = "mps" if torch.backends.mps.is_available() else "cpu"; torch.manual_seed(0)
OUT = f"{HERE}/runs/{a.out}"; os.makedirs(OUT, exist_ok=True)


def load_flow(name):
    ck = torch.load(f"{HERE}/runs/{name}/model.pt", map_location=dev, weights_only=False)
    m = Flow(**ck["arch"]).to(dev); m.load_state_dict(ck["state"]); mark_initialized(m); m.eval()
    m.norm, m.epoch = ck["norm"], ck["epoch"]; return m


FD, FB = load_flow(a.ds), load_flow(a.bah)
C0, C1 = 2, 98; N = 96
yy, xx = np.indices((N, N)); r = np.hypot(yy - 47.5, xx - 47.5); rb = np.minimum(r.astype(int), 47); CORE = r < 2
prof = lambda d: np.stack([np.bincount(rb.ravel(), m.ravel(), 48) for m in d]) / np.bincount(rb.ravel(), minlength=48)
core = lambda d: (d * CORE).sum((-2, -1)) / CORE.sum()

# ---- darkskies train set -> power-law radial template
S_ = np.load(f"{HERE}/data/darkskies_pairs.npz", allow_pickle=True)
dk = (np.abs(S_["shift"]).max(1) <= 2) & (S_["split"][S_["xi"]] == "train")
dd = (S_["y"][dk] - S_["x"][S_["xi"][dk]])[:, C0:C1, C0:C1]; ds_sig = S_["sigma"][dk]
DSIG = np.unique(ds_sig)
med_prof = np.stack([np.median(prof(dd[ds_sig == s]), 0) for s in DSIG])  # (4, 48)
med_core = np.array([np.median(core(dd[ds_sig == s])) for s in DSIG])
ps = np.linspace(0.1, 1.5, 281)
err = [np.sum((med_core - (med_core @ DSIG ** p) / (DSIG ** p @ DSIG ** p) * DSIG ** p) ** 2) for p in ps]
P = float(ps[np.argmin(err)]); A_r = (med_prof.T @ DSIG ** P) / (DSIG ** P @ DSIG ** P)
A_core = float(med_core @ DSIG ** P / (DSIG ** P @ DSIG ** P))
print(f"darkskies core medians {dict(zip(DSIG.tolist(), np.round(med_core, 4).tolist()))}; power law core = {A_core:.4f} sigma^{P:.2f}")

# ---- BAHAMAS evaluation pairs
D = np.load(f"{HERE}/data/pairs.npz", allow_pickle=True)
x, y, xi, sig, sh, spl, zz, M = (D[k] for k in ("x", "y", "xi", "sigma", "shift", "split", "redshift", "log10_M200"))
x, y = x[:, C0:C1, C0:C1], y[:, C0:C1, C0:C1]
ev = np.where((np.abs(sh).max(1) <= 2) & (sig > 0) & (spl[xi] == a.split))[0]
dt = (y[ev] - x[xi[ev]]).astype(np.float64); S = sig[ev]; Z = zz[xi[ev]]; MM = M[xi[ev]]
print(f"{len(ev)} BAHAMAS {a.split} pairs; ds model epoch {FD.epoch}, bah model epoch {FB.epoch}")
T = lambda v: torch.tensor(np.ascontiguousarray(v), dtype=torch.float32, device=dev)[:, None]


def run_flow(F):
    mu, sd, DS = F.norm["mu"], F.norm["sd"], F.norm["dscale"]
    out = np.empty((len(ev), a.n, N, N), np.float32); nll = np.empty(len(ev))
    with torch.no_grad():
        for s0 in range(0, len(ev), 32):
            k = ev[s0:s0 + 32]; B = len(k)
            cimg = T((x[xi[k]] - mu) / sd); cv = cond_vec(sig[k], zz[xi[k]]).to(dev)
            for j in range(a.n):
                u = F.decode(F.sample_eps(B, dev, H=N), cv, cimg, H=N)[:, 0].cpu().numpy()
                out[s0:s0 + B, j] = np.sinh(np.clip(np.nan_to_num(u), -4, 4)) * DS
            nll[s0:s0 + B] = F.nll(T(np.arcsinh(dt[s0:s0 + B] / DS)), cv, cimg)[0].cpu().numpy()
    return out, nll


sd_, nll_d = run_flow(FD); sb_, nll_b = run_flow(FB)
PRED = dict(zero=np.zeros_like(dt[:, :1, :1]) + 0 * dt[:, :1, :1],
            powerlaw=np.stack([(A_r * s ** P)[rb] for s in S]),
            ds_mean=sd_.mean(1), ds_sample=sd_[:, 0], bah_mean=sb_.mean(1), bah_sample=sb_[:, 0])
PRED["zero"] = np.zeros_like(dt)


def pk(m):
    w = np.hanning(N)[:, None] * np.hanning(N)[None]
    F = np.abs(np.fft.fft2((m - m.mean((-2, -1), keepdims=True)) * w)) ** 2
    kk = np.hypot(*np.meshgrid(np.fft.fftfreq(N), np.fft.fftfreq(N), indexing="ij")); kb = (kk * N).astype(int).ravel()
    return np.stack([np.bincount(kb, f.ravel(), N)[:N // 2] for f in F.reshape(-1, N, N)])


LAM = (24, 8, 4, 2.5); KI = [int(round(N / L)) for L in LAM]
ct = core(dt); tp = prof(dt)
res = dict(powerlaw=dict(p=P, A_core=A_core, darkskies_core_medians=dict(zip(map(str, DSIG), med_core.tolist()))),
           nll_per_px=dict(ds_model=float(nll_d.mean()), bah_model=float(nll_b.mean()),
                           ds_model_on_ds_val=float(np.loadtxt(f"{HERE}/runs/{a.ds}/history.csv", delimiter=",", skiprows=1,
                                                               usecols=3).min()) if os.path.exists(f"{HERE}/runs/{a.ds}/history.csv") else None))
MB = [14.0, 14.3, 14.5, 14.7, 15.0, 15.6]
for s in (0.1, 0.3, 1.0):
    m = S == s; R = dict(n=int(m.sum()), true_core_median=float(np.median(ct[m])), true_core_std=float(ct[m].std()),
                          true_ring=float(np.median(tp[m][:, 3:10].mean(1))))
    pc = pk(x[xi[ev[m]]]); ptrue = np.median(pk(y[ev[m]]) / pc, 0)
    R["power_ratio_true"] = dict(zip(map(str, LAM), ptrue[KI].round(3).tolist()))
    for nm, pr in PRED.items():
        c = core(pr[m]); q = dict(core_median=float(np.median(c)), core_rmse=float(np.sqrt(np.mean((c - ct[m]) ** 2))),
                                   core_std_across=float(c.std()), ring=float(np.median(prof(pr[m])[:, 3:10].mean(1))),
                                   profile_rmse_r_lt_10=float(np.sqrt(np.mean((np.median(prof(pr[m]), 0) - np.median(tp[m], 0))[:10] ** 2))))
        if nm in ("ds_sample", "bah_sample"):
            q["power_ratio"] = dict(zip(map(str, LAM), np.median(pk(x[xi[ev[m]]] + pr[m]) / pc, 0)[KI].round(3).tolist()))
        if nm in ("ds_mean", "bah_mean"):
            smp = sd_ if nm == "ds_mean" else sb_
            cs = core(smp[m]); q["calib_z_std"] = float(((ct[m] - cs.mean(1)) / cs.std(1)).std())
        if nm in ("zero",):
            q = {k_: v for k_, v in q.items() if k_ in ("core_rmse",)}
        q["core_median_by_M"] = [float(np.median(c[(MM[m] >= lo) & (MM[m] < hi)])) if ((MM[m] >= lo) & (MM[m] < hi)).sum() > 8 else None
                                 for lo, hi in zip(MB[:-1], MB[1:])]
        R[nm] = q
    R["true_core_median_by_M"] = [float(np.median(ct[m][(MM[m] >= lo) & (MM[m] < hi)])) if ((MM[m] >= lo) & (MM[m] < hi)).sum() > 8 else None
                                  for lo, hi in zip(MB[:-1], MB[1:])]
    res[str(s)] = R
json.dump(res, open(f"{OUT}/summary.json", "w"), indent=1)

print(f"NLL/px of true BAHAMAS delta: ds model {res['nll_per_px']['ds_model']:.3f} (its darkskies val {res['nll_per_px']['ds_model_on_ds_val']}), bah model {res['nll_per_px']['bah_model']:.3f}")
for s in ("0.1", "0.3", "1.0"):
    R = res[s]
    print(f"sigma {s} (n={R['n']}): true core {R['true_core_median']:+.3f} (std {R['true_core_std']:.3f}), ring {R['true_ring']:+.3f}; zero RMSE {R['zero']['core_rmse']:.4f}")
    for nm in ("powerlaw", "ds_mean", "ds_sample", "bah_mean", "bah_sample"):
        q = R[nm]
        extra = (f" calib {q['calib_z_std']:.2f}" if "calib_z_std" in q else "") + (f" P {list(q['power_ratio'].values())}" if "power_ratio" in q else "")
        print(f"   {nm:10s} core {q['core_median']:+.3f} RMSE {q['core_rmse']:.4f} std {q['core_std_across']:.3f} ring {q['ring']:+.3f} profRMSE {q['profile_rmse_r_lt_10']:.4f}{extra}")
    print(f"   power true {list(R['power_ratio_true'].values())}")
    print(f"   core by M {MB}: true {np.round(np.array(R['true_core_median_by_M'], float), 3)} powerlaw {np.round(np.array(R['powerlaw']['core_median_by_M'], float), 3)} ds {np.round(np.array(R['ds_mean']['core_median_by_M'], float), 3)}")

# ---- figure
col = {"true": "k", "powerlaw": "#2a9d8f", "ds_mean": "#c0392b", "bah_mean": "#3a6fb0"}
fig, ax = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
rr = np.arange(48) + 0.5; rr[0] = 0.35
for i, s in enumerate((0.1, 0.3, 1.0)):
    m = S == s; g = ax[0, i]
    g.fill_between(rr, *np.percentile(tp[m], [16, 84], 0), color="0.8", lw=0, label="true 16-84%")
    g.plot(rr, np.median(tp[m], 0), color="k", lw=2, label="true BAHAMAS")
    for nm, ls in (("powerlaw", ":"), ("ds_mean", "--"), ("bah_mean", "-.")):
        g.plot(rr, np.median(prof(PRED[nm][m]), 0), color=col[nm], ls=ls, lw=1.8,
               label={"powerlaw": "darkskies power law", "ds_mean": "flow A (darkskies)", "bah_mean": "flow A (BAHAMAS, ceiling)"}[nm])
    g.axhline(0, color="0.5", lw=0.7); g.set_xscale("log"); g.set_title(f"BAHAMAS σ={s}" + ("" if s == 0.1 else "  (extrapolated)"))
    g.set_xlabel("r [px]"); g.set_ylabel("median azimuthal Δ [dex]"); g.set_ylim(-0.2, 0.06)
ax[0, 0].legend(fontsize=7)
g = ax[1, 0]; sg = np.logspace(-2.2, 0.1, 50)
g.plot(sg, A_core * sg ** P, color=col["powerlaw"], ls=":", label=f"power law {A_core:.3f} σ^{P:.2f} (darkskies fit)")
g.plot(DSIG, med_core, "o", color=col["powerlaw"], label="darkskies true (train)")
g.plot([0.1, 0.3, 1.0], [res[s]["true_core_median"] for s in ("0.1", "0.3", "1.0")], "ks", label="BAHAMAS true")
g.plot([0.1, 0.3, 1.0], [res[s]["ds_mean"]["core_median"] for s in ("0.1", "0.3", "1.0")], "^", color=col["ds_mean"], label="flow A (darkskies)")
g.plot([0.1, 0.3, 1.0], [res[s]["bah_mean"]["core_median"] for s in ("0.1", "0.3", "1.0")], "v", color=col["bah_mean"], label="flow A (BAHAMAS)")
g.axvspan(DSIG.min(), DSIG.max(), color="#2a9d8f", alpha=0.07)
g.set_xscale("log"); g.set_xlabel("σ [cm²/g]"); g.set_ylabel("median core Δ (r<2 px) [dex]"); g.legend(fontsize=7); g.set_title("core response vs σ (shaded: darkskies training range)")
g = ax[1, 1]; mc = 0.5 * (np.array(MB[:-1]) + np.array(MB[1:]))
for s, c_ in zip(("0.1", "0.3", "1.0"), ("#e9a03b", "#8e44ad", "#c0392b")):
    g.plot(mc, np.array(res[s]["true_core_median_by_M"], float), "-o", color=c_, label=f"true σ={s}")
    g.plot(mc, np.array(res[s]["ds_mean"]["core_median_by_M"], float), "--^", color=c_, label=f"flow A (ds) σ={s}")
    g.plot(mc, np.array(res[s]["powerlaw"]["core_median_by_M"], float), ":", color=c_)
g.set_xlabel("log10 M200"); g.set_ylabel("median core Δ"); g.legend(fontsize=7, ncol=2); g.set_title("mass dependence (dotted: power law, no mass)")
g = ax[1, 2]
for s, c_ in zip(("0.1", "0.3", "1.0"), ("#e9a03b", "#8e44ad", "#c0392b")):
    g.plot(LAM, list(res[s]["power_ratio_true"].values()), "-o", color=c_, label=f"true σ={s}")
    g.plot(LAM, list(res[s]["ds_sample"]["power_ratio"].values()), "--^", color=c_, label=f"flow A (ds) σ={s}")
g.set_xscale("log"); g.set_xlabel("wavelength [px]"); g.set_ylabel("P(SIDM)/P(CDM)"); g.axhline(1, color="0.5", lw=0.7)
g.legend(fontsize=7, ncol=2); g.set_title("power ratio")
fig.suptitle(f"darkskies → BAHAMAS: trained on darkskies only, tested on BAHAMAS {a.split} ({len(ev)} pairs)")
fig.savefig(f"{OUT}/results.png", dpi=105); print("wrote", OUT)
