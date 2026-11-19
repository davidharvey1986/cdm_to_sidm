"""Apply flow A (trained on BAHAMAS) to the darkskies zooms: CDM -> SIDM0.01/0.05/0.1/0.2.

Pairs from ../halo_database.csv (matched == True, unfixable6 excluded), 20 kpc 100x100 maps,
log10 Sigma; SIDM map re-centred on CDM by integer cross-correlation shift, kept if |shift| <= 2 px
(same cut as training). Crop 2:98. sigma 0.1 is inside training (0.1/0.3/1), 0.2 interpolates,
0.01 and 0.05 extrapolate below the lowest trained sigma.
Outputs runs/darkskies_test/{results.npz, summary.json, profiles.png, examples.png}
usage: python3 test_darkskies.py [--n 8]
"""
import os, sys, json, pickle, argparse
import numpy as np, pandas as pd, torch, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from flows import Flow, cond_vec, mark_initialized

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=8); a = ap.parse_args()
dev = "mps" if torch.backends.mps.is_available() else "cpu"; torch.manual_seed(0); np.random.seed(0)
OUT = f"{HERE}/runs/darkskies_test"; os.makedirs(OUT, exist_ok=True)
ck = torch.load(f"{HERE}/runs/flowA/model.pt", map_location=dev, weights_only=False)
FA = Flow(**ck["arch"]).to(dev); FA.load_state_dict(ck["state"]); mark_initialized(FA); FA.eval()
C0, C1 = ck["norm"]["crop"]; DS, mu, sd = ck["norm"]["dscale"], ck["norm"]["mu"], ck["norm"]["sd"]
SIG = (0.01, 0.05, 0.1, 0.2)

# ---------- pairs
db = pd.read_csv(f"{ROOT}/halo_database.csv")
bad = set("/".join(l.split()[:3]) for l in open(f"{ROOT}/unfixable6.txt") if l.strip().startswith("SIDM"))
db = db[db.matched & db.sidm_exists]
db = db[~db.apply(lambda r: f"{r.sidm_model}/{r.snap}/{r.halo}" in bad, axis=1)]
print("excluded unfixable:", len(bad))
ZMAP = {7: 0.375, 8: 0.25, 9: 0.125, 10: 0.0}


def best_shift(A, B, M=12):
    A, B = A - A.mean(), B - B.mean()
    xc = np.fft.ifft2(np.fft.fft2(A) * np.conj(np.fft.fft2(B))).real
    d = (np.arange(100) + 50) % 100 - 50; ok = np.abs(d) <= M
    xc = np.where(ok[:, None] & ok[None, :], xc, -np.inf)
    i, j = np.unravel_index(np.argmax(xc), xc.shape); return int(d[i]), int(d[j])


load = lambda p: np.log10(np.asarray(pickle.load(open(p, "rb"))[0])[0]).astype(np.float32)
cdm_cache, X, Y, S, Z, H, sh = {}, [], [], [], [], [], []
for r in db.itertuples():
    if r.cdm_path not in cdm_cache:
        cdm_cache[r.cdm_path] = load(r.cdm_path)
    c, t = cdm_cache[r.cdm_path], load(r.sidm_path)
    dy, dx = best_shift(c, t); sh.append(max(abs(dy), abs(dx)))
    X.append(c); Y.append(np.roll(t, (dy, dx), (0, 1))); S.append(r.sigma); Z.append(ZMAP[r.snap])
    H.append(f"{r.halo}_{r.projection}_{r.snap}")
X, Y, S, Z, sh, H = (np.array(v) for v in (X, Y, S, Z, sh, H))
print(f"{len(X)} matched pairs; |shift|<=2: {(sh <= 2).sum()}  ({ {s: int(((S == s) & (sh <= 2)).sum()) for s in SIG} })")
k = (sh <= 2) & np.isfinite(X).all((1, 2)) & np.isfinite(Y).all((1, 2))
X, Y, S, Z, H = X[k][:, C0:C1, C0:C1], Y[k][:, C0:C1, C0:C1], S[k], Z[k], H[k]
dt = (Y - X).astype(np.float64)
N = X.shape[-1]; cc = (N - 1) / 2; yy, xx = np.indices((N, N)); r = np.hypot(yy - cc, xx - cc)
rb = r.astype(int).ravel(); NR = 48; cnt = np.bincount(rb, minlength=80)[:NR]; CORE = r < 2
lprof = lambda m: np.log10(np.bincount(rb, (10.0 ** m).ravel(), 80)[:NR] / cnt)
T = lambda v: torch.tensor(np.ascontiguousarray(v), dtype=torch.float32, device=dev)[:, None]

# ---------- flow A samples, NLL of the true delta at each sigma (incl. BAHAMAS sigmas)
SGRID = (0.01, 0.05, 0.1, 0.2, 0.3, 1.0)
samp = np.empty((len(X), a.n, N, N), np.float32); nll = np.empty((len(X), len(SGRID)))
with torch.no_grad():
    for s0 in range(0, len(X), 32):
        sl = slice(s0, s0 + 32); B = len(X[sl]); cimg = T((X[sl] - mu) / sd)
        cv = cond_vec(S[sl], Z[sl]).to(dev)
        for j in range(a.n):
            u = FA.decode(FA.sample_eps(B, dev, H=N), cv, cimg, H=N)[:, 0].cpu().numpy()
            samp[sl, j] = np.sinh(np.clip(u, -4, 4)) * DS
        ut = T(np.arcsinh(dt[sl] / DS))
        for q, s in enumerate(SGRID):
            nll[sl, q] = FA.nll(ut, cond_vec(np.full(B, s), Z[sl]).to(dev), cimg)[0].cpu().numpy()
        print(f"  {s0 + B}/{len(X)}", flush=True, end="\r")
print()

# ---------- metrics
res = {}
PC = np.stack([lprof(m) for m in X]); PT = np.stack([lprof(m) for m in Y])
PF = np.stack([[lprof(X[i] + samp[i, j]) for j in range(a.n)] for i in range(len(X))])
core_t = (dt * CORE).sum((1, 2)) / CORE.sum(); core_s = (samp * CORE).sum((2, 3)) / CORE.sum()
D = np.load(f"{HERE}/data/pairs.npz", allow_pickle=True)
bx, by, bxi, bs, bsh, bsp = (D[q] for q in ("x", "y", "xi", "sigma", "shift", "split"))
bk = (np.abs(bsh).max(1) <= 2) & (bs > 0) & (bsp[bxi] == "test")
bcore = {s: np.median((((by[bk & (bs == s)] - bx[bxi[bk & (bs == s)]])[:, C0:C1, C0:C1]) * CORE).sum((1, 2)) / CORE.sum())
         for s in (0.1, 0.3, 1.0)}
bcore = {s: float(v) for s, v in bcore.items()}


def pk(m):
    w = np.hanning(N)[:, None] * np.hanning(N)[None]
    F = np.abs(np.fft.fft2((m - m.mean((-2, -1), keepdims=True)) * w)) ** 2
    kk = np.hypot(*np.meshgrid(np.fft.fftfreq(N), np.fft.fftfreq(N), indexing="ij")); kb = (kk * N).astype(int).ravel()
    return np.stack([np.bincount(kb, f.ravel(), N)[:N // 2] for f in F.reshape(-1, N, N)]).reshape(*m.shape[:-2], N // 2)


for s in SIG:
    m = S == s
    ct, cs = core_t[m], core_s[m]
    z = (ct - cs.mean(1)) / cs.std(1)
    pr_t = np.median(PT[m] - PC[m], 0); pr_f = np.median(PF[m] - PC[m][:, None], (0, 1))
    pc, ptt, pff = pk(X[m]), pk(Y[m]), pk(X[m][:, None] + samp[m][:, :2])
    rat_t, rat_f = np.median(ptt / pc, 0), np.median(pff / pc[:, None], (0, 1))
    res[s] = dict(n=int(m.sum()), core_true_median=float(np.median(ct)), core_true_std=float(ct.std()),
                  core_flow_median=float(np.median(cs)), core_flow_std_across=float(cs[:, 0].std()),
                  core_rmse_flowmean=float(np.sqrt(np.mean((cs.mean(1) - ct) ** 2))),
                  core_rmse_zero=float(np.sqrt(np.mean(ct ** 2))),
                  core_corr_flowmean=float(np.corrcoef(cs.mean(1), ct)[0, 1]),
                  calib_z_std=float(z.std()),
                  ring_true=float(pr_t[3:10].mean()), ring_flow=float(pr_f[3:10].mean()),
                  power_ratio_true={L: float(rat_t[int(round(N / L))]) for L in (24, 8, 4, 2.5)},
                  power_ratio_flow={L: float(rat_f[int(round(N / L))]) for L in (24, 8, 4, 2.5)},
                  best_sigma_by_nll={str(g): int((np.argmin(nll[m], 1) == q).sum()) for q, g in enumerate(SGRID)},
                  mean_nll_true_minus_s0p1=float((nll[m][:, SGRID.index(s)] - nll[m][:, SGRID.index(0.1)]).mean() * N * N) if s in SGRID else None)
res["bahamas_test_core_true_median"] = bcore
res["mean_nll_per_px_at_true_sigma"] = {str(s): float(nll[S == s][:, SGRID.index(s)].mean()) for s in SIG}
json.dump(res, open(f"{OUT}/summary.json", "w"), indent=1)
np.savez(f"{OUT}/results.npz", core_t=core_t, core_s=core_s, nll=nll, sgrid=SGRID, sigma=S, z=Z, halo=H, PC=PC, PT=PT, PF=PF)

for s in SIG:
    q = res[s]
    print(f"sigma {s}: n {q['n']}  core true {q['core_true_median']:+.3f} (std {q['core_true_std']:.3f})  flow {q['core_flow_median']:+.3f} "
          f"(std {q['core_flow_std_across']:.3f});  RMSE flow-mean {q['core_rmse_flowmean']:.3f} vs zero {q['core_rmse_zero']:.3f}; "
          f"corr {q['core_corr_flowmean']:+.2f}; calib {q['calib_z_std']:.2f}; ring true {q['ring_true']:+.3f} flow {q['ring_flow']:+.3f}")
    print(f"    power ratio @24/8/4/2.5 px  true {[round(v, 3) for v in q['power_ratio_true'].values()]}  flow {[round(v, 3) for v in q['power_ratio_flow'].values()]}")
    print(f"    best sigma by flow-A likelihood {q['best_sigma_by_nll']}")
print("BAHAMAS test core medians", {k: round(v, 3) for k, v in bcore.items()})
print("mean NLL/px of true delta at true sigma (BAHAMAS val ~ -1.06):", res["mean_nll_per_px_at_true_sigma"])

# ---------- figures
col = dict(zip(SIG, ("#7fb3d5", "#2a9d8f", "#e9a03b", "#c0392b")))
rr = np.arange(NR) + 0.5; rr[0] = 0.35
fig, ax = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
for s in SIG:
    m = S == s; d_t, d_f = PT[m] - PC[m], (PF[m] - PC[m][:, None])[:, 0]
    ax[0].plot(rr, np.median(d_t, 0), color=col[s], lw=2, label=f"σ={s} (n={m.sum()})")
    ax[0].plot(rr, np.median(d_f, 0), color=col[s], lw=1.5, ls="--")
    ax[0].fill_between(rr, *np.percentile(d_t, [16, 84], 0), color=col[s], alpha=0.12, lw=0)
for s, c_ in zip((0.1, 0.3, 1.0), ("#e9a03b", "#8e44ad", "#555555")):
    ax[1].axvline(bcore[s], color=c_, ls=":", lw=1.5, label=f"BAHAMAS σ={s} median")
bins = np.linspace(-0.45, 0.25, 36)
for s in SIG:
    m = S == s
    ax[1].hist(core_t[m], bins, histtype="step", color=col[s], lw=2, density=True, label=f"darkskies true σ={s}")
    ax[1].hist(core_s[m][:, 0], bins, histtype="step", color=col[s], lw=1.2, ls="--", density=True)
cm = np.array([[res[s]["best_sigma_by_nll"][str(g)] for g in SGRID] for s in SIG], float)
cm /= cm.sum(1, keepdims=True)
ax[2].imshow(cm, cmap="Greys", vmin=0, vmax=1)
for i in range(len(SIG)):
    for j in range(len(SGRID)):
        ax[2].text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", color="w" if cm[i, j] > 0.5 else "k", fontsize=9)
ax[2].set_xticks(range(len(SGRID)), SGRID); ax[2].set_yticks(range(len(SIG)), SIG)
ax[2].set_xlabel("σ maximising flow-A likelihood of the true Δ"); ax[2].set_ylabel("true darkskies σ")
ax[2].set_title("σ from likelihood (row-normalised)")
ax[0].axhline(0, color="0.5", lw=0.7); ax[0].set_xscale("log"); ax[0].set_xlabel("r [px = 20 kpc]  (r=0 at 0.35)")
ax[0].set_ylabel("log10 ⟨Σ⟩ − log10 ⟨Σ⟩_CDM"); ax[0].legend(fontsize=8)
ax[0].set_title("darkskies radial Δ: solid true (band 16-84%), dashed flow A")
ax[1].set_xlabel("core Δ (r < 2 px) [dex]"); ax[1].legend(fontsize=7); ax[1].set_title("core Δ: solid true, dashed flow A sample")
fig.suptitle("flow A (trained on BAHAMAS σ = 0.1/0.3/1) applied to darkskies zooms")
fig.savefig(f"{OUT}/profiles.png", dpi=110)

fig, ax = plt.subplots(len(SIG), 4, figsize=(11, 11), constrained_layout=True)
h0 = H[S == 0.1][0].rsplit("_", 2)[0]
for i, s in enumerate(SIG):
    m = np.where((S == s) & np.char.startswith(H, h0) & np.char.endswith(H, "_10"))[0]
    m = m[0] if len(m) else np.where(S == s)[0][0]
    ax[i, 0].imshow(X[m], cmap="magma", origin="lower"); ax[i, 0].set_ylabel(f"σ={s}\n{H[m]}", fontsize=8)
    ax[i, 1].imshow(dt[m], cmap="RdBu_r", vmin=-.3, vmax=.3, origin="lower")
    ax[i, 2].imshow(samp[m, 0], cmap="RdBu_r", vmin=-.3, vmax=.3, origin="lower")
    im = ax[i, 3].imshow(samp[m, 1], cmap="RdBu_r", vmin=-.3, vmax=.3, origin="lower")
for j, t in enumerate(("CDM", "true Δ", "flow A Δ sample 1", "flow A Δ sample 2")):
    ax[0, j].set_title(t)
for a_ in ax.ravel():
    a_.set_xticks([]); a_.set_yticks([])
fig.colorbar(im, ax=ax, shrink=0.4, label="Δ [dex]")
fig.savefig(f"{OUT}/examples.png", dpi=100)
print("wrote", OUT)
