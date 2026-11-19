"""Evaluate a trained UNet on the test split (or val) against two baselines.

delta = log10 SIDM - log10 CDM (dex), per test pair with sigma > 0:
  zero      predict no change
  profile   mean radial delta(r) of the training pairs at the same sigma and
            redshift (for a held-out sigma: interpolated linearly in sqrt(sigma)
            between the neighbouring trained sigmas)
  unet      the model
Metrics per sigma: pixel RMSE (masked as in training), RMSE of each cluster's
azimuthal profile, and the central delta (r < 4 px): RMSE and the correlation
between predicted and true across clusters (the profile baseline has ~none
within a redshift, so this is where per-cluster information shows up).
Figure: runs/<name>/eval_<split>.png
usage: python3 evaluate.py --name all_sigma [--split test]
"""
import os, json, argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from sidm_unet import load, cond_features

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--split", default="test")
ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
ap.add_argument("--no-wandb", action="store_true", help="don't log into the training run's wandb run")
a = ap.parse_args()
run = f"{HERE}/runs/{a.name}"
cfg = json.load(open(f"{run}/config.json"))
holdout = cfg.get("holdout")
D = np.load(f"{HERE}/data/pairs.npz", allow_pickle=True)
x, y, xi, sigma, shift, split, zz = (D[k] for k in ("x", "y", "xi", "sigma", "shift", "split", "redshift"))
delta = y - x[xi]

yy, xx = np.indices((100, 100))
r = np.hypot(yy - 49.5, xx - 49.5)
rb = np.minimum(r.astype(int), 49)  # 1-px radial bins, 0..49
CEN = r < 4
CORE = r < 2  # the SIDM core deficit lives here; r<4 also mixes in the positive ring


def wmask(s):
    w = np.ones((len(s), 100, 100))
    for k, (dy, dx) in enumerate(s):
        if dy > 0: w[k, :dy] = 0
        if dy < 0: w[k, dy:] = 0
        if dx > 0: w[k, :, :dx] = 0
        if dx < 0: w[k, :, dx:] = 0
    return w


def profiles(d, w):
    num = np.stack([np.bincount(rb.ravel(), (dk * wk).ravel(), 50) for dk, wk in zip(d, w)])
    den = np.stack([np.bincount(rb.ravel(), wk.ravel(), 50) for wk in w])
    return num / np.maximum(den, 1)


# ---- profile baseline from training pairs
tr = (split[xi] == "train") & (sigma > 0)
trained = sorted(s for s in set(sigma[tr]) if s != holdout)
base = {}
for s in trained:
    for z in np.unique(zz):
        k = np.where(tr & (sigma == s) & (zz[xi] == z))[0]
        base[(s, z)] = profiles(delta[k], wmask(shift[k])).mean(0)


def baseline_profile(s, z):
    if (s, z) in base:
        return base[(s, z)]
    lo = max([t for t in trained if t < s], default=0.0)
    hi = min(t for t in trained if t > s)
    f = (np.sqrt(s) - np.sqrt(lo)) / (np.sqrt(hi) - np.sqrt(lo))
    plo = base[(lo, z)] if lo > 0 else np.zeros(50)
    return (1 - f) * plo + f * base[(hi, z)]


# ---- predictions on the evaluation split
model = load(f"{run}/model.pt", a.device)
mu, sd = model.norm["mu"], model.norm["sd"]
LIN_IN = model.norm.get("input_space", "log") == "linear"
RATIO = model.norm.get("target", "logratio") == "ratio"
ev = np.where((split[xi] == a.split) & (sigma > 0))[0]
pred = np.empty((len(ev), 100, 100), np.float32)
with torch.no_grad():
    for s in range(0, len(ev), 64):
        k = ev[s:s + 64]
        lx = x[xi[k]].astype(np.float64)
        xin = torch.tensor(((10 ** lx if LIN_IN else lx) - mu) / sd, dtype=torch.float32, device=a.device)[:, None]
        p = model(xin, cond_features(sigma[k], zz[xi[k]]).to(a.device))[:, 0].cpu().numpy()
        # metrics are all in dex: a ratio prediction R-1 becomes log10(R), floored at R = 1e-3
        pred[s:s + 64] = np.log10(np.maximum(1 + p, 1e-3)) if RATIO else p

w = wmask(shift[ev]); d = delta[ev]
bprof = np.stack([baseline_profile(sigma[k], zz[xi[k]]) for k in ev])
bmap = bprof[:, rb]


# ---- Fourier baseline: keep the CDM phases, rescale each |k| by a transfer function
# T(|k|; sigma, z) = sum Re(F_target F_cdm*) / sum |F_cdm|^2 over training pairs (the optimal
# isotropic linear filter = a fixed convolution kernel). Log maps, mirror-padded to 200x200
# so the FFT sees a periodic field. Held-out sigma: interpolated in sqrt(sigma) as above.
NPAD = 200
fy, fx = np.meshgrid(np.fft.fftfreq(NPAD) * NPAD, np.fft.fftfreq(NPAD) * NPAD, indexing="ij")
KB = np.rint(np.hypot(fy, fx)).astype(int)  # integer |k| bin, 0..141
NKB = KB.max() + 1
mirror = lambda m: np.pad(m, ((0, 0), (0, NPAD - 100), (0, NPAD - 100)), mode="symmetric")


def fourier_T(k, lin):
    """lin=False: filter the log10 maps; lin=True: filter the linear mass maps (mass-conserving smoothing)."""
    g = (lambda v: 10.0 ** v) if lin else (lambda v: v)
    num, den = np.zeros(NKB), np.zeros(NKB)
    for s in range(0, len(k), 256):
        kk = k[s:s + 256]
        Fx, Fy = np.fft.fft2(mirror(g(x[xi[kk]]))), np.fft.fft2(mirror(g(y[kk])))
        num += np.bincount(KB.ravel(), (Fy * Fx.conj()).real.sum(0).ravel(), NKB)
        den += np.bincount(KB.ravel(), (np.abs(Fx) ** 2).sum(0).ravel(), NKB)
    return num / den


def fourier_predict(lin):
    TF = {(s, z): fourier_T(np.where(tr & (sigma == s) & (zz[xi] == z))[0], lin)
          for s in trained for z in np.unique(zz)}

    def transfer(s, z):
        if (s, z) in TF:
            return TF[(s, z)]
        lo = max([t for t in trained if t < s], default=0.0)
        hi = min(t for t in trained if t > s)
        f = (np.sqrt(s) - np.sqrt(lo)) / (np.sqrt(hi) - np.sqrt(lo))
        return (1 - f) * (TF[(lo, z)] if lo > 0 else np.ones(NKB)) + f * TF[(hi, z)]

    out = np.empty_like(d)
    for s in range(0, len(ev), 256):
        kk = ev[s:s + 256]
        T = np.stack([transfer(sigma[k], zz[xi[k]])[KB] for k in kk])
        lx = x[xi[kk]]
        if lin:
            X = 10.0 ** lx
            Y = np.fft.ifft2(np.fft.fft2(mirror(X)) * T).real[:, :100, :100]
            out[s:s + 256] = np.log10(np.maximum(Y, 0.1 * X)) - lx  # floor: never below 10% of CDM
        else:
            out[s:s + 256] = np.fft.ifft2(np.fft.fft2(mirror(lx)) * T).real[:, :100, :100] - lx
    return out, TF


fmap, TF_log = fourier_predict(False)
lmap, TF_lin = fourier_predict(True)
P_four, P_lin = profiles(fmap, w), profiles(lmap, w)
np.savez(f"{run}/fourier_transfer.npz", k=np.arange(NKB), npad=NPAD,
         **{f"log_s{s}_z{z:.3f}": v for (s, z), v in TF_log.items()},
         **{f"lin_s{s}_z{z:.3f}": v for (s, z), v in TF_lin.items()})
P_true, P_unet = profiles(d, w), profiles(pred, w)
RF = r.astype(int)  # 1-px bins over the whole field, 0..70 (corners included)


def profiles_full(dd, ww):
    num = np.stack([np.bincount(RF.ravel(), (dk * wk).ravel(), 71) for dk, wk in zip(dd, ww)])
    return num / np.maximum(np.stack([np.bincount(RF.ravel(), wk.ravel(), 71) for wk in ww]), 1)


PF_true = profiles_full(d, w)
NF = np.stack([np.bincount(RF.ravel(), wk.ravel(), 71) for wk in w])  # pixels per ring: weights the field profile RMSE by area
rows = []
for s in sorted(set(sigma[ev])):
    m = sigma[ev] == s
    out = dict(sigma=s, n=int(m.sum()), held_out=bool(s == holdout))
    for nm, pm, pp in (("zero", 0 * d, 0 * P_true), ("profile", bmap, bprof), ("fourier", fmap, P_four), ("fourier_lin", lmap, P_lin),
                       ("unet", pred, P_unet)):
        out[f"pix_rmse_{nm}"] = float(np.sqrt((((pm - d) ** 2 * w)[m]).sum() / w[m].sum()))
        out[f"prof_rmse_{nm}"] = float(np.sqrt(((pp - P_true)[m][:, :40] ** 2).mean()))
        cp, ct = (pm[m][:, CEN]).mean(1), (d[m][:, CEN]).mean(1)
        out[f"cen_rmse_{nm}"] = float(np.sqrt(((cp - ct) ** 2).mean()))
        out[f"cen_corr_{nm}"] = float(np.corrcoef(cp, ct)[0, 1]) if cp.std() > 0 else 0.0
        cp2, ct2 = (pm[m][:, CORE]).mean(1), (d[m][:, CORE]).mean(1)
        out[f"core_rmse_{nm}"] = float(np.sqrt(((cp2 - ct2) ** 2).mean()))
        out[f"core_corr_{nm}"] = float(np.corrcoef(cp2, ct2)[0, 1]) if cp2.std() > 0 else 0.0
        out[f"core_median_{nm}"] = float(np.median(cp2))
        # whole field: per-cluster weighted Pearson correlation of the delta maps (all valid pixels),
        # and the RMSE of the azimuthal profile over every radius out to the corners (r <= 70 px)
        wm, dm_, pm_ = w[m], d[m], pm[m]
        ws = wm.sum((1, 2))
        dc = dm_ - (dm_ * wm).sum((1, 2))[:, None, None] / ws[:, None, None]
        pc = pm_ - (pm_ * wm).sum((1, 2))[:, None, None] / ws[:, None, None]
        num = (dc * pc * wm).sum((1, 2))
        den = np.sqrt((dc ** 2 * wm).sum((1, 2)) * (pc ** 2 * wm).sum((1, 2)))
        out[f"field_corr_{nm}"] = float(np.median(np.where(den > 0, num / np.where(den > 0, den, 1), 0)))
        pf = (profiles_full(pm_, wm) if nm != "zero" else 0) - PF_true[m]
        out[f"field_prof_rmse_{nm}"] = float(np.sqrt((pf ** 2 * NF[m]).sum() / NF[m].sum()))
        out[f"field_r2_{nm}"] = float(1 - (((pm - d) ** 2 * w)[m]).sum() / ((d ** 2 * w)[m]).sum())
    out["cen_true_median"] = float(np.median(d[m][:, CEN].mean(1)))
    out["core_true_median"] = float(np.median(d[m][:, CORE].mean(1)))
    out["cen_unet_median"] = float(np.median(pred[m][:, CEN].mean(1)))
    rows.append(out)
print(f"{a.name} on {a.split} (held-out sigma: {holdout})")
NM = ("zero", "profile", "fourier", "fourier_lin", "unet")
f3 = lambda o, k, p=3, nm=NM: "/".join(f"{o[f'{k}_{n}']:.{p}f}" for n in nm)
print("methods: zero / mean profile / Fourier filter on log map / Fourier filter on linear mass map / UNet")
print(f"{'sigma':>6} {'n':>5}  {'pixel RMSE':>34}  {'profile RMSE r<40':>34}  {'central RMSE r<4':>34}  {'central corr p/f/fl/u':>22}")
for o in rows:
    print(f"{o['sigma']:>6} {o['n']:>5}  {f3(o, 'pix_rmse', 4):>34}  {f3(o, 'prof_rmse', 4):>34}  "
          f"{f3(o, 'cen_rmse', 4):>34}  {f3(o, 'cen_corr', 2, NM[1:]):>22}" + ("   <- held out" if o["held_out"] else ""))
print(f"\ncore r<2 px\n{'sigma':>6} {'n':>5}  {'RMSE':>34}  {'corr p/f/fl/u':>22}  {'median true / unet':>20}")
for o in rows:
    print(f"{o['sigma']:>6} {o['n']:>5}  {f3(o, 'core_rmse', 4):>34}  {f3(o, 'core_corr', 2, NM[1:]):>22}  "
          f"{o['core_true_median']:+.3f} / {o['core_median_unet']:+.3f}")
print(f"\nwhole field\n{'sigma':>6} {'n':>5}  {'R^2 of delta map':>29}  {'median map corr':>29}  {'profile RMSE r<=70':>29}")
for o in rows:
    print(f"{o['sigma']:>6} {o['n']:>5}  {f3(o, 'field_r2'):>29}  {f3(o, 'field_corr'):>29}  {f3(o, 'field_prof_rmse'):>29}")
json.dump(rows, open(f"{run}/eval_{a.split}.json", "w"), indent=1)

# ---- figure: median profiles and central delta scatter per sigma
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#1f1f1e", "#6b6a63"
S = sorted(set(sigma[ev]))
fig, ax = plt.subplots(2, len(S), figsize=(4.2 * len(S), 7.6), facecolor="white", squeeze=False)
rr = np.arange(50) + 0.5
for j, s in enumerate(S):
    m = sigma[ev] == s
    a0 = ax[0, j]
    for P, c, lab in ((P_true, INK, "true"), (P_unet, BLUE, "UNet"), (bprof, ORANGE, "mean-profile baseline"),
                      (P_four, "#3a9a5b", "Fourier filter, log map"),
                      (P_lin, "#9b59b6", "Fourier filter, linear mass")):
        q = np.percentile(P[m], [16, 50, 84], axis=0)
        a0.plot(rr, q[1], color=c, lw=2, label=lab)
        if lab in ("true", "UNet"):
            a0.fill_between(rr, q[0], q[2], color=c, alpha=0.12, lw=0)
    a0.axhline(0, color=MUTED, lw=0.6)
    a0.set_xlim(0, 45); a0.set_xlabel("radius [px]")
    a0.set_title(f"sigma = {s}" + ("  (held out)" if s == holdout else ""), color=INK, fontsize=10)
    if j == 0:
        a0.set_ylabel("delta = log10(SIDM/CDM) [dex]\nmedian, 16-84% band"); a0.legend(frameon=False, fontsize=8)
    a1 = ax[1, j]
    ct, cu = d[m][:, CEN].mean(1), pred[m][:, CEN].mean(1)
    a1.scatter(ct, cu, s=6, color=BLUE, alpha=0.5, lw=0)
    lim = np.percentile(np.r_[ct, cu], [1, 99]); a1.plot(lim, lim, color=MUTED, lw=0.8)
    a1.set_xlim(lim); a1.set_ylim(lim)
    o = rows[j]
    a1.set_title(f"central delta (r<4 px): corr {o['cen_corr_unet']:.2f}, "
                 f"RMSE {o['cen_rmse_unet']:.3f} (baseline {o['cen_rmse_profile']:.3f})", fontsize=8.5, color=INK)
    a1.set_xlabel("true"); a1.set_ylabel("UNet" if j == 0 else "")
fig.suptitle(f"{a.name}: {a.split} split", color=INK)
fig.tight_layout()
fig.savefig(f"{run}/eval_{a.split}.png", dpi=120)
print(f"{run}/eval_{a.split}.png")

# ---- log into the training run on wandb (resumed by id)
if not a.no_wandb and os.path.isfile(f"{run}/wandb_id.txt"):
    import wandb
    wr = wandb.init(project=cfg.get("wandb_project", "cdm_to_sidm_unet"),
                    id=open(f"{run}/wandb_id.txt").read().strip(), resume="must", dir=run)
    cols = list(rows[0])
    wr.log({f"eval_{a.split}/table": wandb.Table(columns=cols, data=[[o[c] for c in cols] for o in rows]),
            f"eval_{a.split}/figure": wandb.Image(f"{run}/eval_{a.split}.png")})
    for o in rows:
        for k, v in o.items():
            if k not in ("sigma", "n", "held_out"):
                wr.summary[f"{a.split}/sigma{o['sigma']}/{k}"] = v
    wr.finish()
    print("logged to wandb run", wr.id)
