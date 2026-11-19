"""Evaluate flow A (conditional, delta | CDM) and flow B (shared latent) against the UNet and
the mean-profile baseline, on test pairs (sigma > 0, re-centring shift <= 2 px), central 96x96 px.

Point metrics (dex): core r<2 px RMSE / correlation / median, radial-profile RMSE (r<40 px),
whole-field R^2 of the delta map. Methods:
  profile      mean radial delta of training pairs at the same (sigma, z)
  unet         runs/<unet>/model.pt
  A_mode       flow A at eps = 0
  A_mean       mean of N flow-A samples (T = 1)
  A_sample     one flow-A sample
  B            flow B transform  f^-1(f(CDM; 0, z); sigma, z)
Distributional checks (where generative models should beat regression):
  core spread  std over clusters of the core delta: truth vs one sample per cluster; calibration
               z = (true - mean of samples) / std of samples should have std ~ 1
  power ratio  P(SIDM map) / P(CDM map) vs wavelength: truth vs each method
  sigma test   classify sigma by maximum likelihood: A on the true delta given CDM; B on the SIDM
               map alone (sigma = 0 included, CDM maps should come out as 0)
usage: python3 evaluate_flows.py [--A flowA] [--B flowB] [--unet all_sigma] [--split test] [--n 16]
"""
import os, json, argparse
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flows import Flow, cond_vec, mark_initialized
from sidm_unet import load as load_unet, cond_features

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--A", default="flowA"); ap.add_argument("--B", default="flowB")
ap.add_argument("--unet", default="all_sigma"); ap.add_argument("--split", default="test")
ap.add_argument("--n", type=int, default=16); ap.add_argument("--T", type=float, default=1.0)
ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
ap.add_argument("--no-wandb", action="store_true")
a = ap.parse_args()
dev = a.device
torch.manual_seed(0); np.random.seed(0)


def load_flow(name):
    ck = torch.load(f"{HERE}/runs/{name}/model.pt", map_location=dev, weights_only=False)
    m = Flow(**ck["arch"]).to(dev); m.load_state_dict(ck["state"]); mark_initialized(m); m.eval()
    m.norm, m.epoch = ck["norm"], ck["epoch"]
    return m


FA, FB = load_flow(a.A), load_flow(a.B)
C0, C1 = FA.norm["crop"]; DS = FA.norm["dscale"]; mu, sd = FA.norm["mu"], FA.norm["sd"]
D = np.load(f"{HERE}/data/pairs.npz", allow_pickle=True)
x_full, y_full, xi, sigma, shift, split, zz = (D[k] for k in ("x", "y", "xi", "sigma", "shift", "split", "redshift"))
x, y = x_full[:, C0:C1, C0:C1], y_full[:, C0:C1, C0:C1]
ok = (np.abs(shift).max(1) <= 2) & (sigma > 0)
ev = np.where(ok & (split[xi] == a.split))[0]
tr = np.where(ok & (split[xi] == "train"))[0]
d_true = (y[ev] - x[xi[ev]]).astype(np.float64)
N = x.shape[-1]; cc = (N - 1) / 2
yy, xx = np.indices((N, N)); r = np.hypot(yy - cc, xx - cc); rb = np.minimum(r.astype(int), 49)
CORE = r < 2
T = lambda v: torch.tensor(np.ascontiguousarray(v), dtype=torch.float32, device=dev)[:, None]
SIG = sorted(set(sigma[ev]))
print(f"{len(ev)} {a.split} pairs; flow A epoch {FA.epoch}, flow B epoch {FB.epoch}")


def prof(d):
    return np.stack([np.bincount(rb.ravel(), m.ravel(), 50) for m in d]) / np.bincount(rb.ravel(), minlength=50)


# ---- predictions (all in dex, 96x96)
P = {}
base = {(s, z): prof(y[k] - x[xi[k]]).mean(0) for s in SIG for z in np.unique(zz)
        for k in [tr[(sigma[tr] == s) & (zz[xi[tr]] == z)]]}
P["profile"] = np.stack([base[(sigma[k], zz[xi[k]])][rb] for k in ev])

U = load_unet(f"{HERE}/runs/{a.unet}/model.pt", dev)
out = []
with torch.no_grad():
    for s in range(0, len(ev), 64):
        k = ev[s:s + 64]
        xin = torch.tensor((x_full[xi[k]] - U.norm["mu"]) / U.norm["sd"], dtype=torch.float32, device=dev)[:, None]
        out.append(U(xin, cond_features(sigma[k], zz[xi[k]]).to(dev))[:, 0, C0:C1, C0:C1].cpu().numpy())
P["unet"] = np.concatenate(out)

UMAX = 4.0  # |u| range of the training data is ~3.8; clamp samples there (|delta| <= 2.7 dex)
nclamp = [0, 0]


def usinh(u):
    nclamp[0] += int((~np.isfinite(u) | (np.abs(u) > UMAX)).sum()); nclamp[1] += u.size
    return np.sinh(np.clip(np.nan_to_num(u, nan=0.0), -UMAX, UMAX)) * DS


samples = np.empty((len(ev), a.n, N, N), np.float32)
amode, btr = np.empty((len(ev), N, N), np.float32), np.empty((len(ev), N, N), np.float32)
with torch.no_grad():
    for s in range(0, len(ev), 32):
        k = ev[s:s + 32]; B = len(k)
        cv = cond_vec(sigma[k], zz[xi[k]]).to(dev); cimg = T((x[xi[k]] - mu) / sd)
        eps0 = [0 * e for e in FA.sample_eps(B, dev, H=N)]
        amode[s:s + B] = usinh(FA.decode(eps0, cv, cimg, H=N)[:, 0].cpu().numpy())
        for j in range(a.n):
            samples[s:s + B, j] = usinh(FA.decode(FA.sample_eps(B, dev, a.T, H=N), cv, cimg, H=N)[:, 0].cpu().numpy())
        cv0 = cond_vec(np.zeros(B), zz[xi[k]]).to(dev)
        eps, _ = FB.encode(cimg, cv0)
        btr[s:s + B] = (FB.decode(eps, cv, None, H=N)[:, 0].cpu().numpy() * sd + mu) - x[xi[k]]
print(f"flow A: {100 * nclamp[0] / nclamp[1]:.4f}% of decoded pixels clamped to |u| <= {UMAX}")
valid = np.isfinite(samples).all((2, 3))
samples[~valid] = np.nan
has = valid.any(1)
first = np.where(has, valid.argmax(1), 0)
P["A_mode"], P["B"] = amode, btr
P["A_mean"] = np.where(has[:, None, None], np.nanmean(np.where(valid[:, :, None, None], samples, np.nan), 1), amode)
P["A_sample"] = np.where(has[:, None, None], samples[np.arange(len(ev)), first], amode)
METHODS = list(P)

# ---- point metrics per sigma
rows = []
Pt = prof(d_true)
for s in SIG:
    m = sigma[ev] == s
    o = dict(sigma=float(s), n=int(m.sum()))
    ct = d_true[m][:, CORE].mean(1)
    o["core_true_median"], o["core_true_std"] = float(np.median(ct)), float(ct.std())
    for nm in METHODS:
        pm = P[nm][m].astype(np.float64)
        cp = pm[:, CORE].mean(1)
        o[f"core_rmse_{nm}"] = float(np.sqrt(((cp - ct) ** 2).mean()))
        o[f"core_corr_{nm}"] = float(np.corrcoef(cp, ct)[0, 1]) if cp.std() > 0 else 0.0
        o[f"core_median_{nm}"] = float(np.median(cp))
        o[f"core_std_{nm}"] = float(cp.std())
        o[f"prof_rmse_{nm}"] = float(np.sqrt(((prof(pm) - Pt[m])[:, :40] ** 2).mean()))
        o[f"field_r2_{nm}"] = float(1 - ((pm - d_true[m]) ** 2).sum() / (d_true[m] ** 2).sum())
    cs = samples[m][:, :, CORE].mean(2)  # (n_clusters, n_samples), NaN where invalid
    good = np.isfinite(cs).sum(1) >= 2
    zsc = (ct[good] - np.nanmean(cs[good], 1)) / np.nanstd(cs[good], 1)
    o["core_calib_std_z"] = float(np.std(zsc))
    o["core_sample_spread_median"] = float(np.median(np.nanstd(cs[good], 1)))
    rows.append(o)

w = 12
print("\ncore r<2 px: RMSE / corr / median   (true median, true std across clusters)")
for o in rows:
    print(f"sigma {o['sigma']}: true {o['core_true_median']:+.3f} (std {o['core_true_std']:.3f})")
    for nm in METHODS:
        print(f"   {nm:{w}s} RMSE {o[f'core_rmse_{nm}']:.4f}  corr {o[f'core_corr_{nm}']:+.2f}  "
              f"median {o[f'core_median_{nm}']:+.3f}  std across clusters {o[f'core_std_{nm}']:.3f}  "
              f"profile RMSE {o[f'prof_rmse_{nm}']:.4f}  field R^2 {o[f'field_r2_{nm}']:+.3f}")
    print(f"   flow A calibration: std of (true - sample mean)/sample std = {o['core_calib_std_z']:.2f} (1 = calibrated);"
          f" median per-cluster sample spread {o['core_sample_spread_median']:.3f}")

# ---- power spectrum ratio P(SIDM) / P(CDM)
win = np.outer(np.hanning(N), np.hanning(N))
fy, fx = np.meshgrid(np.fft.fftfreq(N) * N, np.fft.fftfreq(N) * N, indexing="ij")
kb = np.rint(np.hypot(fy, fx)).astype(int); nk = N // 2


def pk(maps):
    m = maps - maps.mean((1, 2), keepdims=True)
    F2 = np.abs(np.fft.fft2(m * win)) ** 2
    return np.stack([np.bincount(kb.ravel(), f.ravel(), kb.max() + 1)[:nk] for f in F2]).mean(0)


lc = x[xi[ev]].astype(np.float64)
spec = {}
for s in SIG:
    m = sigma[ev] == s
    p0 = pk(lc[m])
    spec[s] = {"true": pk(lc[m] + d_true[m]) / p0}
    for nm in ("unet", "A_mode", "A_sample", "B"):
        spec[s][nm] = pk(lc[m] + P[nm][m]) / p0
kk = np.arange(nk); lam = N / np.maximum(kk, 1)
print("\npower ratio P_SIDM / P_CDM at wavelength 24 / 8 / 4 / 2.5 px")
pick = [int(round(N / l)) for l in (24, 8, 4, 2.5)]
for s in SIG:
    print(f"sigma {s}: " + "   ".join(f"{nm} " + "/".join(f"{spec[s][nm][p]:.3f}" for p in pick) for nm in spec[s]))

# ---- sigma classification by likelihood
cand_A = SIG
nllA = np.empty((len(ev), len(cand_A)))
cand_B = [0.0] + SIG
nllB = np.empty((len(ev), len(cand_B))); nllB0 = np.empty((len(ev), len(cand_B)))
with torch.no_grad():
    for s in range(0, len(ev), 64):
        k = ev[s:s + 64]; B = len(k)
        cimg = T((x[xi[k]] - mu) / sd); xs = T((y[k] - mu) / sd); dA = T(np.arcsinh((y[k] - x[xi[k]]) / DS))
        for j, c in enumerate(cand_A):
            nllA[s:s + B, j] = FA.nll(dA, cond_vec(np.full(B, c), zz[xi[k]]).to(dev), cimg)[0].cpu().numpy()
        for j, c in enumerate(cand_B):
            cv = cond_vec(np.full(B, c), zz[xi[k]]).to(dev)
            nllB[s:s + B, j] = FB.nll(xs, cv)[0].cpu().numpy()
            nllB0[s:s + B, j] = FB.nll(cimg, cv)[0].cpu().numpy()
truthA = np.array([cand_A.index(v) for v in sigma[ev]])
truthB = np.array([cand_B.index(v) for v in sigma[ev]])
confA = np.zeros((len(cand_A),) * 2, int); confB = np.zeros((len(cand_B),) * 2, int)
for t, p in zip(truthA, nllA.argmin(1)): confA[t, p] += 1
for t, p in zip(truthB, nllB.argmin(1)): confB[t, p] += 1
cdm_as0 = float((nllB0.argmin(1) == 0).mean())
accA, accB = float(np.trace(confA) / confA.sum()), float(np.trace(confB) / confB.sum())
print(f"\nsigma from likelihood, flow A (true delta given CDM): accuracy {accA:.2f} (chance {1/len(cand_A):.2f})")
print("   rows true, cols chosen", cand_A, "\n", confA)
print(f"sigma from likelihood, flow B (SIDM map alone, sigma 0 allowed): accuracy {accB:.2f}; "
      f"CDM maps classified as sigma=0: {cdm_as0:.2f}")
print("   rows true", SIG, "cols chosen", cand_B, "\n", confB)
# mean NLL gap: how much better the true sigma is than sigma=0 (B) / than the others (A), nats per pixel x 1e4
gapB = (nllB[:, 0] - nllB[np.arange(len(ev)), truthB]).mean() * 1e4
print(f"flow B: mean NLL(sigma=0) - NLL(true sigma) = {gapB:.2f} x 1e-4 nats/px")

res = dict(rows=rows, accA=accA, accB=accB, cdm_as0=cdm_as0, confA=confA.tolist(), confB=confB.tolist(),
           cand_A=cand_A, cand_B=cand_B, n_eval=len(ev), flowA_epoch=FA.epoch, flowB_epoch=FB.epoch)
odir = f"{HERE}/runs/flow_eval"; os.makedirs(odir, exist_ok=True)
json.dump(res, open(f"{odir}/eval_{a.split}.json", "w"), indent=1)

# ---- figures
INK, MUTED = "#1f1f1e", "#6b6a63"
COL = dict(true=INK, profile="#eb6834", unet="#2a78d6", A_mode="#3a9a5b", A_mean="#7fc49a", A_sample="#9b59b6", B="#c0392b")
fig, ax = plt.subplots(2, len(SIG), figsize=(5 * len(SIG), 8.5), facecolor="white", squeeze=False)
for j, s in enumerate(SIG):
    m = sigma[ev] == s
    a0 = ax[0, j]
    bins = np.linspace(-0.45, 0.25, 36)
    a0.hist(d_true[m][:, CORE].mean(1), bins, color=INK, histtype="step", lw=2, label="true")
    for nm in ("unet", "A_sample", "B"):
        a0.hist(P[nm][m][:, CORE].mean(1), bins, color=COL[nm], histtype="step", lw=1.6, label=nm)
    a0.set_title(f"sigma = {s}: core delta (r < 2 px) across clusters", fontsize=10)
    a0.set_xlabel("dex")
    a1 = ax[1, j]
    sel = (kk >= 2)
    for nm, v in spec[s].items():
        a1.plot(lam[sel], v[sel], color=COL[nm], lw=2 if nm == "true" else 1.5, label=nm)
    a1.axhline(1, color=MUTED, lw=0.6); a1.set_xscale("log"); a1.invert_xaxis()
    a1.set_xlabel("wavelength [px]"); a1.set_title("power ratio P(SIDM) / P(CDM)", fontsize=10)
ax[0, 0].legend(frameon=False, fontsize=8); ax[1, 0].legend(frameon=False, fontsize=8)
fig.suptitle(f"flows vs UNet, {a.split} split ({len(ev)} pairs). sigma from likelihood: A {accA:.2f}, B {accB:.2f}")
fig.tight_layout(); fig.savefig(f"{odir}/summary_{a.split}.png", dpi=110)

# examples: one cluster per sigma
fig, ax = plt.subplots(len(SIG), 7, figsize=(21, 3.2 * len(SIG)), facecolor="white", squeeze=False)
mid = ev[np.argsort(-x[xi[ev]][:, 44:52, 44:52].mean((1, 2)))]
for i, s in enumerate(SIG):
    k = [q for q in mid if sigma[q] == s][len(mid) // 12]
    e = int(np.where(ev == k)[0][0])
    panels = [(x[xi[k]], "log10 CDM", "magma", None), (d_true[e], "true delta", "RdBu_r", 0.4),
              (P["unet"][e], "UNet", "RdBu_r", 0.4), (P["A_mode"][e], "flow A, mode", "RdBu_r", 0.4),
              (samples[e, 0], "flow A, sample 1", "RdBu_r", 0.4), (samples[e, 1], "flow A, sample 2", "RdBu_r", 0.4),
              (P["B"][e], "flow B transform", "RdBu_r", 0.4)]
    for jj, (m_, t, cm, v) in enumerate(panels):
        ax[i, jj].imshow(m_, origin="lower", cmap=cm, vmin=-v if v else None, vmax=v)
        ax[i, jj].set_xticks([]); ax[i, jj].set_yticks([])
        if i == 0: ax[i, jj].set_title(t, fontsize=10)
    ax[i, 0].set_ylabel(f"sigma = {s}\n{D['match_id'][xi[k]]} p{D['projection'][xi[k]]}", fontsize=9)
fig.tight_layout(); fig.savefig(f"{odir}/examples_{a.split}.png", dpi=100)
print(f"{odir}/summary_{a.split}.png\n{odir}/examples_{a.split}.png")

if not a.no_wandb:
    import wandb
    wr = wandb.init(project="cdm_to_sidm_unet", name=f"flow_eval_{a.split}", tags=["flow", "eval"], dir=odir,
                    config=dict(A=a.A, B=a.B, unet=a.unet, n=a.n, T=a.T))
    wr.log({"summary": wandb.Image(f"{odir}/summary_{a.split}.png"),
            "examples": wandb.Image(f"{odir}/examples_{a.split}.png")})
    for o in rows:
        for kx, v in o.items():
            wr.summary[f"sigma{o['sigma']}/{kx}"] = v
    wr.summary.update(dict(sigma_acc_A=accA, sigma_acc_B=accB, cdm_as_sigma0_B=cdm_as0))
    wr.finish()
