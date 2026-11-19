"""Radial surface-density profiles, true vs flow A, for CDM and the three SIDM models.
Top: one test cluster (16 flow samples). Bottom: whole test split (1 flow sample per pair).
Profile = log10 of the azimuthal mean of linear Sigma (map units as in pairs.npz).
usage: python3 plot_flow_profiles.py [cdm_row_index]   (default 8)"""
import os, sys
import numpy as np, torch, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from flows import Flow, cond_vec, mark_initialized

HERE = os.path.dirname(os.path.abspath(__file__))
dev = "mps" if torch.backends.mps.is_available() else "cpu"
torch.manual_seed(1)
ck = torch.load(f"{HERE}/runs/flowA/model.pt", map_location=dev, weights_only=False)
FA = Flow(**ck["arch"]).to(dev); FA.load_state_dict(ck["state"]); mark_initialized(FA); FA.eval()
C0, C1 = ck["norm"]["crop"]; DS, mu, sd = ck["norm"]["dscale"], ck["norm"]["mu"], ck["norm"]["sd"]
D = np.load(f"{HERE}/data/pairs.npz", allow_pickle=True)
x, y, xi, sigma, shift, split, zz = (D[k] for k in ("x", "y", "xi", "sigma", "shift", "split", "redshift"))
x, y = x[:, C0:C1, C0:C1], y[:, C0:C1, C0:C1]
ok = (np.abs(shift).max(1) <= 2) & (sigma > 0) & (split[xi] == "test")
SIG = (0.1, 0.3, 1.0)
c = int(sys.argv[1]) if len(sys.argv) > 1 else 8
N = x.shape[-1]; cc = (N - 1) / 2; yy, xx = np.indices((N, N)); rb = np.hypot(yy - cc, xx - cc).astype(int).ravel()
NR = 48; cnt = np.bincount(rb, minlength=NR + 30)[:NR]; r = np.arange(NR) + 0.5; r[0] = 0.35
prof = lambda m: np.log10(np.bincount(rb, (10.0 ** m).ravel(), NR + 30)[:NR] / cnt)


def decode(cdm, s, z, eps=None):
    B = len(cdm)
    cv = cond_vec(np.full(B, s) if np.isscalar(s) else s, z).to(dev)
    cimg = torch.tensor((cdm - mu) / sd, dtype=torch.float32, device=dev)[:, None]
    eps = FA.sample_eps(B, dev, H=N) if eps is None else eps
    with torch.no_grad():
        u = FA.decode(eps, cv, cimg, H=N)[:, 0].cpu().numpy()
    return cdm + np.sinh(np.clip(u, -4, 4)) * DS


# one cluster
NS = 16
one = {s: np.stack([prof(m) for m in decode(np.repeat(x[c][None], NS, 0), s, np.full(NS, zz[c]))]) for s in SIG}
tru = {s: prof(y[np.where((xi == c) & ok & (sigma == s))[0][0]]) for s in SIG}
pc = prof(x[c])
# population
ev = np.where(ok)[0]
PT, PF = np.empty((len(ev), NR)), np.empty((len(ev), NR))
for s0 in range(0, len(ev), 64):
    k = ev[s0:s0 + 64]
    f = decode(x[xi[k]], sigma[k], zz[xi[k]])
    PT[s0:s0 + 64] = [prof(m) for m in y[k]]; PF[s0:s0 + 64] = [prof(m) for m in f]
PC = np.stack([prof(m) for m in x[xi[ev]]])

col = dict(zip(SIG, ("#2a9d8f", "#e9a03b", "#c0392b")))
fig, ax = plt.subplots(2, 2, figsize=(12, 9.5), constrained_layout=True)
# --- one cluster
a = ax[0, 0]
a.plot(r, pc, color="k", lw=2, label="CDM")
for s in SIG:
    a.plot(r, tru[s], color=col[s], lw=2, label=f"true σ={s}")
    a.plot(r, np.median(one[s], 0), color=col[s], lw=1.5, ls="--", label=f"flow A σ={s}")
a.set_xscale("log"); a.set_ylabel("log10 ⟨Σ⟩(r)"); a.set_title(f"cluster row {c}: profiles"); a.legend(fontsize=7, ncol=2)
a = ax[0, 1]
for s in SIG:
    lo, hi = np.percentile(one[s] - pc, [16, 84], 0)
    a.fill_between(r, lo, hi, color=col[s], alpha=0.25, lw=0)
    a.plot(r, np.median(one[s] - pc, 0), color=col[s], ls="--", lw=1.5)
    a.plot(r, tru[s] - pc, color=col[s], lw=2, label=f"σ={s}")
a.set_title(f"cluster row {c}: ratio to CDM\nsolid true; dashed flow A median, band 16-84% of {NS} samples", fontsize=9)
# --- population
a = ax[1, 0]
a.plot(r, np.median(PC, 0), color="k", lw=2, label="CDM")
for s in SIG:
    m = sigma[ev] == s
    a.plot(r, np.median(PT[m], 0), color=col[s], lw=2, label=f"true σ={s}")
    a.plot(r, np.median(PF[m], 0), color=col[s], lw=1.5, ls="--", label=f"flow A σ={s}")
a.set_xscale("log"); a.set_ylabel("log10 ⟨Σ⟩(r)"); a.set_title(f"test split, median of {len(ev)} pairs"); a.legend(fontsize=7, ncol=2)
a = ax[1, 1]
for s in SIG:
    m = sigma[ev] == s
    dt, df = PT[m] - PC[m], PF[m] - PC[m]
    a.fill_between(r, *np.percentile(dt, [16, 84], 0), color=col[s], alpha=0.15, lw=0)
    a.plot(r, np.percentile(df, 16, 0), color=col[s], lw=0.8, ls=":"); a.plot(r, np.percentile(df, 84, 0), color=col[s], lw=0.8, ls=":")
    a.plot(r, np.median(dt, 0), color=col[s], lw=2, label=f"σ={s}")
    a.plot(r, np.median(df, 0), color=col[s], lw=1.5, ls="--")
a.set_title("test split, ratio to CDM: median (solid true, dashed flow A);\n16-84% across clusters (band true, dotted flow A)", fontsize=9)
for a in ax[:, 1]:
    a.axhline(0, color="0.5", lw=0.7); a.set_xscale("log"); a.set_ylabel("log10 ⟨Σ⟩ − log10 ⟨Σ⟩_CDM [dex]"); a.legend(fontsize=8)
for a in ax.ravel():
    a.set_xlabel("r [px]  (r=0 bin plotted at 0.35)")
out = f"{HERE}/figures/flowA_profiles.png"; fig.savefig(out, dpi=110); print(out)
for s in SIG:
    m = sigma[ev] == s
    print(f"sigma {s}: inner bin median true {np.median(PT[m]-PC[m],0)[0]:+.3f} flow {np.median(PF[m]-PC[m],0)[0]:+.3f};"
          f" scatter true {np.std(PT[m]-PC[m],0)[0]:.3f} flow {np.std(PF[m]-PC[m],0)[0]:.3f}")
