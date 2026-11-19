"""One test cluster in all four models: true maps vs flow A samples (same latent eps at every sigma).
usage: python3 plot_flow_four.py [cdm_row_index] ; default: first test CDM map with all three sigma pairs"""
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
if len(sys.argv) > 1:
    c = int(sys.argv[1])
else:
    c = next(i for i in np.unique(xi[ok]) if all(((xi == i) & ok & (sigma == s)).any() for s in SIG))
pr = {s: np.where((xi == c) & ok & (sigma == s))[0][0] for s in SIG}
N, NS = x.shape[-1], 16
# same eps for all sigma -> differences between columns are driven by sigma only
eps = FA.sample_eps(NS, dev, H=N)
cimg = torch.tensor((x[c] - mu) / sd, dtype=torch.float32, device=dev)[None, None].expand(NS, 1, N, N)
fl = {}
with torch.no_grad():
    for s in SIG:
        cv = cond_vec(np.full(NS, s), np.full(NS, zz[c])).to(dev)
        u = FA.decode(eps, cv, cimg, H=N)[:, 0].cpu().numpy()
        fl[s] = np.sinh(np.clip(u, -4, 4)) * DS
cc = (N - 1) / 2; yy, xx = np.indices((N, N)); rb = np.hypot(yy - cc, xx - cc).astype(int)
prof = lambda m: np.bincount(rb.ravel(), m.ravel()) / np.bincount(rb.ravel())

fig, ax = plt.subplots(4, 4, figsize=(13, 13.3), constrained_layout=True)
lo, hi = np.percentile(x[c], [1, 99.9]); DL = 0.3
kw = dict(origin="lower", interpolation="nearest")
ax[0, 0].imshow(x[c], cmap="magma", vmin=lo, vmax=hi, **kw); ax[0, 0].set_title("CDM (log10 Σ)")
ax[1, 0].imshow(x[c], cmap="magma", vmin=lo, vmax=hi, **kw); ax[1, 0].set_title("CDM (input to flow A)")
for j, s in enumerate(SIG, 1):
    ax[0, j].imshow(y[pr[s]], cmap="magma", vmin=lo, vmax=hi, **kw); ax[0, j].set_title(f"true SIDM σ={s}")
    ax[1, j].imshow(x[c] + fl[s][0], cmap="magma", vmin=lo, vmax=hi, **kw); ax[1, j].set_title(f"flow A sample, σ={s}")
    ax[2, j].imshow(y[pr[s]] - x[c], cmap="RdBu_r", vmin=-DL, vmax=DL, **kw); ax[2, j].set_title(f"true Δ, σ={s}")
    im = ax[3, j].imshow(fl[s][0], cmap="RdBu_r", vmin=-DL, vmax=DL, **kw); ax[3, j].set_title(f"flow A Δ (same sample), σ={s}")
fig.colorbar(im, ax=ax[2:, 1:], shrink=0.5, label="Δ = log10 SIDM − log10 CDM [dex]")
for a_ in ax[:2].ravel().tolist() + ax[2:, 1:].ravel().tolist():
    a_.set_xticks([]); a_.set_yticks([])
r = np.arange(rb.max() + 1)
cols = dict(zip(SIG, ("#2a9d8f", "#e9a03b", "#c0392b")))
for gi, rmax in ((2, 48), (3, 10)):
    g = ax[gi, 0]
    for s in SIG:
        P = np.stack([prof(d) for d in fl[s]])
        g.fill_between(r, *np.percentile(P, [16, 84], 0), color=cols[s], alpha=0.25, lw=0)
        g.plot(r, np.median(P, 0), color=cols[s], ls="--", lw=1.5)
        g.plot(r, prof(y[pr[s]] - x[c]), color=cols[s], lw=2, label=f"σ={s}")
    g.axhline(0, color="0.5", lw=0.7); g.set_xlim(0, rmax); g.set_xlabel("r [px]"); g.set_ylabel("azimuthal mean Δ [dex]")
    g.set_title("radial Δ: true (solid), flow A median ±1σ of 16 (dashed)" if gi == 2 else "zoom: inner 10 px", fontsize=9)
ax[2, 0].legend(fontsize=8)
fig.suptitle(f"flow A, test cluster row {c} (z={zz[c]:.3f}): one latent draw decoded at each σ")
out = f"{HERE}/figures/flowA_four_models.png"; fig.savefig(out, dpi=110); print(out, "cdm row", c)
