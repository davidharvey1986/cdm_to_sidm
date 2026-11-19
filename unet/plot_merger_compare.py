"""One BAHAMAS cluster (default the merger, row 1951) at sigma = 0.1/0.3/1: true SIDM vs the two-scale converter
(template + small-scale map, no CNN; trained on darkskies -> cross-suite, and on BAHAMAS -> in-suite) vs flow A
(BAHAMAS-trained, one sample). Maps on the central 96x96 px.
usage: python3 plot_merger_compare.py [row]"""
import sys, json, os
import numpy as np, torch, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import two_scale as T2
from flows import Flow, cond_vec, mark_initialized

HERE = os.path.dirname(os.path.abspath(__file__))
row = int(sys.argv[1]) if len(sys.argv) > 1 else 1951
D = np.load(f"{HERE}/data/pairs.npz", allow_pickle=True)
x, y, xi, sig, sh, zz = (D[k] for k in ("x", "y", "xi", "sigma", "shift", "redshift"))
SIG = (0.1, 0.3, 1.0)
pr = {s: np.where((xi == row) & (sig == s))[0][0] for s in SIG}
c = x[row, 2:98, 2:98].astype(np.float64); z = float(zz[row])
true = {s: y[pr[s], 2:98, 2:98].astype(np.float64) for s in SIG}
conv = {}
for tag in ("darkskies", "bahamas"):
    J = json.load(open(f"{HERE}/runs/two_scale_{tag}_nocnn/template.json"))
    for s in SIG:
        sg = np.array([s]); Lc = T2.smooth(c[None]); Sc = c[None] - Lc
        d = T2.predict_large(Lc, sg, np.array([z]), J["template"])
        conv[tag, s] = (c[None] + d + (T2.apply_small(Sc, sg, J["small"]) - Sc))[0]
dev = T2.dev; torch.manual_seed(3)
ck = torch.load(f"{HERE}/runs/flowA/model.pt", map_location=dev, weights_only=False)
FA = Flow(**ck["arch"]).to(dev); FA.load_state_dict(ck["state"]); mark_initialized(FA); FA.eval()
eps = FA.sample_eps(1, dev, H=96); cimg = torch.tensor((c - ck["norm"]["mu"]) / ck["norm"]["sd"], dtype=torch.float32, device=dev)[None, None]
flow = {}
with torch.no_grad():
    for s in SIG:
        u = FA.decode(eps, cond_vec(np.array([s]), np.array([z])).to(dev), cimg, H=96)[0, 0].cpu().numpy()
        flow[s] = c + np.sinh(np.clip(u, -4, 4)) * 0.1

cols = [("true SIDM", true), ("converter (trained on darkskies)", {s: conv["darkskies", s] for s in SIG}),
        ("converter (trained on BAHAMAS)", {s: conv["bahamas", s] for s in SIG}), ("flow A sample (BAHAMAS)", flow)]
fig, ax = plt.subplots(6, 5, figsize=(19, 23.5), constrained_layout=True)
lo, hi = np.percentile(c, [1, 99.9]); kw = dict(origin="lower", interpolation="nearest")
for i, s in enumerate(SIG):
    a0, a1 = ax[2 * i], ax[2 * i + 1]
    a0[0].imshow(c, cmap="magma", vmin=lo, vmax=hi, **kw); a0[0].set_ylabel(f"σ = {s}\nlog10 Σ", fontsize=12); a0[0].set_title("CDM")
    for j, (nm, M) in enumerate(cols, 1):
        a0[j].imshow(M[s], cmap="magma", vmin=lo, vmax=hi, **kw); a0[j].set_title(nm, fontsize=10)
        im = a1[j].imshow(M[s] - c, cmap="RdBu_r", vmin=-.3, vmax=.3, **kw)
    a1[1].set_ylabel("Δ = map − CDM", fontsize=11)
    g = a1[0]; rr = np.arange(48) + 0.5; rr[0] = 0.35
    for (nm, M), cl, ls in zip(cols, ("k", "#c0392b", "#2a9d8f", "#3a6fb0"), ("-", "--", ":", "-.")):
        g.plot(rr, T2.rprof((M[s] - c)[None])[0], color=cl, ls=ls, lw=2 if cl == "k" else 1.6, label=nm)
    g.axhline(0, color="0.5", lw=0.7); g.set_xscale("log"); g.set_ylim(-0.35, 0.15); g.set_xlabel("r [px]"); g.set_title("azimuthal mean Δ", fontsize=10)
    if i == 0: g.legend(fontsize=7)
for a_ in ax.ravel():
    if a_ not in ax[1::2, 0]: a_.set_xticks([]); a_.set_yticks([])
fig.colorbar(im, ax=ax[1::2, 1:], shrink=0.3, label="Δ [dex]")
fig.suptitle(f"BAHAMAS test row {row} (merger, z={z}): true vs two-scale converter vs flow A", fontsize=14)
out = f"{HERE}/figures/merger_compare_{row}.png"; fig.savefig(out, dpi=85); print(out)
