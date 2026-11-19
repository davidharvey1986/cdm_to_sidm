"""Show the smoothing difference in the 20 kpc product: remade SIDM stamps
(current darkskies_images, nn=100) vs CDM stamps of the same halos (pre-22-Oct
code). Residual = map / (3x3 box-smoothed map) - 1; roughness = std of the
residual in a 300-900 kpc annulus. Output figures/smoothing_20kpc.png"""
import os, glob, pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

D = "/Users/davidharvey/Work/swift/eiger_runs/zooms/hydro/images/darkskies"
HERE = os.path.dirname(os.path.abspath(__file__))
Z = {7: 0.374, 8: 0.25, 9: 0.124, 10: 0.0}


def load(f):
    return np.asarray(pickle.load(open(f, "rb"))[0])[0], pickle.load(open(f, "rb"))[1]


def residual(m):
    p = np.pad(m, 1, mode="edge")
    s = sum(p[i:i + m.shape[0], j:j + m.shape[1]] for i in range(3) for j in range(3)) / 9
    return m / s - 1


y, x = np.indices((100, 100))
R = np.hypot(x - 49.5, y - 49.5) * 20
ANN = (R > 300) & (R < 900)
rough = lambda m: residual(m)[ANN].std()

remade = []
for f in sorted(glob.glob(f"{D}/SIDM*/*/*_x.pkl")):
    m, meta = load(f)
    if "matched_by" in meta:
        model, snap, name = f.split("/")[-3:]
        cdm = load(f"{D}/CDM/{snap}/{name}")[0]
        old = load(f"{D}_mismatched_backup/{model}/{snap}/{name}")[0]
        remade.append(dict(f=f, model=model, snap=int(snap), halo=name[:-6], m=m, cdm=cdm,
                           r_new=rough(m), r_cdm=rough(cdm), r_old=rough(old)))
print(len(remade), "remade x-projections")
rn, rc, ro = (np.array([e[k] for e in remade]) for k in ("r_new", "r_cdm", "r_old"))
print("remade %.3f  cdm %.3f  old %.3f   remade>cdm in %d/%d" %
      (np.median(rn), np.median(rc), np.median(ro), (rn > rc).sum(), len(rn)))

# example: the entry whose remake roughness is closest to the median
ex = min(remade, key=lambda e: abs(e["r_new"] - np.median(rn)))
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED = "#1f1f1e", "#6b6a63"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED})

fig = plt.figure(figsize=(11, 6.6), facecolor="white")
gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.25], wspace=0.28, hspace=0.32)
ext = [-1, 1, -1, 1]
vmin, vmax = np.percentile(np.r_[ex["cdm"].ravel(), ex["m"].ravel()], [1, 99.9])
panels = [(ex["cdm"], f"CDM (pre-22-Oct code)\nroughness {ex['r_cdm']:.3f}"),
          (ex["m"], f"{ex['model']} remake (current code, nn=100)\nroughness {ex['r_new']:.3f}")]
for i, (m, t) in enumerate(panels):
    ax = fig.add_subplot(gs[0, i])
    im = ax.imshow(m, origin="lower", extent=ext, cmap="Blues", norm=LogNorm(vmin, vmax))
    ax.set_title(t, fontsize=9, color=INK)
    ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1])
    if i == 0:
        ax.set_ylabel("pMpc")
    ax2 = fig.add_subplot(gs[1, i])
    rr = ax2.imshow(residual(m), origin="lower", extent=ext, cmap="RdBu_r", vmin=-0.3, vmax=0.3)
    for r in (0.3, 0.9):
        ax2.add_patch(plt.Circle((0, 0), r, fill=False, ls="--", lw=0.8, color=INK))
    ax2.set_title("map / 3x3-smoothed - 1", fontsize=9, color=INK)
    ax2.set_xticks([-1, 0, 1]); ax2.set_yticks([-1, 0, 1]); ax2.set_xlabel("pMpc")
    if i == 0:
        ax2.set_ylabel("pMpc")
fig.colorbar(im, ax=[fig.axes[0], fig.axes[2]], shrink=0.85, pad=0.02,
             label=r"$\Sigma$ [M$_\odot$ pMpc$^{-2}$]")
fig.colorbar(rr, ax=[fig.axes[1], fig.axes[3]], shrink=0.85, pad=0.02, label="residual")

ax = fig.add_subplot(gs[:, 2])
bins = np.linspace(0.03, 0.18, 31)
for v, c, lab in [(rn, BLUE, "SIDM remakes (current code)"),
                  (rc, ORANGE, "CDM, same halos (pre-22-Oct)"),
                  (ro, AQUA, "old SIDM stamps (pre-22-Oct)")]:
    ax.hist(v, bins, histtype="step", lw=2, color=c, label=f"{lab}  median {np.median(v):.3f}")
ax.axvline(ex["r_new"], color=MUTED, lw=0.8, ls=":")
ax.set_xlabel("roughness: std of residual, 300-900 kpc")
ax.set_ylabel(f"x-projections (of {len(remade)})")
ax.grid(axis="y", color="#e1e0d9", lw=0.6); ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.set_ylim(0, ax.get_ylim()[1] * 1.3)
ax.legend(frameon=False, fontsize=8, loc="upper right")
ax.set_title("All 94 remade entries (x projection)", fontsize=9, color=INK)
fig.suptitle(f"20 kpc stamps: smoothing differs between code versions   "
             f"(example: {ex['halo']}, z={Z[ex['snap']]})", fontsize=11, color=INK)
os.makedirs(f"{HERE}/figures", exist_ok=True)
out = f"{HERE}/figures/smoothing_20kpc.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(out)
