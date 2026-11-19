"""Training pairs for the CDM -> SIDM(sigma) UNet, from bahamas_matched.pkl.

Each sample with a CDM channel gives one pair per SIDM channel present
(sigma = 0.1, 0.3, 1 cm^2/g) plus an identity pair (sigma = 0, target = CDM).
Maps are log10 total-mass maps (100x100, +baryons).

Re-centring: each SIDM stamp is centred on its own halo centre, up to
0.2 Mpc/h from the CDM one, which puts a fake dipole into the SIDM/CDM ratio.
The integer shift that maximises the cross-correlation of the log maps
(|shift| <= MAXSHIFT px) is applied to the target with np.roll; the wrapped
edge strips are zeroed in the per-pair loss weight (stored as the shift).

Split by the CDM halo's x position in the 400 Mpc/h box, so the 3 projections
and 4 redshifts of a cluster land in the same split: train x < 280,
val 280-340, test >= 340, dropping halos within BUFFER Mpc/h of a boundary
(0/400 included) so a cluster that moves between snapshots can't straddle it.

Output: data/pairs.npz
  x         (Nc,100,100) float32  log10 CDM map per CDM sample
  y         (P,100,100)  float32  log10 target map (re-centred)
  xi        (P,) int     row of x for each pair
  sigma     (P,) float   0, 0.1, 0.3 or 1
  shift     (P,2) int    (dy, dx) applied to the target
  split     (Nc,) str    train / val / test / buffer
  redshift, projection, log10_M200, match_id, x_Mpc_h  (Nc,)
usage: python3 prepare_data.py
"""
import os, pickle
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = "/Users/davidharvey/Work/swift/eiger_runs/zooms/hydro/images/bahamas"
STAMPS = f"{HERE}/../bahamas/bahamas_stamps.csv"
SIGMA = {1: 0.1, 2: 0.3, 3: 1.0}  # channel -> sigma
MAXSHIFT, BUFFER = 12, 3.0
EDGES = (280.0, 340.0)

meta, im = pickle.load(open(f"{IMG}/bahamas_matched.pkl", "rb"))
mask = meta["mask"]
keep = np.where(mask[:, 0])[0]  # needs a CDM input
logim = np.log10(im[keep]).astype(np.float32)  # NaN for missing channels
mk = mask[keep]
print(len(keep), "samples with a CDM map")

# split by CDM halo x position
st = pd.read_csv(STAMPS, keep_default_na=False).set_index("idx_combined")
xpos = st.loc[meta["idx_combined_cdm"][keep], "x_Mpc_h"].values
dist = np.min(np.abs(np.stack([xpos - e for e in (0.0, *EDGES, 400.0)])), axis=0)
split = np.where(xpos < EDGES[0], "train", np.where(xpos < EDGES[1], "val", "test")).astype("<U6")
split[dist < BUFFER] = "buffer"
print({s: int((split == s).sum()) for s in ("train", "val", "test", "buffer")})


def best_shift(a, b):
    """(dy, dx) such that np.roll(b, (dy, dx)) best matches a."""
    a, b = a - a.mean(), b - b.mean()
    xc = np.fft.ifft2(np.fft.fft2(a) * np.conj(np.fft.fft2(b))).real
    d = np.arange(100); d = (d + 50) % 100 - 50
    ok = np.abs(d) <= MAXSHIFT
    xc = np.where(ok[:, None] & ok[None, :], xc, -np.inf)
    dy, dx = np.unravel_index(np.argmax(xc), xc.shape)
    return int(d[dy]), int(d[dx])


x = logim[:, 0]
ys, xi, sig, shifts = [], [], [], []
for n in range(len(keep)):
    ys.append(x[n]); xi.append(n); sig.append(0.0); shifts.append((0, 0))
    for c, s in SIGMA.items():
        if not mk[n, c]:
            continue
        t = logim[n, c]
        dy, dx = best_shift(x[n], t)
        ys.append(np.roll(t, (dy, dx), (0, 1))); xi.append(n); sig.append(s); shifts.append((dy, dx))
y = np.stack(ys).astype(np.float32)
shifts = np.array(shifts)
sig = np.array(sig)
nz = np.abs(shifts[sig > 0]).max(1)
print(len(y), "pairs;", {s: int((sig == s).sum()) for s in (0.0, 0.1, 0.3, 1.0)})
print("re-centring shift (max |component|, px): median", np.median(nz),
      " >=3 px:", int((nz >= 3).sum()), " >=8 px:", int((nz >= 8).sum()))

os.makedirs(f"{HERE}/data", exist_ok=True)
np.savez(f"{HERE}/data/pairs.npz", x=x, y=y, xi=np.array(xi), sigma=sig, shift=shifts,
         split=split, redshift=meta["redshift"][keep], projection=meta["projection"][keep],
         log10_M200=meta["log10_M200"][keep], match_id=meta["match_id"][keep], x_Mpc_h=xpos)
print("wrote", f"{HERE}/data/pairs.npz")
