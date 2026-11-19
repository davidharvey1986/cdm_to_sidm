"""Train the sigma-conditioned UNet on data/pairs.npz (see prepare_data.py).

Loss on delta = log10 target - log10 CDM (dex):
  masked pixel MSE (wrapped strips from re-centring masked out)
  + LAMBDA_PROF * MSE of the azimuthally averaged delta profile (25 radial bins),
which weights the smooth core response that SIDM actually predicts.
Augmentation: random D4 (rotations + flips) applied jointly.

usage: python3 train.py --name test --epochs 2 [--holdout 0.3] [--device mps|cuda|cpu]
Writes runs/<name>/model.pt (best val loss), history.csv, config.json
"""
import os, json, time, argparse
import numpy as np
import torch
import torch.nn.functional as F
from sidm_unet import SIDMUNet, cond_features

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--epochs", type=int, default=60)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--width", type=int, default=32)
ap.add_argument("--lambda-prof", type=float, default=1.0)
ap.add_argument("--holdout", type=float, default=None, help="sigma left out of training, e.g. 0.3")
ap.add_argument("--max-steps", type=int, default=None, help="cap steps per epoch (smoke tests)")
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else
                ("mps" if torch.backends.mps.is_available() else "cpu"))
ap.add_argument("--data", default=f"{HERE}/data/pairs.npz")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--no-wandb", action="store_true", help="don't log to Weights & Biases")
ap.add_argument("--wandb-project", default="cdm_to_sidm_unet")
ap.add_argument("--input-space", choices=("log", "linear"), default="log",
                help="CDM input as log10 map or linear mass map (both globally standardised)")
ap.add_argument("--target", choices=("logratio", "ratio"), default="logratio",
                help="predict log10(SIDM/CDM) or SIDM/CDM - 1")
ap.add_argument("--agree-weight", action="store_true",
                help="weight the pixel and profile loss by data/agree_weights.npy (make_agreement_weights.py): up where CDM and "
                     "target show the same substructure, down where only one does. Training only; val loss unweighted")
a = ap.parse_args()
torch.manual_seed(a.seed); np.random.seed(a.seed)
out = f"{HERE}/runs/{a.name}"
os.makedirs(out, exist_ok=True)
json.dump(vars(a), open(f"{out}/config.json", "w"), indent=1)
run = None
if not a.no_wandb:
    import wandb
    run = wandb.init(project=a.wandb_project, name=a.name, config=vars(a), dir=out,
                     tags=["holdout_" + str(a.holdout)] if a.holdout is not None else ["all_sigma"])
    open(f"{out}/wandb_id.txt", "w").write(run.id)  # evaluate.py logs into the same run

D = np.load(a.data, allow_pickle=True)
x, y, xi, sigma, shift, split, zz = (D[k] for k in ("x", "y", "xi", "sigma", "shift", "split", "redshift"))
psplit = split[xi]
AW = np.load(f"{HERE}/data/agree_weights.npy", mmap_mode="r") if a.agree_weight else None
train = np.where((psplit == "train") & ((sigma != a.holdout) if a.holdout is not None else True))[0]
val = np.where((psplit == "val") & ((sigma != a.holdout) if a.holdout is not None else True))[0]
xtr = x[split == "train"].astype(np.float64)
if a.input_space == "linear":
    xtr = 10.0 ** xtr
mu, sd = float(xtr.mean()), float(xtr.std())
del xtr
xin_of = (lambda lx: (10.0 ** lx.astype(np.float64) - mu) / sd) if a.input_space == "linear" else (lambda lx: (lx - mu) / sd)
tgt_of = (lambda ly, lx: 10.0 ** (ly - lx).astype(np.float64) - 1) if a.target == "ratio" else (lambda ly, lx: ly - lx)
print(f"input {a.input_space}, target {a.target}; device {a.device}; train pairs {len(train)}, val pairs {len(val)}; holdout sigma {a.holdout}")

# radial bins for the profile loss
yy, xx = np.indices((100, 100))
rbin = np.minimum((np.hypot(yy - 49.5, xx - 49.5) / 2).astype(int), 24)
R = torch.tensor(np.eye(25)[rbin].reshape(-1, 25), dtype=torch.float32)  # (10000, 25)


def weight_mask(s):
    """Zero the strips that np.roll wrapped around."""
    w = np.ones((len(s), 100, 100), np.float32)
    for k, (dy, dx) in enumerate(s):
        if dy > 0: w[k, :dy] = 0
        if dy < 0: w[k, dy:] = 0
        if dx > 0: w[k, :, :dx] = 0
        if dx < 0: w[k, :, dx:] = 0
    return w


def batch(idx, augment, agree=False):
    xb = xin_of(x[xi[idx]])
    db = tgt_of(y[idx], x[xi[idx]])
    wb = weight_mask(shift[idx])
    if agree:
        wb = wb * np.asarray(AW[np.sort(idx)][np.argsort(np.argsort(idx))], np.float32)  # sorted read from the memmap
    if augment:
        k = np.random.randint(4); f = np.random.rand() < 0.5
        xb, db, wb = (np.rot90(t, k, axes=(1, 2)) for t in (xb, db, wb))
        if f:
            xb, db, wb = (t[:, :, ::-1] for t in (xb, db, wb))
    t = lambda v: torch.tensor(np.ascontiguousarray(v), dtype=torch.float32, device=a.device)[:, None]
    return t(xb), t(db), t(wb), cond_features(sigma[idx], zz[xi[idx]]).to(a.device)


Rd = R.to(a.device)


def loss_fn(pred, d, w):
    pix = ((pred - d) ** 2 * w).sum() / w.sum()
    pw = (w.flatten(1) @ Rd).clamp_min(1)
    prof = lambda v: ((v * w).flatten(1) @ Rd) / pw
    return pix + a.lambda_prof * ((prof(pred) - prof(d)) ** 2).mean(), pix


model = SIDMUNet(width=a.width).to(a.device)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
steps = len(train) // a.batch if a.max_steps is None else min(a.max_steps, len(train) // a.batch)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * steps)
print(f"{sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters, {steps} steps/epoch")

# fixed validation cluster with all three SIDM targets, for the wandb example panels
_full = [n for n in np.unique(xi[val]) if set(sigma[xi == n]) >= {0.1, 0.3, 1.0}]
EX = sorted(_full, key=lambda n: -x[n][45:55, 45:55].mean())[len(_full) // 4]  # a fairly massive one


def example_figure():
    """Rows: sigma 0.1, 0.3, 1. Columns: CDM map, true delta, predicted delta, radial delta profiles."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [np.where((xi == EX) & (sigma == s))[0][0] for s in (0.1, 0.3, 1.0)]
    xb, db, wb_, cb = batch(np.array(rows), False)
    with torch.no_grad():
        pb = model(xb, cb)
    d, p, w = (t[:, 0].cpu().numpy() for t in (db, pb, wb_))
    fig, ax = plt.subplots(3, 4, figsize=(12, 8.6), facecolor="white")
    r = np.hypot(yy - 49.5, xx - 49.5).astype(int)
    for k, s in enumerate((0.1, 0.3, 1.0)):
        ax[k, 0].imshow(x[EX], origin="lower", cmap="magma")
        for j, m in ((1, d[k]), (2, p[k])):
            ax[k, j].imshow(np.where(w[k] > 0, m, np.nan), origin="lower", cmap="RdBu_r", vmin=-0.4, vmax=0.4)
        for m, c, lab in ((d[k], "#1f1f1e", "true"), (p[k], "#2a78d6", "UNet")):
            prof = np.bincount(r.ravel(), (m * w[k]).ravel(), 71) / np.maximum(np.bincount(r.ravel(), w[k].ravel(), 71), 1)
            ax[k, 3].plot(prof[:45], color=c, lw=2, label=lab)
        ax[k, 3].axhline(0, color="#6b6a63", lw=0.6); ax[k, 3].set_ylim(-0.15, 0.15)
        ax[k, 0].set_ylabel(f"sigma = {s}")
        for j in range(3):
            ax[k, j].set_xticks([]); ax[k, j].set_yticks([])
    tl_ = "SIDM/CDM - 1" if a.target == "ratio" else "delta (dex)"
    for j, t in enumerate(("log10 CDM", f"true {tl_}", f"predicted {tl_}", "radial profile")):
        ax[0, j].set_title(t, fontsize=10)
    ax[0, 3].legend(frameon=False, fontsize=8); ax[2, 3].set_xlabel("radius [px]")
    fig.tight_layout()
    img = wandb.Image(fig, caption=f"val cluster {D['match_id'][EX]} p{D['projection'][EX]}")
    plt.close(fig)
    return img


best, hist = np.inf, []
for ep in range(a.epochs):
    t0 = time.time(); model.train(); perm = np.random.permutation(train); tl = []
    for s in range(steps):
        xb, db, wb, cb = batch(perm[s * a.batch:(s + 1) * a.batch], True, a.agree_weight)
        loss, _ = loss_fn(model(xb, cb), db, wb)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tl.append(loss.item())
    model.eval(); vl, vp, zero = [], [], []
    with torch.no_grad():
        for s in range(0, len(val), 64):
            xb, db, wb, cb = batch(val[s:s + 64], False)
            l, p = loss_fn(model(xb, cb), db, wb); vl.append(l.item() * len(xb))
            zero.append(loss_fn(torch.zeros_like(db), db, wb)[0].item() * len(xb))
    vl, zero = sum(vl) / len(val), sum(zero) / len(val)
    hist.append(dict(epoch=ep, train=np.mean(tl), val=vl, val_zero_baseline=zero, sec=time.time() - t0))
    print(f"epoch {ep:3d}  train {np.mean(tl):.5f}  val {vl:.5f}  (predict-no-change {zero:.5f})  "
          f"{time.time()-t0:.0f}s", flush=True)
    if vl < best:
        best = vl
        torch.save(dict(state=model.state_dict(), arch=dict(width=a.width),
                        norm=dict(mu=mu, sd=sd, input_space=a.input_space, target=a.target), config=vars(a), epoch=ep), f"{out}/model.pt")
    import csv
    with open(f"{out}/history.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, list(hist[0])); w.writeheader(); w.writerows(hist)
    if run is not None:
        log = dict(epoch=ep, train_loss=np.mean(tl), val_loss=vl, val_loss_no_change=zero,
                   val_gain_over_no_change=1 - vl / zero, best_val_loss=best,
                   lr=sched.get_last_lr()[0], epoch_sec=time.time() - t0)
        if ep % 5 == 0 or ep == a.epochs - 1:
            log["examples"] = example_figure()
        run.log(log, step=ep)
print("best val", best, "->", f"{out}/model.pt")
if run is not None:
    run.summary["best_val_loss"] = best
    run.finish()
