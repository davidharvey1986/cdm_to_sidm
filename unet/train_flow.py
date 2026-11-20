"""Train the CDM -> SIDM normalising flows (flows.py) on data/pairs.npz.

Maps are cropped to the central 96x96 px; only pairs whose re-centring shift is <= 2 px are used
(the wrapped strips then fall outside the crop). Small Gaussian dequantisation noise is added.

--model A  conditional flow  p(u | CDM map, sigma, z),  u = asinh(delta / DSCALE), delta = log10 SIDM - log10 CDM
           (asinh compresses the +-1-2 dex tails from moved subhaloes; delta = DSCALE sinh(u). Its Jacobian
           does not depend on the model or sigma, so it is left out of the NLL)
           sigma > 0 pairs only (delta is identically 0 at sigma = 0).
           Transform a CDM map: eps = 0 (mode) or eps ~ N(0, T^2) (samples) -> delta -> SIDM map.
--model B  one flow for every model  p(log map | sigma, z), CDM is sigma = 0, trained on pairs
           (CDM, SIDM_sigma) with NLL of both maps + LAMBDA * mean (eps_CDM - eps_SIDM)^2,
           which ties a cluster's CDM and SIDM maps to the same latent.
           Transform: eps = f(CDM; 0, z) -> SIDM map = f^-1(eps; sigma, z).
Extra conditioning (model A): if the data file has cx (Nx, c1, 100, 100; per CDM map) and/or cy (P, c2, 100, 100; per pair),
they are appended to the CDM map as conditioning channels (each standardised on the train split), e.g. data/pairs_stars.npz:
x, y = CDM, SIDM stars; cx = CDM DM; cy = SIDM DM - CDM DM. A per-pair ok mask is used if present.
--offset f: hybrid target (prepare_gas_hybrid.py).
--scale-aug a: per-pair stellar-scale augmentation (stars flow; validation is not augmented), see stars_scale_check.py.
usage: python3 train_flow.py --model A --name flowA [--epochs 40] [--holdout 0.3] [--data darkskies_pairs.npz] [--dscale 0.1] [--scale-aug 0.5]
"""
import os, json, time, argparse, csv
import numpy as np
import torch
from flows import Flow, cond_vec

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--model", choices=("A", "B"), required=True)
ap.add_argument("--name", required=True)
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--lr", type=float, default=5e-4)
ap.add_argument("--levels", type=int, default=3)
ap.add_argument("--K", type=int, default=8)
ap.add_argument("--hidden", type=int, default=96)
ap.add_argument("--lam", type=float, default=1.0, help="model B latent-alignment weight")
ap.add_argument("--noise", type=float, default=0.003, help="dequantisation noise, dex")
ap.add_argument("--dscale", type=float, default=0.1, help="asinh scale of delta, dex")
ap.add_argument("--holdout", type=float, default=None)
ap.add_argument("--scale-aug", type=float, default=0.0, help="train-time scale augmentation: multiply each pair's CDM and SIDM linear maps by the same 10^U(-a, a)")
ap.add_argument("--scale-floor", type=float, default=1e11, help="floor of the log maps, re-applied after the rescaling")
ap.add_argument("--floor-mode", choices=("soft", "hard"), default="soft", help="soft: log10(Sigma + floor) (stars); hard: log10 max(Sigma, floor) (gas)")
ap.add_argument("--scale-aug-cx", type=int, nargs="*", default=[], help="cx channels (soft floor 1e11, e.g. CDM stars) rescaled by their own independent factor")
ap.add_argument("--data", default="pairs.npz", help="file in data/: pairs.npz (BAHAMAS) or darkskies_pairs.npz")
ap.add_argument("--offset", default=None, help="file in data/ with off_ring (P, 48): a per-pair radial mean change subtracted from the SIDM map after augmentation (hybrid; prepare_gas_hybrid.py)")
ap.add_argument("--max-steps", type=int, default=None)
ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
ap.add_argument("--no-wandb", action="store_true")
ap.add_argument("--wandb-project", default="cdm_to_sidm_unet")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
torch.manual_seed(a.seed); np.random.seed(a.seed)
DSCALE, C0, C1 = a.dscale, 2, 98
out = f"{HERE}/runs/{a.name}"
os.makedirs(out, exist_ok=True)

D = np.load(f"{HERE}/data/{a.data}", allow_pickle=True)
x, y, xi, sigma, shift, split, zz = (D[k] for k in ("x", "y", "xi", "sigma", "shift", "split", "redshift"))
x, y = x[:, C0:C1, C0:C1], y[:, C0:C1, C0:C1]
ok = (np.abs(shift).max(1) <= 2) & (sigma > 0)
if "ok" in D.files: ok &= D["ok"]
cx = D["cx"][..., C0:C1, C0:C1] if "cx" in D.files else None; cy = D["cy"][..., C0:C1, C0:C1] if "cy" in D.files else None
if a.holdout is not None:
    ok &= sigma != a.holdout
psplit = split[xi]
train, val = np.where(ok & (psplit == "train"))[0], np.where(ok & (psplit == "val"))[0]
mu, sd = float(x[split == "train"].mean()), float(x[split == "train"].std())
norm = dict(mu=mu, sd=sd, dscale=DSCALE, crop=(C0, C1), data=a.data, offset=a.offset)
OFF = np.load(f"{HERE}/data/{a.offset}")["off_ring"] if a.offset else None
_yy, _xx = np.indices((C1 - C0, C1 - C0)); RBC = np.minimum(np.hypot(_yy - (C1 - C0 - 1) / 2, _xx - (C1 - C0 - 1) / 2).astype(int), 47)
cstat = lambda c: ([float(c[:, j].mean()) for j in range(c.shape[1])], [float(c[:, j].std()) for j in range(c.shape[1])])
if cx is not None: norm["cx_mu"], norm["cx_sd"] = cstat(cx[split == "train"])
if cy is not None: norm["cy_mu"], norm["cy_sd"] = cstat(cy[train])
NCOND = 1 + (0 if cx is None else cx.shape[1]) + (0 if cy is None else cy.shape[1])
cfg = dict(vars(a), **norm)
json.dump(cfg, open(f"{out}/config.json", "w"), indent=1)
print(f"model {a.model}; train pairs {len(train)}, val pairs {len(val)}; device {a.device}")

model = Flow(levels=a.levels, K=a.K, hidden=a.hidden, cond_img=(a.model == "A"), ncond=NCOND).to(a.device)
print(f"{sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
opt = torch.optim.Adam(model.parameters(), lr=a.lr)
steps = len(train) // a.batch if a.max_steps is None else min(a.max_steps, len(train) // a.batch)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * steps,
                                            pct_start=max(0.05, 3 / (a.epochs * steps)))
T = lambda v: torch.tensor(np.ascontiguousarray(v), dtype=torch.float32, device=a.device)[:, None]


def batch(idx, augment):
    lc, ls = x[xi[idx]], y[idx]
    ex = []
    if cx is not None:
        cxr = cx[xi[idx]]
        if augment and a.scale_aug > 0 and a.scale_aug_cx:
            cxr = cxr.copy()
            for j in a.scale_aug_cx:
                g = 10 ** np.random.uniform(-a.scale_aug, a.scale_aug, len(idx))[:, None, None]
                cxr[:, j] = np.log10(g * np.maximum(10 ** cxr[:, j].astype(np.float64) - 1e11, 0) + 1e11)
        ex.append((cxr - np.array(norm["cx_mu"])[:, None, None]) / np.array(norm["cx_sd"])[:, None, None])
    if cy is not None: ex.append((cy[idx] - np.array(norm["cy_mu"])[:, None, None]) / np.array(norm["cy_sd"])[:, None, None])
    ex = np.concatenate(ex, 1).astype(np.float32) if ex else None
    if augment and a.scale_aug > 0:  # same random factor for a pair's CDM and SIDM map: removes the absolute scale as information
        g = 10 ** np.random.uniform(-a.scale_aug, a.scale_aug, len(idx))[:, None, None]; F0 = a.scale_floor
        if a.floor_mode == "soft":
            lc, ls = (np.log10(g * np.maximum(10 ** t.astype(np.float64) - F0, 0) + F0).astype(np.float32) for t in (lc, ls))
        else:  # floored pixels count as empty, so scaling up does not lift the floor
            lc, ls = (np.log10(np.maximum(g * np.where(t > np.log10(F0) + 1e-4, 10 ** t.astype(np.float64), 0), F0)).astype(np.float32) for t in (lc, ls))
    if OFF is not None: ls = (ls - OFF[idx][:, RBC]).astype(np.float32)  # hybrid: residual about the converter mean (radial, so before/after rotation is the same)
    if augment:
        k, f = np.random.randint(4), np.random.rand() < 0.5
        lc, ls = (np.rot90(t, k, axes=(1, 2)) for t in (lc, ls))
        if ex is not None: ex = np.rot90(ex, k, axes=(2, 3))
        if f:
            lc, ls = (t[:, :, ::-1] for t in (lc, ls))
            if ex is not None: ex = ex[..., ::-1]
    n = lambda t: t + a.noise * np.random.randn(*t.shape).astype(np.float32)
    cv = cond_vec(sigma[idx], zz[xi[idx]]).to(a.device)
    if a.model == "A":
        ci = T((lc - mu) / sd)
        if ex is not None: ci = torch.cat([ci, torch.tensor(np.ascontiguousarray(ex), dtype=torch.float32, device=a.device)], 1)
        return dict(x=T(np.arcsinh(n(ls - lc) / DSCALE)), cimg=ci, cv=cv)
    cv0 = cond_vec(np.zeros(len(idx)), zz[xi[idx]]).to(a.device)
    return dict(xc=T((n(lc) - mu) / sd), xs=T((n(ls) - mu) / sd), cv0=cv0, cv=cv)


def loss_fn(b):
    if a.model == "A":
        nll, _ = model.nll(b["x"], b["cv"], b["cimg"])
        return nll.mean(), dict(nll=nll.mean().item())
    B = b["xc"].shape[0]
    nll, eps = model.nll(torch.cat([b["xc"], b["xs"]]), torch.cat([b["cv0"], b["cv"]]))
    align = sum(((e[:B] - e[B:]) ** 2).flatten(1).sum(1) for e in eps) / b["xc"][0].numel()
    return nll.mean() + a.lam * align.mean(), dict(nll=nll.mean().item(), align=align.mean().item())


run = None
if not a.no_wandb:
    import wandb
    run = wandb.init(project=a.wandb_project, name=a.name, config=cfg, dir=out, tags=["flow", "flow" + a.model])
    open(f"{out}/wandb_id.txt", "w").write(run.id)

best, hist = np.inf, []
for ep in range(a.epochs):
    t0 = time.time(); model.train(); perm = np.random.permutation(train); tl = []
    for s in range(steps):
        loss, parts = loss_fn(batch(perm[s * a.batch:(s + 1) * a.batch], True))
        if not torch.isfinite(loss):
            print("non-finite loss, skipping step", flush=True); opt.zero_grad(); continue
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 50.0)
        opt.step(); sched.step(); tl.append([loss.item()] + list(parts.values()))
    model.eval(); vl = []
    with torch.no_grad():
        for s in range(0, len(val), 64):
            l, parts = loss_fn(batch(val[s:s + 64], False))
            vl.append(np.array([l.item()] + list(parts.values())) * len(val[s:s + 64]))
    vl = np.sum(vl, 0) / len(val); tl = np.mean(tl, 0)
    keys = ["loss"] + list(parts)
    rec = dict(epoch=ep, **{f"train_{k}": float(v) for k, v in zip(keys, tl)},
               **{f"val_{k}": float(v) for k, v in zip(keys, vl)}, sec=time.time() - t0)
    hist.append(rec)
    print("  ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in rec.items()), flush=True)
    if vl[0] < best:
        best = vl[0]
        torch.save(dict(state=model.state_dict(), arch=dict(levels=a.levels, K=a.K, hidden=a.hidden,
                        cond_img=(a.model == "A"), ncond=NCOND), norm=norm, config=cfg, epoch=ep), f"{out}/model.pt")
    with open(f"{out}/history.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, list(hist[0])); w.writeheader(); w.writerows(hist)
    if run is not None:
        run.log(dict(rec, lr=sched.get_last_lr()[0], best_val_loss=best), step=ep)
print("best val", best)
if run is not None:
    run.summary["best_val_loss"] = best; run.finish()
