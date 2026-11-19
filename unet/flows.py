"""Glow-style normalising flow on 96x96 single-channel maps, with conditioning.

Bijection x <-> eps (a list of standard-normal tensors, one per level):
  per level: squeeze (2x2 -> channels), K x [ActNorm, invertible 1x1 conv (LU), affine coupling],
  then split half the channels off through a conditional Gaussian prior, written as an affine
  step eps = (z - mu) * exp(-logs) so that eps is the full latent of the bijection.
Conditioning (both modes): a vector c = [sigma, sqrt(sigma), redshift], broadcast as constant channels.
cond_img=True adds an image encoder (the CDM map) whose features enter every coupling and prior
at the matching resolution (model A: p(delta | CDM, sigma, z)). cond_img=False is model B
(one flow for all dark-matter models, p(map | sigma, z)).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

NCS = 3  # [sigma, sqrt sigma, z]
LSMAX, SHMAX = 0.5, 4.0  # coupling log-scale and shift bounds: keep the inverse stable


def cond_vec(sigma, z):
    s = torch.as_tensor(sigma, dtype=torch.float32)
    return torch.stack([s, s.sqrt(), torch.as_tensor(z, dtype=torch.float32)], -1)


def squeeze(x):
    B, C, H, W = x.shape
    return x.view(B, C, H // 2, 2, W // 2, 2).permute(0, 1, 3, 5, 2, 4).reshape(B, C * 4, H // 2, W // 2)


def unsqueeze(x):
    B, C, H, W = x.shape
    return x.view(B, C // 4, 2, 2, H, W).permute(0, 1, 4, 2, 5, 3).reshape(B, C // 4, H * 2, W * 2)


class ZeroConv(nn.Conv2d):
    def __init__(self, cin, cout):
        super().__init__(cin, cout, 3, padding=1)
        nn.init.zeros_(self.weight); nn.init.zeros_(self.bias)


class ActNorm(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.loc = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.logs = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.initialized = False

    def forward(self, x, ld, rev=False):
        if not rev and not self.initialized and self.training:
            with torch.no_grad():
                self.loc.data.copy_(-x.mean((0, 2, 3), keepdim=True))
                self.logs.data.copy_(-(x.std((0, 2, 3), keepdim=True) + 1e-6).log())
            self.initialized = True
        hw = x.shape[2] * x.shape[3]
        if not rev:
            return (x + self.loc) * self.logs.exp(), ld + hw * self.logs.sum()
        return x * (-self.logs).exp() - self.loc, ld - hw * self.logs.sum()


class InvConv(nn.Module):
    """Invertible 1x1 convolution, LU-parameterised (log|det| = sum logs)."""
    def __init__(self, c):
        super().__init__()
        w = torch.linalg.qr(torch.randn(c, c))[0]
        P, L, U = torch.linalg.lu(w)
        d = torch.diag(U)
        self.register_buffer("P", P)
        self.register_buffer("sign", torch.sign(d))
        self.register_buffer("lmask", torch.tril(torch.ones(c, c), -1))
        self.register_buffer("eye", torch.eye(c))
        self.L = nn.Parameter(L)
        self.U = nn.Parameter(torch.triu(U, 1))
        self.logs = nn.Parameter(d.abs().log())

    def weight(self):
        L = self.L * self.lmask + self.eye
        U = self.U * self.lmask.T + torch.diag(self.sign * self.logs.exp())
        return self.P @ L @ U

    def forward(self, x, ld, rev=False):
        hw = x.shape[2] * x.shape[3]
        W = self.weight()
        if not rev:
            return F.conv2d(x, W[:, :, None, None]), ld + hw * self.logs.sum()
        Wi = torch.inverse(W.cpu().double()).to(device=x.device, dtype=x.dtype)
        return F.conv2d(x, Wi[:, :, None, None]), ld - hw * self.logs.sum()


class Coupling(nn.Module):
    def __init__(self, c, cc, h):
        super().__init__()
        self.ca = c // 2
        cb = c - self.ca
        self.net = nn.Sequential(nn.Conv2d(self.ca + cc, h, 3, padding=1), nn.ReLU(),
                                 nn.Conv2d(h, h, 1), nn.ReLU(), ZeroConv(h, 2 * cb))

    def forward(self, x, cond, ld, rev=False):
        xa, xb = x[:, :self.ca], x[:, self.ca:]
        o = self.net(torch.cat([xa, cond], 1))
        shift, raw = o[:, 0::2], o[:, 1::2]
        logs = LSMAX * torch.tanh(raw)
        shift = SHMAX * torch.tanh(shift / SHMAX)
        scale = logs.exp()
        lds = logs.flatten(1).sum(1)
        if not rev:
            return torch.cat([xa, (xb + shift) * scale], 1), ld + lds
        return torch.cat([xa, xb / scale - shift], 1), ld - lds


class Step(nn.Module):
    def __init__(self, c, cc, h):
        super().__init__()
        self.an, self.ic, self.cp = ActNorm(c), InvConv(c), Coupling(c, cc, h)

    def forward(self, x, cond, ld, rev=False):
        if not rev:
            x, ld = self.an(x, ld); x, ld = self.ic(x, ld); return self.cp(x, cond, ld)
        x, ld = self.cp(x, cond, ld, True); x, ld = self.ic(x, ld, True); return self.an(x, ld, True)


class Encoder(nn.Module):
    """CDM map (+ scalar channels) at 96 px -> features at 48, 24, 12 px."""
    def __init__(self, cin, ce):
        super().__init__()
        self.b0 = nn.Sequential(nn.Conv2d(cin, ce, 3, padding=1), nn.ReLU(), nn.Conv2d(ce, ce, 3, padding=1), nn.ReLU())
        self.down = nn.ModuleList([nn.Sequential(nn.Conv2d(ce, ce, 3, stride=2, padding=1), nn.ReLU(),
                                                 nn.Conv2d(ce, ce, 3, padding=1), nn.ReLU()) for _ in range(3)])

    def forward(self, x):
        h, out = self.b0(x), []
        for d in self.down:
            h = d(h); out.append(h)
        return out


class Flow(nn.Module):
    def __init__(self, levels=3, K=8, hidden=96, cond_img=False, ce=32, ncond=1):
        super().__init__()
        self.nlev, self.cond_img = levels, cond_img
        self.enc = Encoder(ncond + NCS, ce) if cond_img else None  # ncond conditioning image channels
        cc = (ce if cond_img else 0) + NCS
        self.steps, self.priors, self.chans = nn.ModuleList(), nn.ModuleList(), []
        c = 1
        for l in range(levels):
            c *= 4
            last = l == levels - 1
            self.steps.append(nn.ModuleList([Step(c, cc, hidden) for _ in range(K)]))
            zc = c if last else c // 2
            self.priors.append(ZeroConv((0 if last else c - zc) + cc, 2 * zc))
            self.chans.append((c, zc))
            if not last:
                c -= zc

    def _conds(self, cvec, cimg, H):
        B = cvec.shape[0]
        feats = None
        if self.cond_img:
            feats = self.enc(torch.cat([cimg, cvec[:, :, None, None].expand(B, NCS, H, H)], 1))
        out = []
        for l in range(self.nlev):
            r = H >> (l + 1)
            sc = cvec[:, :, None, None].expand(B, NCS, r, r)
            out.append(torch.cat([feats[l], sc], 1) if feats is not None else sc)
        return out

    def encode(self, x, cvec, cimg=None):
        """x (B,1,96,96) -> eps list, log|det J| (B,)"""
        conds = self._conds(cvec, cimg, x.shape[-1])
        ld = torch.zeros(x.shape[0], device=x.device)
        eps, h = [], x
        for l in range(self.nlev):
            h = squeeze(h)
            for st in self.steps[l]:
                h, ld = st(h, conds[l], ld)
            c, zc = self.chans[l]
            if l < self.nlev - 1:
                z, h = h[:, :zc], h[:, zc:]
                mu, logs = self.priors[l](torch.cat([h, conds[l]], 1)).chunk(2, 1)
            else:
                z = h
                mu, logs = self.priors[l](conds[l]).chunk(2, 1)
            eps.append((z - mu) * (-logs).exp())
            ld = ld - logs.flatten(1).sum(1)
        return eps, ld

    def decode(self, eps, cvec, cimg=None, H=96):
        conds = self._conds(cvec, cimg, H)
        ld = torch.zeros(cvec.shape[0], device=cvec.device)
        h = None
        for l in reversed(range(self.nlev)):
            if l == self.nlev - 1:
                mu, logs = self.priors[l](conds[l]).chunk(2, 1)
                h = eps[l] * logs.exp() + mu
            else:
                mu, logs = self.priors[l](torch.cat([h, conds[l]], 1)).chunk(2, 1)
                h = torch.cat([eps[l] * logs.exp() + mu, h], 1)
            for st in reversed(self.steps[l]):
                h, ld = st(h, conds[l], ld, rev=True)
            h = unsqueeze(h)
        return h

    def nll(self, x, cvec, cimg=None):
        """negative log-likelihood per dimension (nats), per sample; also returns eps"""
        eps, ld = self.encode(x, cvec, cimg)
        D = x[0].numel()
        lp = sum((-0.5 * e ** 2).flatten(1).sum(1) for e in eps) - 0.5 * D * math.log(2 * math.pi)
        return -(lp + ld) / D, eps

    def sample_eps(self, B, device, T=1.0, H=96):
        shapes, c, r = [], 1, H
        for l in range(self.nlev):
            r //= 2
            shapes.append((B, self.chans[l][1], r, r))
        return [T * torch.randn(s, device=device) for s in shapes]


def mark_initialized(model):
    for m in model.modules():
        if isinstance(m, ActNorm):
            m.initialized = True
