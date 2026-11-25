"""
paper2_train.py -- Paper 2: f=8 spatial VAE with a strong decoder, LPIPS and a PatchGAN critic.

Model
  encoder : small, 64x64 -> 4 x 8 x 8 latent (256 dims, the SD-VAE convention of 4 channels)
  decoder : "strong" (~13 M, ResBlocks + attention at 8x8) or "small" (~1 M, for the ablation)
  loss    : Gaussian NLL with learnable sigma + w_lpips * LPIPS + beta * KL
            + adaptive-weight GAN term (hinge / wgan_gp / sobolev), switched on after a
              reconstruction-only warm-up (--gan_start_epoch)

Ablation ladder (run each with a different --tag):
  A  --decoder small  --w_lpips 0 --no_gan     the old Paper 2 model (pixel loss only)
  B  --decoder strong --w_lpips 0 --no_gan     capacity alone
  C  --decoder strong             --no_gan     + perceptual loss
  D  --decoder strong                          + GAN (hinge)      <- main model
  E  --decoder strong --critic wgan_gp         Kantorovich critic
  F  --decoder strong --critic sobolev         Sobolev critic

Eval reports, overall and per complexity quintile:
  PSNR (aggregate and per-image mean), SSIM, LPIPS, reconstruction FID
Complexity for the quintiles = high-frequency energy fraction of the image (training-free FFT),
the same kind of measure DDR-SR uses, so the per-quintile tables are directly comparable.

Usage (same data folders and splits as nested_vae_spectral.py):
  python paper2_train.py train --dataset cifar10 --root D:\\new_vae_research\\data --tag D_strong_gan
  (if a run stops, rerun the SAME train command with --resume added; it continues from the last epoch)
  python paper2_train.py eval  --dataset cifar10 --root D:\\new_vae_research\\data --tag D_strong_gan

Requires: pip install lpips torchmetrics torch-fidelity
"""
import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from paper2_decoder_gan import (StrongDecoder, PatchDiscriminator, AutoencoderGANLoss,
                                ResBlock, Norm)
from nested_vae_spectral import get_datasets


# ============================================================== model
class Downsample(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Encoder(nn.Module):
    """64x64x3 -> (mu, logvar) each [B, latent_ch, 8, 8]. Deliberately small (asymmetric AE)."""
    def __init__(self, latent_ch=4, ch=(64, 128, 256)):
        super().__init__()
        layers, c = [nn.Conv2d(3, ch[0], 3, padding=1)], ch[0]
        for co in ch:                                   # 64 -> 32 -> 16 -> 8
            layers += [ResBlock(c, co), Downsample(co)]
            c = co
        layers += [ResBlock(c, c), Norm(c), nn.SiLU(), nn.Conv2d(c, 2 * latent_ch, 3, padding=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        mu, lv = self.net(x).chunk(2, 1)
        return mu, lv.clamp(-12, 8)


def build(a):
    enc = Encoder(a.latent_ch)
    if a.decoder == "strong":
        dec = StrongDecoder(latent_ch=a.latent_ch)                       # ~13 M
    else:
        dec = StrongDecoder(latent_ch=a.latent_ch, ch=(128, 64, 64, 32), n_res=1, attn_8x8=False)
    D = PatchDiscriminator(critic=a.critic)
    return enc, dec, D


def n_params(m):
    return sum(p.numel() for p in m.parameters()) / 1e6


# ============================================================== metrics
def to_pm1(x):          # [0,1] -> [-1,1]
    return x * 2 - 1


def to_01(x):
    return ((x + 1) / 2).clamp(0, 1)


def hf_fraction(x, cut=8):
    """Training-free complexity: share of (non-DC) spectral energy above `cut` cycles/image."""
    g = x.mean(1)
    G = torch.fft.fft2(g - g.mean((1, 2), keepdim=True)).abs() ** 2
    H, W = g.shape[1:]
    fy = torch.fft.fftfreq(H, device=x.device) * H
    fx = torch.fft.fftfreq(W, device=x.device) * W
    R = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    return ((G * (R > cut)).sum((1, 2)) / (G.sum((1, 2)) + 1e-12))


def ssim(x, y, C1=0.01 ** 2, C2=0.03 ** 2):
    """Per-image SSIM, 11x11 Gaussian window (sigma 1.5), inputs in [0,1]."""
    k = torch.exp(-((torch.arange(11, device=x.device) - 5.0) ** 2) / (2 * 1.5 ** 2))
    w = (k[:, None] * k[None, :]); w = (w / w.sum()).expand(x.shape[1], 1, 11, 11)
    f = lambda t: F.conv2d(t, w, groups=x.shape[1])
    mx, my = f(x), f(y)
    sxx, syy, sxy = f(x * x) - mx ** 2, f(y * y) - my ** 2, f(x * y) - mx * my
    s = ((2 * mx * my + C1) * (2 * sxy + C2)) / ((mx ** 2 + my ** 2 + C1) * (sxx + syy + C2))
    return s.flatten(1).mean(1)


# ============================================================== train
def train(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    tr, ev = get_datasets(a.dataset, a.root, 64, n_eval=1000)
    if a.n_train and len(tr) > a.n_train:
        # fixed subset (generator seed 0, independent of --seed): every run on this dataset sees the same images
        from torch.utils.data import Subset
        idx = torch.randperm(len(tr), generator=torch.Generator().manual_seed(0))[:a.n_train]
        tr = Subset(tr, sorted(idx.tolist()))
    tl = DataLoader(tr, a.bs, shuffle=True, num_workers=a.workers, drop_last=True,
                    pin_memory=(dev == "cuda"), persistent_workers=a.workers > 0)
    el = DataLoader(ev, 250, shuffle=False, num_workers=a.workers)
    enc, dec, D = [m.to(dev) for m in build(a)]
    steps_per_epoch = min(len(tl), a.max_steps or len(tl))
    gan_start = a.gan_start_epoch * steps_per_epoch
    lossf = AutoencoderGANLoss(critic=a.critic, disc_start=10 ** 12 if a.no_gan else gan_start,
                               w_lpips=a.w_lpips, w_gan=a.w_gan, beta_kl=a.beta,
                               use_lpips=True, n_pixels=3 * 64 * 64).to(dev)
    with torch.no_grad():
        lossf.log_sigma.fill_(math.log(0.1))
    opt_ae = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()) + [lossf.log_sigma],
                              a.lr, betas=(0.5, 0.9))
    opt_d = torch.optim.Adam(D.parameters(), a.lr, betas=(0.5, 0.9))
    out = os.path.join("runs_p2", f"{a.dataset}_{a.tag}")
    os.makedirs(out, exist_ok=True)
    json.dump(vars(a), open(os.path.join(out, "args.json"), "w"), indent=1)
    print(f"train {len(tr)}  eval {len(ev)}  enc {n_params(enc):.2f} M  dec {n_params(dec):.2f} M  "
          f"D {n_params(D):.2f} M  device {dev}  GAN {'off' if a.no_gan else f'from epoch {a.gan_start_epoch}'}")

    step, best, log, start_ep = 0, -1e9, [], 0
    last_path = os.path.join(out, "last.pt")
    if a.resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location=dev)
        if "opt_ae" not in ck:
            raise SystemExit("last.pt is from an older script version without optimizer state; "
                             "start this run fresh (drop --resume or use a new --tag)")
        same = {k: v for k, v in vars(a).items() if k not in ("resume", "workers")}
        old = {k: v for k, v in ck["args"].items() if k not in ("resume", "workers")}
        diff = {k: (old.get(k), v) for k, v in same.items() if old.get(k) != v}
        if diff:
            raise SystemExit(f"--resume with different settings than the original run: {diff}")
        enc.load_state_dict(ck["enc"]); dec.load_state_dict(ck["dec"]); D.load_state_dict(ck["D"])
        opt_ae.load_state_dict(ck["opt_ae"]); opt_d.load_state_dict(ck["opt_d"])
        with torch.no_grad():
            lossf.log_sigma.copy_(ck["log_sigma"].to(dev)); lossf.sob_lambda.copy_(ck["sob_lambda"].to(dev))
        step, best, log, start_ep = ck["step"], ck["best"], ck["log"], ck["epoch"]
        print(f"resumed from epoch {start_ep} (step {step})", flush=True)
    elif a.resume:
        print("no last.pt found - starting from epoch 1", flush=True)

    def atomic_save(obj, path):                         # a crash mid-save cannot corrupt the file
        torch.save(obj, path + ".tmp"); os.replace(path + ".tmp", path)

    for ep in range(start_ep, a.epochs):
        enc.train(); dec.train(); D.train()
        t0, acc, n = time.time(), {}, 0
        for i, (x, *_) in enumerate(tl):
            if a.max_steps and i >= a.max_steps:
                break
            x = to_pm1(x.to(dev, non_blocking=True))
            mu, lv = enc(x)
            z = mu + torch.randn_like(mu) * (0.5 * lv).exp()
            xh = dec(z)
            kl = 0.5 * (mu ** 2 + lv.exp() - 1 - lv).flatten(1).sum(1).mean()
            loss, logs = lossf.generator_loss(x, xh, kl, D, dec.last_layer, step)
            opt_ae.zero_grad(set_to_none=True); loss.backward(); opt_ae.step()
            if step >= lossf.disc_start:
                for _ in range(a.n_critic):
                    d_loss, dl = lossf.critic_loss(x, xh, D)
                    opt_d.zero_grad(set_to_none=True); d_loss.backward(); opt_d.step()
                logs.update(dl)
            for k, v in logs.items():
                acc[k] = acc.get(k, 0.0) + v
            n += 1; step += 1
        val = quick_val(enc, dec, el, dev, lossf.lpips)
        rec = {"epoch": ep + 1, **{k: round(v / n, 5) for k, v in acc.items()}, **val,
               "sec": round(time.time() - t0, 1)}
        log.append(rec); print(json.dumps(rec), flush=True)
        state = {"enc": enc.state_dict(), "dec": dec.state_dict(), "D": D.state_dict(),
                 "log_sigma": lossf.log_sigma.detach().cpu(), "args": vars(a), "epoch": ep + 1}
        # select on LPIPS (perceptual), only among epochs where the final objective is active
        active = a.no_gan or step >= lossf.disc_start
        if active and -val["val_lpips"] > best:
            best = -val["val_lpips"]
            atomic_save(state, os.path.join(out, "best.pt"))
        atomic_save({**state, "opt_ae": opt_ae.state_dict(), "opt_d": opt_d.state_dict(),
                     "sob_lambda": lossf.sob_lambda.detach().cpu(), "step": step, "best": best,
                     "log": log}, last_path)
        json.dump(log, open(os.path.join(out, "log.json"), "w"), indent=1)
    if best == -1e9:                                     # GAN never started (short debug run)
        atomic_save(state, os.path.join(out, "best.pt"))


@torch.no_grad()
def quick_val(enc, dec, loader, dev, lp):
    enc.eval(); dec.eval()
    mse, lps = [], []
    for x, *_ in loader:
        x = x.to(dev)
        xh = to_01(dec(enc(to_pm1(x))[0]))
        mse.append(((xh - x) ** 2).flatten(1).mean(1))
        if lp is not None:
            lps.append(lp(to_pm1(xh), to_pm1(x)).flatten())
    m = torch.cat(mse)
    out = {"val_psnr": round(10 * math.log10(1 / m.mean().item()), 3)}
    out["val_lpips"] = round(torch.cat(lps).mean().item(), 4) if lps else float("nan")
    return out


# ============================================================== eval
def make_fid(dev):
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
        return lambda: FrechetInceptionDistance(feature=2048, normalize=True).to(dev)
    except Exception as e:
        print(f"FID disabled ({e}); install with: pip install torchmetrics torch-fidelity")
        return None


@torch.no_grad()
def evaluate(a):
    import lpips
    from torchvision.utils import save_image
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = os.path.join("runs_p2", f"{a.dataset}_{a.tag}")
    ck = torch.load(os.path.join(out, a.which + ".pt"), map_location=dev)
    targs = argparse.Namespace(**ck["args"])
    enc, dec, _ = build(targs)
    enc.load_state_dict(ck["enc"]); dec.load_state_dict(ck["dec"])
    enc, dec = enc.to(dev).eval(), dec.to(dev).eval()
    lp = lpips.LPIPS(net="vgg", verbose=False).to(dev).eval()
    _, ev = get_datasets(a.dataset, a.root, 64, n_eval=a.n_eval)
    el = DataLoader(ev, 200, shuffle=False, num_workers=a.workers)

    X, XH, cx, mse, ss, lps = [], [], [], [], [], []
    for x, *_ in el:
        x = x.to(dev)
        xh = to_01(dec(enc(to_pm1(x))[0]))
        cx.append(hf_fraction(x).cpu())
        mse.append(((xh - x) ** 2).flatten(1).mean(1).cpu())
        ss.append(ssim(x, xh).cpu())
        lps.append(lp(to_pm1(xh), to_pm1(x)).flatten().cpu())
        X.append(x.cpu()); XH.append(xh.cpu())
    X, XH = torch.cat(X), torch.cat(XH)
    cx, mse, ss, lps = [torch.cat(t).numpy() for t in (cx, mse, ss, lps)]
    q = np.digitize(cx, np.quantile(cx, [.2, .4, .6, .8]))

    fid_new = make_fid(dev)
    def fid(idx):
        if fid_new is None:
            return float("nan")
        m = fid_new()
        for s in range(0, len(idx), 200):
            b = idx[s:s + 200]
            m.update(X[b].to(dev), real=True); m.update(XH[b].to(dev), real=False)
        return float(m.compute())

    def row(idx):
        return {"n": int(len(idx)), "psnr_agg": 10 * math.log10(1 / mse[idx].mean()),
                "psnr_mean": float(np.mean(10 * np.log10(1 / mse[idx]))),
                "ssim": float(ss[idx].mean()), "lpips": float(lps[idx].mean()), "fid": fid(idx)}

    res = {"model": a.tag, "dataset": a.dataset, "epoch": ck["epoch"],
           "enc_M": n_params(enc), "dec_M": n_params(dec), "all": row(np.arange(len(mse)))}
    for j in range(5):
        res[f"Q{j + 1}"] = row(np.where(q == j)[0])
    res["quintile_gap_psnr"] = res["Q1"]["psnr_mean"] - res["Q5"]["psnr_mean"]
    res["quintile_gap_lpips"] = res["Q5"]["lpips"] - res["Q1"]["lpips"]

    print(f"\n{a.tag} on {a.dataset} (epoch {ck['epoch']}, n={len(mse)}, "
          f"enc {res['enc_M']:.2f} M, dec {res['dec_M']:.2f} M)")
    print(f"  {'':5s} {'PSNRagg':>8s} {'PSNRmean':>9s} {'SSIM':>7s} {'LPIPS':>7s} {'FID':>7s}")
    for k in ["all"] + [f"Q{j + 1}" for j in range(5)]:
        r = res[k]
        print(f"  {k:5s} {r['psnr_agg']:8.2f} {r['psnr_mean']:9.2f} {r['ssim']:7.4f} "
              f"{r['lpips']:7.4f} {r['fid']:7.2f}")
    print("  (Q1 = least high-frequency energy, Q5 = most; per-quintile FID uses n/5 images "
          "and is biased upward - compare across models, not with published FIDs)")
    json.dump(res, open(os.path.join(out, f"eval_{a.which}.json"), "w"), indent=1)

    rows = []
    for j in range(5):
        idx = np.where(q == j)[0][:8]
        rows += [X[idx], XH[idx]]
    save_image(torch.cat(rows), os.path.join(out, f"recon_grid_{a.which}.png"), nrow=8, padding=1)
    print("saved", os.path.join(out, f"eval_{a.which}.json"), "and recon_grid_" + a.which + ".png "
          "(row pairs: original / reconstruction, Q1 at top)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "eval"])
    ap.add_argument("--dataset", required=True, choices=["cifar10", "celeba", "chestxray14"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--tag", required=True, help="run name, e.g. D_strong_gan")
    ap.add_argument("--decoder", default="strong", choices=["strong", "small"])
    ap.add_argument("--latent_ch", type=int, default=4)
    ap.add_argument("--critic", default="hinge", choices=["hinge", "wgan_gp", "sobolev"])
    ap.add_argument("--no_gan", action="store_true")
    ap.add_argument("--w_lpips", type=float, default=1.0)
    ap.add_argument("--w_gan", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--gan_start_epoch", type=int, default=20)
    ap.add_argument("--n_critic", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max_steps", type=int, default=0, help="debug: cap steps per epoch")
    ap.add_argument("--resume", action="store_true", help="continue from runs_p2/<dataset>_<tag>/last.pt")
    ap.add_argument("--which", default="best", choices=["best", "last"])
    ap.add_argument("--n_eval", type=int, default=10000)
    ap.add_argument("--n_train", type=int, default=None,
                    help="use a fixed random subset of this many training images (e.g. 50000 = CIFAR-10 size)")
    a = ap.parse_args()
    train(a) if a.cmd == "train" else evaluate(a)
