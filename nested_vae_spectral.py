"""
nested_vae_spectral.py -- train a nested-dropout VAE and run the Idea A spectral test,
on CIFAR-10, CelebA or ChestX-ray14 (all local, no downloads).

Why nested dropout: the spectral test needs ORDERED latents, so that "decode with the first d
dims" is a valid model at every d. During training each image gets a random cut d, dims beyond d
are zeroed, and KL is charged only on the active dims. One model then gives every image's full
rate-distortion curve MSE_i(d) in a single eval pass (Rippel et al. 2014, nested dropout).

Keeps the Paper 1 rebuild rules: learnable decoder sigma, hard masking, 80 epochs.

Usage (lab PC, Windows paths fine):
  # one-time resize for the big folder datasets (minutes, then training is fast):
  python nested_vae_spectral.py prep --dataset celeba --root D:/data/celeba/img_align_celeba --out D:/data/celeba64 --workers 8
  python nested_vae_spectral.py prep --dataset chestxray14 --root D:/data/chestxray14/images --out D:/data/chest64 --workers 8
  # then pass the resized folder as --root
  python nested_vae_spectral.py train --dataset cifar10 --root D:/data/cifar10
  python nested_vae_spectral.py train --dataset celeba  --root D:/data/celeba64
  python nested_vae_spectral.py train --dataset chestxray14 --root D:/data/chest64
  python nested_vae_spectral.py eval  --dataset cifar10 --root D:/data/cifar10 --ckpt runs/cifar10_64/best.pt

Folder datasets (CelebA, ChestX-ray14) are read recursively (*.jpg, *.jpeg, *.png); the split is a
deterministic filename hash (10% eval), so it is identical on every run and machine.
"""
import argparse, glob, hashlib, json, math, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from PIL import Image

import spectral_rd_vae_test as srd

D_GRID = [8, 16, 32, 48, 64, 96, 128, 192, 256]   # standard sizes; d_max = 256


# ============================================================== data
_is_prepped = False


def _tf(dataset, res):
    import torchvision.transforms as T
    if dataset in ("celeba", "chestxray14") and _is_prepped:
        return T.Compose([T.Resize(res, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
                          T.ToTensor()])
    if dataset == "celeba":
        return T.Compose([T.CenterCrop(178), T.Resize(res, interpolation=T.InterpolationMode.BICUBIC,
                                                      antialias=True), T.ToTensor()])
    if dataset == "chestxray14":
        return T.Compose([T.Grayscale(3), T.Resize(res, interpolation=T.InterpolationMode.BICUBIC,
                                                   antialias=True), T.ToTensor()])
    return T.Compose([T.Resize(res, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
                      T.ToTensor()])


class FolderImages(Dataset):
    def __init__(self, files, tf):
        self.files, self.tf = files, tf

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        return self.tf(Image.open(self.files[i]).convert("RGB")), 0


def _is_eval(path):
    return int(hashlib.md5(os.path.splitext(os.path.basename(path))[0].encode()).hexdigest(), 16) % 10 == 0


def get_datasets(dataset, root, res=64, n_eval=None):
    global _is_prepped
    first = next(iter(glob.glob(os.path.join(root, "**", "*.png"), recursive=True)), None)
    _is_prepped = first is not None and Image.open(first).size == (res, res)   # already cropped+resized
    tf = _tf(dataset, res)
    if dataset == "cifar10":
        import torchvision
        tr = torchvision.datasets.CIFAR10(root, train=True, download=False, transform=tf)
        ev = torchvision.datasets.CIFAR10(root, train=False, download=False, transform=tf)
    else:
        files = sorted(f for ext in ("jpg", "jpeg", "png")
                       for f in glob.glob(os.path.join(root, "**", f"*.{ext}"), recursive=True))
        if not files:
            raise SystemExit(f"no images found under {root}")
        tr = FolderImages([f for f in files if not _is_eval(f)], tf)
        ev = FolderImages([f for f in files if _is_eval(f)], tf)
    if n_eval and len(ev) > n_eval:
        ev = Subset(ev, range(n_eval))
    return tr, ev


# ============================================================== model
class NestedVAE(nn.Module):
    """64x64 conv VAE, flat latent of d_max (=256) dims ordered by nested dropout."""
    def __init__(self, d_max=256, ch=(64, 128, 256, 256)):
        super().__init__()
        self.d_max = d_max
        enc, c = [], 3
        for co in ch:                                     # 64 -> 32 -> 16 -> 8 -> 4
            enc += [nn.Conv2d(c, co, 4, 2, 1), nn.GroupNorm(8, co), nn.SiLU()]
            c = co
        self.enc = nn.Sequential(*enc)
        self.fc_mu = nn.Linear(c * 16, d_max)
        self.fc_lv = nn.Linear(c * 16, d_max)
        self.fc_dec = nn.Linear(d_max, c * 16)
        dec = []
        for co in list(ch[::-1][1:]) + [32]:
            dec += [nn.ConvTranspose2d(c, co, 4, 2, 1), nn.GroupNorm(8, co), nn.SiLU()]
            c = co
        self.dec = nn.Sequential(*dec, nn.Conv2d(c, 3, 3, padding=1))
        self.c_last = ch[-1]
        self.log_sigma = nn.Parameter(torch.tensor(-2.0))   # learnable decoder sigma

    def encode(self, x):
        h = self.enc(x).flatten(1)
        return self.fc_mu(h), self.fc_lv(h).clamp(-12, 8)

    def decode(self, z):
        h = self.fc_dec(z).view(-1, self.c_last, 4, 4)
        return torch.sigmoid(self.dec(h))


def sample_cuts(B, d_max, device, d_min=4):
    """Log-uniform cut per image, so small d is trained as often as large d."""
    u = torch.rand(B, device=device)
    return torch.exp(math.log(d_min) + u * (math.log(d_max) - math.log(d_min))).round().long()


def nested_loss(model, x, beta=1.0):
    mu, lv = model.encode(x)
    z = mu + torch.randn_like(mu) * (0.5 * lv).exp()
    cut = sample_cuts(x.size(0), model.d_max, x.device)
    mask = (torch.arange(model.d_max, device=x.device)[None, :] < cut[:, None]).float()
    xh = model.decode(z * mask)
    s = model.log_sigma
    nll = (0.5 * (x - xh) ** 2 / torch.exp(2 * s) + s + 0.5 * math.log(2 * math.pi)).flatten(1).sum(1)
    kl = (0.5 * (mu ** 2 + lv.exp() - 1 - lv) * mask).sum(1)        # KL only on active dims
    return (nll + beta * kl).mean(), nll.mean().item(), kl.mean().item()


@torch.no_grad()
def eval_curve(model, loader, device, grid=None, max_batches=20):
    grid = grid or (16, 64, 128, model.d_max)
    out = {d: [] for d in grid}
    for b, (x, *_) in enumerate(loader):
        if b >= max_batches:
            break
        x = x.to(device)
        mu, _ = model.encode(x)
        for d in grid:
            m = torch.zeros_like(mu); m[:, :d] = 1
            out[d].append(((model.decode(mu * m) - x) ** 2).flatten(1).mean(1))
    return {d: 10 * math.log10(1 / torch.cat(v).mean().item()) for d, v in out.items()}


def make_recon_fn(model):
    def recon(x, d):
        mu, _ = model.encode(x)
        m = torch.zeros_like(mu); m[:, :d] = 1
        return model.decode(mu * m)
    return recon


# ============================================================== commands
def _prep_one(args):
    src, dst, dataset, res = args
    if os.path.exists(dst):
        return
    img = Image.open(src).convert("RGB")
    if dataset == "celeba":                               # same centre crop as training
        w, h = img.size
        l, t = (w - 178) // 2, (h - 178) // 2
        img = img.crop((l, t, l + 178, t + 178))
    elif dataset == "chestxray14":
        img = img.convert("L").convert("RGB")
    img.resize((res, res), Image.BICUBIC).save(dst)


def prep(a):
    """One-time resize of a folder dataset to res x res PNGs (CelebA 202k, ChestX-ray14 112k at
    1024^2 would otherwise be decoded at full size every epoch). Point --root at --out afterwards."""
    from multiprocessing import Pool
    files = sorted(f for ext in ("jpg", "jpeg", "png")
                   for f in glob.glob(os.path.join(a.root, "**", f"*.{ext}"), recursive=True))
    os.makedirs(a.out, exist_ok=True)
    jobs = [(f, os.path.join(a.out, os.path.splitext(os.path.basename(f))[0] + ".png"), a.dataset, a.res)
            for f in files]
    with Pool(a.workers or 1) as p:
        for i, _ in enumerate(p.imap_unordered(_prep_one, jobs, chunksize=256)):
            if i % 20000 == 0:
                print(f"{i}/{len(jobs)}")
    print(f"done: {len(jobs)} images -> {a.out}")


def train(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    tr, ev = get_datasets(a.dataset, a.root, a.res, n_eval=2000)
    tl = DataLoader(tr, a.bs, shuffle=True, num_workers=a.workers, drop_last=True, pin_memory=True)
    el = DataLoader(ev, 256, shuffle=False, num_workers=a.workers)
    model = NestedVAE(d_max=a.d_max).to(dev)
    opt = torch.optim.Adam(model.parameters(), a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    out = os.path.join("runs", f"{a.dataset}_{a.res}")
    os.makedirs(out, exist_ok=True)
    print(f"train {len(tr)}  eval {len(ev)}  params {sum(p.numel() for p in model.parameters())/1e6:.2f} M  device {dev}")
    best, log = -1e9, []
    for ep in range(a.epochs):
        model.train(); t0 = time.time(); acc = [0, 0, 0]; n = 0
        for i, (x, *_) in enumerate(tl):
            if a.max_steps and i >= a.max_steps:
                break
            loss, nll, kl = nested_loss(model, x.to(dev, non_blocking=True), a.beta)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            acc[0] += loss.item(); acc[1] += nll; acc[2] += kl; n += 1
        sched.step()
        model.eval()
        ps = eval_curve(model, el, dev)
        rec = dict(epoch=ep + 1, loss=acc[0] / n, nll=acc[1] / n, kl=acc[2] / n,
                   sigma=model.log_sigma.exp().item(), psnr=ps, sec=round(time.time() - t0, 1))
        log.append(rec)
        print(json.dumps(rec))
        score = float(np.mean(list(ps.values())))        # select on mean PSNR over d (no single-d bias)
        if score > best:
            best = score
            torch.save({"model": model.state_dict(), "args": vars(a), "epoch": ep + 1}, os.path.join(out, "best.pt"))
        torch.save({"model": model.state_dict(), "args": vars(a), "epoch": ep + 1}, os.path.join(out, "last.pt"))
        json.dump(log, open(os.path.join(out, "log.json"), "w"), indent=1)


def evaluate(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(a.ckpt, map_location=dev)
    model = NestedVAE(d_max=ck["args"]["d_max"]).to(dev).eval()
    model.load_state_dict(ck["model"])
    _, ev = get_datasets(a.dataset, a.root, a.res, n_eval=a.n_eval)
    el = DataLoader(ev, 256, shuffle=False, num_workers=a.workers)
    grid = [d for d in D_GRID if d <= model.d_max]
    # ordering check: the test is only meaningful if quality rises with d
    ps = eval_curve(model, el, dev, grid=tuple(grid), max_batches=8)
    print("ordering check (PSNR vs d):", {d: round(v, 2) for d, v in ps.items()})
    if any(ps[grid[i + 1]] < ps[grid[i]] - 0.05 for i in range(len(grid) - 1)):
        print("WARNING: PSNR is not monotone in d -- latents are not properly ordered; results unreliable")
    alpha, logvar, band, mse = srd.collect(make_recon_fn(model), el, grid, dev)
    os.makedirs("results", exist_ok=True)
    path = f"results/spectral_rd_{a.dataset}_{a.res}.npz"
    np.savez(path, alpha=alpha, logvar=logvar, band=band, mse=mse, d_grid=np.array(grid))
    print("saved", path)
    srd.analyse(alpha, logvar, band, mse, grid, a.d_avg, a.fit_lo, a.fit_hi)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "train", "eval"])
    ap.add_argument("--dataset", required=True, choices=["cifar10", "celeba", "chestxray14"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--res", type=int, default=64)
    ap.add_argument("--d_max", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max_steps", type=int, default=0, help="debug: cap steps per epoch")
    ap.add_argument("--ckpt")
    ap.add_argument("--out", help="prep: output folder for resized images")
    ap.add_argument("--n_eval", type=int, default=10000)
    ap.add_argument("--d_avg", type=float, default=64, help="mean budget; d_max/d_avg = 4x headroom")
    ap.add_argument("--fit_lo", type=int, default=16)
    ap.add_argument("--fit_hi", type=int, default=256)
    a = ap.parse_args()
    if a.res != 64:
        raise SystemExit("NestedVAE is built for 64x64; change ch/fc sizes before using another resolution")
    {"prep": prep, "train": train, "eval": evaluate}[a.cmd](a)
