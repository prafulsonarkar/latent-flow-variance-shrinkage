"""
p3_latent_gen.py -- Paper 3 pilot: does a better Paper 2 tokenizer give better GENERATION, and on which images?

For one Paper 2 VAE (runs_p2/<ds>_<tag>/last.pt, frozen) it
  encode : encodes the VAE's own training images (and the 10k test images) to 4x8x8 latents        (~1 min)
  train  : trains a small DiT (flow matching, unconditional) on those latents                        (hours)
  sample : generates n images, decodes them with the same VAE, and reports
             gFID against the same 10,000 test images used for rFID in Paper 2,
             the share of generated images falling in each complexity quintile of h(x) (20% each = matched),
             FID inside each quintile (generated Qk vs real Qk)
  quint  : (runs automatically at the end of sample) FID at an EQUAL number of images per quintile and KID
           (unbiased in n), so quintiles and runs are comparable; strata_grid.png = real vs generated per quintile
  summary: one table over all runs of a dataset (gFID, rFID, iFID, KID, tokenizer gap, generator gap, Q5 share)
  recon  : control: complexity-quintile shares of the VAE's own reconstructions of the test images  (~2 min)
  ifid   : interpolated FID (Xu et al. 2026, arXiv 2603.05630): decode the midpoint between each test latent
           and its nearest training latent; FID against the test images. No diffusion needed.        (~5 min)
The DiT, its training budget and the sampler are identical for every VAE, so differences come from the latent space.

Protocol notes (state them in the paper):
  * gFID here is against the 10k test images (to match the Paper 2 rFID), not the usual 50k-train statistics;
    compare runs with each other, not with published numbers.
  * CelebA uses the same --n_train subset (generator seed 0) the VAE was trained on, read from the VAE checkpoint.

Run from D:\\new_vae_research (needs paper2_train.py etc.), one VAE at a time; GPU recommended for train/sample:
  python p3_latent_gen.py encode --dataset cifar10 --root D:\\new_vae_research\\data --tag D_strong_gan_v2
  python p3_latent_gen.py ifid   --dataset cifar10 --root D:\\new_vae_research\\data --tag D_strong_gan_v2
  python p3_latent_gen.py train  --dataset cifar10 --root D:\\new_vae_research\\data --tag D_strong_gan_v2
  python p3_latent_gen.py sample --dataset cifar10 --root D:\\new_vae_research\\data --tag D_strong_gan_v2
  (train saves every --save_every steps; if it stops, rerun the same train command and it resumes)
Second DiT seed (same latents, new folder runs_p3/<ds>_<tag>_seed1/):  add --seed 1 to train and sample.
Smoke test first (a few minutes, checks the whole chain, numbers meaningless):  add --smoke to every command.
Outputs: runs_p3/<ds>_<tag>/  latents.npz  dit.pt  log.json  gen_eval.json  gen_images.npz  quint_eval.json
         samples.png  strata_grid.png  ifid.json  recon_strata.json      (DiT seed k > 0: dit/gen/quint files in <ds>_<tag>_seed<k>/)
"""
import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from paper2_train import build, to_pm1, to_01, hf_fraction, make_fid
from nested_vae_spectral import get_datasets


# =========================================================================================== helpers
def out_dir(a, seeded=False):
    """runs_p3/<ds>_<tag>/ holds the latents (shared by all DiT seeds); DiT outputs for --seed k > 0 go to <ds>_<tag>_seed<k>/."""
    d = os.path.join(a.out_root, f"{a.dataset}_{a.tag}" + ("_smoke" if a.smoke else "")
                     + (f"_seed{a.seed}" if seeded and a.seed else ""))
    os.makedirs(d, exist_ok=True)
    return d


def load_vae(a, dev):
    ck = torch.load(os.path.join(a.runs, f"{a.dataset}_{a.tag}", f"{a.which}.pt"), map_location="cpu",
                    weights_only=False)
    vargs = argparse.Namespace(**ck["args"])
    enc, dec, _ = build(vargs)
    enc.load_state_dict(ck["enc"]); dec.load_state_dict(ck["dec"])
    for m in (enc, dec):
        m.to(dev).eval().requires_grad_(False)
    return enc, dec, vargs


def datasets(a, vargs):
    tr, ev = get_datasets(a.dataset, a.root, 64, n_eval=a.n_eval)
    n_train = getattr(vargs, "n_train", 0) or 0
    if n_train and len(tr) > n_train:                 # the exact subset paper2_train.py used for this VAE
        from torch.utils.data import Subset
        idx = torch.randperm(len(tr), generator=torch.Generator().manual_seed(0))[:n_train]
        tr = Subset(tr, sorted(idx.tolist()))
    if a.smoke:
        from torch.utils.data import Subset
        tr, ev = Subset(tr, range(min(len(tr), 2000))), Subset(ev, range(min(len(ev), 500)))
    return tr, ev


def atomic_save(obj, path):
    torch.save(obj, path + ".tmp"); os.replace(path + ".tmp", path)


def fid_between(real, fake, dev):
    """torchmetrics FID (2048 features, inputs in [0,1]); nan if unavailable or too few images."""
    new = make_fid(dev)
    if new is None or min(len(real), len(fake)) < 50:
        return float("nan")
    m = new()
    for s in range(0, len(real), 200):
        m.update(real[s:s + 200].to(dev), real=True)
    for s in range(0, len(fake), 200):
        m.update(fake[s:s + 200].to(dev), real=False)
    return float(m.compute())


# =========================================================================================== 1. encode
@torch.no_grad()
def encode(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    enc, _, vargs = load_vae(a, dev)
    tr, ev = datasets(a, vargs)
    res = {}
    for name, ds in (("train", tr), ("test", ev)):
        MU, LV = [], []
        for x, *_ in DataLoader(ds, 250, shuffle=False, num_workers=a.workers):
            mu, lv = enc(to_pm1(x.to(dev)))
            MU.append(mu.half().cpu()); LV.append(lv.half().cpu())
        res[f"{name}_mu"], res[f"{name}_lv"] = torch.cat(MU).numpy(), torch.cat(LV).numpy()
        print(f"{name}: {len(res[f'{name}_mu'])} latents of shape {res[f'{name}_mu'].shape[1:]}", flush=True)
    mu = res["train_mu"].astype(np.float32)
    sd = np.exp(0.5 * res["train_lv"].astype(np.float32))
    zs = mu + sd * np.random.default_rng(0).standard_normal(mu.shape).astype(np.float32)   # z ~ q(z|x)
    res["mu_mean"], res["mu_std"] = mu.mean((0, 2, 3)), mu.std((0, 2, 3))
    res["s_mean"], res["s_std"] = zs.mean((0, 2, 3)), zs.std((0, 2, 3))
    np.savez(os.path.join(out_dir(a), "latents.npz"), **res)
    ratio = sd.mean((0, 2, 3)) / res["mu_std"]
    print("per channel: std of mu", np.round(res["mu_std"], 3), " mean posterior sigma / std of mu", np.round(ratio, 3))
    if ratio.max() > 0.5:
        print("  NOTE: posterior noise is large relative to the signal; sampled latents (--latent sample) are noisy. "
              "Report this, and consider --latent mean as a second setting.")


def load_latents(a, latent=None):
    """Latents plus the per-channel statistics of the variable the DiT is trained on (z sample or mu)."""
    f = dict(np.load(os.path.join(out_dir(a), "latents.npz")))
    k = "s" if (latent or a.latent) == "sample" else "mu"
    f["ch_mean"], f["ch_std"] = f[f"{k}_mean"], f[f"{k}_std"]
    norm = lambda t: (torch.from_numpy(t.astype(np.float32)) - torch.from_numpy(f["ch_mean"])[None, :, None, None]) \
        / torch.from_numpy(f["ch_std"])[None, :, None, None]
    return f, norm


def denorm(zn, z, dev):
    return zn * torch.from_numpy(z["ch_std"]).to(dev)[None, :, None, None] + \
        torch.from_numpy(z["ch_mean"]).to(dev)[None, :, None, None]


# =========================================================================================== 2. DiT
def sincos_2d(dim, n):
    y, x = torch.meshgrid(torch.arange(n, dtype=torch.float32), torch.arange(n, dtype=torch.float32), indexing="ij")
    om = 1.0 / 10000 ** (torch.arange(dim // 4, dtype=torch.float32) / (dim // 4))
    out = [torch.sin(x.flatten()[:, None] * om), torch.cos(x.flatten()[:, None] * om),
           torch.sin(y.flatten()[:, None] * om), torch.cos(y.flatten()[:, None] * om)]
    return torch.cat(out, 1)[None]


def t_embed(t, dim=256):
    f = torch.exp(-math.log(10000) * torch.arange(dim // 2, device=t.device) / (dim // 2))
    a = 1000 * t[:, None] * f[None]
    return torch.cat([torch.cos(a), torch.sin(a)], 1)


class Block(nn.Module):
    def __init__(self, w, heads):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(w, elementwise_affine=False), nn.LayerNorm(w, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(w, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(w, 4 * w), nn.GELU(approximate="tanh"), nn.Linear(4 * w, w))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(w, 6 * w))
        nn.init.zeros_(self.ada[1].weight); nn.init.zeros_(self.ada[1].bias)      # adaLN-Zero

    def forward(self, x, c):
        s1, b1, g1, s2, b2, g2 = self.ada(c)[:, None].chunk(6, -1)
        h = self.n1(x) * (1 + s1) + b1
        x = x + g1 * self.attn(h, h, h, need_weights=False)[0]
        return x + g2 * self.mlp(self.n2(x) * (1 + s2) + b2)


class DiT(nn.Module):
    """Patch size 1 on the 4x8x8 latent -> 64 tokens. Default width 384, depth 12, 6 heads (DiT-S size)."""
    def __init__(self, ch=4, n=8, width=384, depth=12, heads=6):
        super().__init__()
        self.ch, self.n = ch, n
        self.inp = nn.Linear(ch, width)
        self.register_buffer("pos", sincos_2d(width, n), persistent=False)
        self.temb = nn.Sequential(nn.Linear(256, width), nn.SiLU(), nn.Linear(width, width))
        self.blocks = nn.ModuleList(Block(width, heads) for _ in range(depth))
        self.nf = nn.LayerNorm(width, elementwise_affine=False)
        self.ada_f = nn.Sequential(nn.SiLU(), nn.Linear(width, 2 * width))
        self.out = nn.Linear(width, ch)
        for m in (self.ada_f[1], self.out):
            nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x, t):
        B = x.shape[0]
        h = self.inp(x.flatten(2).transpose(1, 2)) + self.pos
        c = self.temb(t_embed(t))
        for b in self.blocks:
            h = b(h, c)
        s, sh = self.ada_f(c)[:, None].chunk(2, -1)
        h = self.out(self.nf(h) * (1 + s) + sh)
        return h.transpose(1, 2).reshape(B, self.ch, self.n, self.n)


def make_dit(a):
    return DiT(width=a.width, depth=a.depth, heads=a.heads)


# =========================================================================================== 3. train (flow matching)
def train(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    z, norm = load_latents(a)
    MU, LV = torch.from_numpy(z["train_mu"].astype(np.float32)).to(dev), torch.from_numpy(z["train_lv"].astype(np.float32)).to(dev)
    m_, s_ = torch.from_numpy(z["ch_mean"]).to(dev)[None, :, None, None], torch.from_numpy(z["ch_std"]).to(dev)[None, :, None, None]
    model, ema = make_dit(a).to(dev), make_dit(a).to(dev)
    ema.load_state_dict(model.state_dict()); ema.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=a.amp and dev == "cuda")
    path, step, log = os.path.join(out_dir(a, True), "dit.pt"), 0, []
    if os.path.exists(path):
        ck = torch.load(path, map_location=dev, weights_only=False)
        diff = {k: (ck["args"].get(k), v) for k, v in vars(a).items()
                if k in ("width", "depth", "heads", "bs", "lr", "ema", "latent", "seed", "amp") and ck["args"].get(k) != v}
        if diff:
            raise SystemExit(f"dit.pt exists with different settings {diff}; use the same flags or another --out_root")
        model.load_state_dict(ck["model"]); ema.load_state_dict(ck["ema"]); opt.load_state_dict(ck["opt"])
        step, log = ck["step"], ck["log"]
        print(f"resumed at step {step}", flush=True)
    steps = 300 if a.smoke else a.steps
    n_par = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"DiT {n_par:.1f} M params; {len(MU)} training latents; {steps} steps of batch {a.bs} on {dev}", flush=True)
    t0, acc = time.time(), 0.0
    ema_p, model_p = list(ema.parameters()), [p.detach() for p in model.parameters()]
    model.train()
    while step < steps:
        i = torch.randint(0, len(MU), (a.bs,), device=dev)
        x0 = MU[i] + (torch.randn_like(MU[i]) * (0.5 * LV[i]).exp() if a.latent == "sample" else 0)
        x0 = (x0 - m_) / s_
        eps, t = torch.randn_like(x0), torch.rand(a.bs, device=dev)
        xt = (1 - t)[:, None, None, None] * x0 + t[:, None, None, None] * eps
        with torch.autocast("cuda", dtype=torch.float16, enabled=a.amp and dev == "cuda"):
            loss = ((model(xt, t).float() - (eps - x0)) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update()
        with torch.no_grad():                         # EMA in one fused call: ema += (1 - d) * (model - ema)
            d = min(a.ema, (1 + step) / (10 + step))
            torch._foreach_lerp_(ema_p, model_p, 1 - d)
        acc += loss.item(); step += 1
        if step % a.log_every == 0 or step == steps:
            rec = {"step": step, "loss": round(acc / a.log_every, 5), "it_s": round(a.log_every / (time.time() - t0), 1)}
            log.append(rec); print(json.dumps(rec), flush=True); acc, t0 = 0.0, time.time()
        if step % a.save_every == 0 or step == steps:
            atomic_save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                         "step": step, "log": log, "args": vars(a)}, path)
    json.dump(log, open(os.path.join(out_dir(a, True), "log.json"), "w"), indent=1)
    print("saved", path)


# =========================================================================================== 4. sample + evaluate
@torch.no_grad()
def euler(model, n, dev, steps, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    x = torch.randn(n, 4, 8, 8, device=dev, generator=g)
    ts = torch.linspace(1, 0, steps + 1, device=dev)
    for k in range(steps):
        t = ts[k].expand(n)
        x = x + (ts[k + 1] - ts[k]) * model(x, t)          # dx/dt = v, integrated from t=1 (noise) to 0 (data)
    return x


def strata(real, fake, dev):
    """Quintile cuts from REAL h(x); share of generated images per quintile and FID inside each quintile."""
    hr, hf = hf_fraction(real).numpy(), hf_fraction(fake).numpy()
    cuts = np.quantile(hr, [.2, .4, .6, .8])
    qr, qf = np.digitize(hr, cuts), np.digitize(hf, cuts)
    out = {"h_real_mean": float(hr.mean()), "h_gen_mean": float(hf.mean())}
    for j in range(5):
        r, f = np.where(qr == j)[0], np.where(qf == j)[0]
        out[f"Q{j + 1}"] = {"share_gen": float(len(f) / len(hf)), "n_real": int(len(r)), "n_gen": int(len(f)),
                            "fid": fid_between(real[r], fake[f], dev) if min(len(r), len(f)) >= 500 else float("nan")}
    return out


def test_images(a, vargs):
    _, ev = datasets(a, vargs)
    return torch.cat([x for x, *_ in DataLoader(ev, 250, shuffle=False, num_workers=a.workers)])


@torch.no_grad()
def sample(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    _, dec, vargs = load_vae(a, dev)
    ck = torch.load(os.path.join(out_dir(a, True), "dit.pt"), map_location=dev, weights_only=False)
    z, _ = load_latents(a, ck["args"]["latent"])
    model = make_dit(argparse.Namespace(**ck["args"])).to(dev).eval()
    model.load_state_dict(ck["ema"])
    n = 500 if a.smoke else a.n_gen
    X = []
    t0 = time.time()
    for s in range(0, n, a.gen_bs):
        zn = euler(model, min(a.gen_bs, n - s), dev, a.sample_steps, 1000003 * a.seed + s)   # seed 0: same noise as before
        X.append(to_01(dec(denorm(zn, z, dev))).cpu())
    fake = torch.cat(X)
    print(f"generated {len(fake)} images in {time.time() - t0:.0f}s (DiT step {ck['step']}, {a.sample_steps} Euler steps)", flush=True)
    real = test_images(a, vargs)
    from torchvision.utils import save_image
    od = out_dir(a, True)
    save_image(fake[:64], os.path.join(od, "samples.png"), nrow=8, padding=1)
    np.savez_compressed(os.path.join(od, "gen_images.npz"), x=(fake * 255).round().clamp(0, 255).to(torch.uint8).numpy())
    res = {"tag": a.tag, "dataset": a.dataset, "seed": a.seed, "dit_step": ck["step"], "n_gen": len(fake), "n_real": len(real),
           "sample_steps": a.sample_steps, "gfid": fid_between(real, fake, dev), **strata(real, fake, dev)}
    rf = os.path.join(a.runs, f"{a.dataset}_{a.tag}", f"eval_{a.which}.json")
    if os.path.exists(rf):
        res["rfid_paper2"] = json.load(open(rf))["all"]["fid"]
    json.dump(res, open(os.path.join(od, "gen_eval.json"), "w"), indent=1)
    print(f"\n{a.tag} on {a.dataset} (DiT seed {a.seed}): gFID {res['gfid']:.2f}" + (f"   (rFID {res['rfid_paper2']:.2f})" if "rfid_paper2" in res else ""))
    print(f"  mean h: real {res['h_real_mean']:.3f}, generated {res['h_gen_mean']:.3f}")
    print(f"  {'':3s} {'share of generated':>19s} {'FID in quintile':>16s}   (share 20% each = complexity distribution matched)")
    for j in range(1, 6):
        q = res[f"Q{j}"]
        print(f"  Q{j} {100 * q['share_gen']:18.1f}% {q['fid']:16.2f}")
    print("  (per-quintile FID above uses unequal n; the equal-n FID and KID from `quint` are the ones to report)")
    print("saved", os.path.join(od, "gen_eval.json"), ", samples.png and gen_images.npz\n")
    quint(a, fake=fake, real=real)


# =========================================================================================== 4b. decoder control
@torch.no_grad()
def recon(a):
    """Control for the strata: where do the VAE's own RECONSTRUCTIONS of the test images fall?
    If reconstructions already lose high-frequency energy, part of the generator's shortfall is the decoder's."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    enc, dec, vargs = load_vae(a, dev)
    real = test_images(a, vargs)
    rec = torch.cat([to_01(dec(enc(to_pm1(real[s:s + 250].to(dev)))[0])).cpu() for s in range(0, len(real), 250)])
    hr, hc = hf_fraction(real).numpy(), hf_fraction(rec).numpy()
    cuts = np.quantile(hr, [.2, .4, .6, .8])
    qr, qc = np.digitize(hr, cuts), np.digitize(hc, cuts)
    res = {"tag": a.tag, "dataset": a.dataset, "n": len(real), "h_real_mean": float(hr.mean()),
           "h_recon_mean": float(hc.mean()), "frac_recon_lower_quintile": float((qc < qr).mean()),
           **{f"Q{j + 1}": {"share_recon": float((qc == j).mean())} for j in range(5)}}
    json.dump(res, open(os.path.join(out_dir(a), "recon_strata.json"), "w"), indent=1)
    g = os.path.join(out_dir(a), "gen_eval.json")
    G = json.load(open(g)) if os.path.exists(g) else None
    print(f"{a.tag} on {a.dataset}: mean h real {hr.mean():.3f}, reconstruction {hc.mean():.3f}"
          + (f", generated {G['h_gen_mean']:.3f}" if G else ""))
    print(f"  {'':3s} {'real':>6s} {'recon':>7s}" + (f" {'generated':>10s}" if G else ""))
    for j in range(5):
        print(f"  Q{j + 1} {20.0:5.1f}% {100 * res[f'Q{j + 1}']['share_recon']:6.1f}%"
              + (f" {100 * G[f'Q{j + 1}']['share_gen']:9.1f}%" if G else ""))
    print(f"  {100 * res['frac_recon_lower_quintile']:.1f}% of test images drop to a lower quintile after reconstruction")
    print("saved", os.path.join(out_dir(a), "recon_strata.json"))


# =========================================================================================== 4c. equal-n quintile metrics
@torch.no_grad()
def inception_feats(x, dev):
    """2048-d pool features of the same InceptionV3 torchmetrics FID uses (inputs in [0,1])."""
    from torchmetrics.image.fid import FrechetInceptionDistance
    net = FrechetInceptionDistance(feature=2048, normalize=True).to(dev).inception
    return torch.cat([net((x[s:s + 200].to(dev) * 255).round().clamp(0, 255).to(torch.uint8)).double().cpu()
                      for s in range(0, len(x), 200)]).numpy()


def fid_np(f1, f2):
    m1, m2, c1, c2 = f1.mean(0), f2.mean(0), np.cov(f1, rowvar=False), np.cov(f2, rowvar=False)
    ev = np.linalg.eigvals(c1 @ c2)                      # tr sqrt(C1 C2) = sum of sqrt of its eigenvalues
    return float(((m1 - m2) ** 2).sum() + np.trace(c1) + np.trace(c2) - 2 * np.sqrt(np.clip(ev.real, 0, None)).sum())


def kid_np(f1, f2):
    """Unbiased MMD^2 with the cubic polynomial kernel (Binkowski et al. 2018), x1000. Unbiased in n, so
    quintiles with different numbers of images can be compared directly."""
    d = f1.shape[1]
    k = lambda a, b: (a @ b.T / d + 1) ** 3
    kxx, kyy, kxy = k(f1, f1), k(f2, f2), k(f1, f2)
    m, n = len(f1), len(f2)
    return 1000 * float((kxx.sum() - np.trace(kxx)) / (m * (m - 1)) + (kyy.sum() - np.trace(kyy)) / (n * (n - 1))
                        - 2 * kxy.mean())


def label_rows(img, labels, row_h):
    """Put a text label at the left of each row of a grid image (uint8 HxWx3) -> PIL image."""
    from PIL import Image, ImageDraw, ImageFont
    pad, top = 150, 40
    im = Image.new("RGB", (img.shape[1] + pad, img.shape[0] + top), "white")
    im.paste(Image.fromarray(img), (pad, top))
    dr = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("arial.ttf", 22)
    except OSError:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 22)
        except OSError:
            font = ImageFont.load_default()
    half = (img.shape[1] - 32) // 2                       # 32 = the 16-px gap, upscaled 2x
    dr.text((pad + half // 2 - 30, 8), "real", fill="black", font=font)
    dr.text((pad + half + 32 + half // 2 - 60, 8), "generated", fill="black", font=font)
    for i, t in enumerate(labels):
        dr.text((8, top + i * row_h + row_h // 2 - 12), t, fill="black", font=font)
    return im


@torch.no_grad()
def quint(a, fake=None, real=None):
    """Complexity-stratified metrics that do not depend on how many generated images land in each quintile:
       FID at an equal n per quintile (--q_n images, --q_rep random subsets, mean and sd) and KID on all images.
       Also writes strata_grid.png: for each quintile, real test images vs generated images of that complexity."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    od = out_dir(a, True)
    if fake is None:
        f = os.path.join(od, "gen_images.npz")
        if not os.path.exists(f):
            raise SystemExit(f"{f} not found: run `sample` once (it saves the generated images and then runs quint)")
        fake = torch.from_numpy(np.load(f)["x"]).float() / 255
    if real is None:
        _, _, vargs = load_vae(a, "cpu")
        real = test_images(a, vargs)
    hr, hf = hf_fraction(real).numpy(), hf_fraction(fake).numpy()
    cuts = np.quantile(hr, [.2, .4, .6, .8])
    qr, qf = np.digitize(hr, cuts), np.digitize(hf, cuts)
    print(f"Inception features for {len(real)} real + {len(fake)} generated images ...", flush=True)
    Fr, Ff = inception_feats(real, dev), inception_feats(fake, dev)
    rng = np.random.default_rng(0)
    n_q = 20 if a.smoke else a.q_n
    res = {"tag": a.tag, "dataset": a.dataset, "seed": a.seed, "q_n": n_q, "q_rep": a.q_rep,
           "kid_all": float(np.mean([kid_np(Fr[rng.choice(len(Fr), 1000, replace=False)], Ff[rng.choice(len(Ff), 1000, replace=False)])
                                       for _ in range(50)])) if min(len(Fr), len(Ff)) >= 1000 else kid_np(Fr, Ff), "h_real_mean": float(hr.mean()), "h_gen_mean": float(hf.mean())}
    for j in range(5):
        r, g = np.where(qr == j)[0], np.where(qf == j)[0]
        share = len(g) / len(hf)
        q = {"share_gen": share, "share_ci95": [share - 1.96 * np.sqrt(share * (1 - share) / len(hf)),
                                                share + 1.96 * np.sqrt(share * (1 - share) / len(hf))],
             "n_real": int(len(r)), "n_gen": int(len(g)), "kid": kid_np(Fr[r], Ff[g]) if len(g) >= 50 else float("nan")}
        if len(g) >= n_q and len(r) >= n_q:
            v = [fid_np(Fr[rng.choice(r, n_q, replace=False)], Ff[rng.choice(g, n_q, replace=False)]) for _ in range(a.q_rep)]
            q["fid_eq"], q["fid_eq_sd"] = float(np.mean(v)), float(np.std(v))
        else:
            q["fid_eq"] = q["fid_eq_sd"] = float("nan")
        res[f"Q{j + 1}"] = q
    # reference: the same equal-n FID between two disjoint halves of the REAL quintile = floor caused by n alone
    floor = []
    for j in range(5):
        r = np.where(qr == j)[0]
        floor.append(float(np.mean([fid_np(*np.split(Fr[rng.choice(r, 2 * n_q, replace=False)], 2)) for _ in range(3)]))
                     if len(r) >= 2 * n_q else float("nan"))
    res["fid_eq_real_vs_real"] = floor
    json.dump(res, open(os.path.join(od, "quint_eval.json"), "w"), indent=1)

    print(f"\n{a.tag} on {a.dataset} (DiT seed {a.seed}): KID x1000 all images {res['kid_all']:.2f}")
    print(f"  {'':3s} {'share gen [95% CI]':>22s} {'KID x1000':>10s} {f'FID @ n={n_q}':>16s} {'real-vs-real':>13s}")
    for j in range(5):
        q = res[f"Q{j + 1}"]
        print(f"  Q{j + 1} {100 * q['share_gen']:6.1f}% [{100 * q['share_ci95'][0]:4.1f}, {100 * q['share_ci95'][1]:4.1f}]"
              f" {q['kid']:10.2f} {q['fid_eq']:9.2f} ± {q['fid_eq_sd']:4.2f} {floor[j]:13.2f}")
    print("  (real-vs-real = FID between two sets of real images at the same n: the part of FID caused by small n)")

    # image grid: rows Q1..Q5, left 8 real, right 8 generated of that quintile
    k = 8
    rows, labels = [], []
    pad = torch.ones(3, 64, 16)
    for j in range(5):
        r, g = np.where(qr == j)[0], np.where(qf == j)[0]
        R = [real[i] for i in rng.choice(r, min(k, len(r)), replace=False)]
        R += [torch.ones(3, 64, 64)] * (k - len(R))
        G = [fake[i] for i in rng.choice(g, min(k, len(g)), replace=False)] if len(g) else []
        G += [torch.ones(3, 64, 64)] * (k - len(G))
        rows.append(torch.cat([torch.cat(R, 2), pad, torch.cat(G, 2)], 2))
        labels.append(f"Q{j + 1}  {100 * len(g) / len(hf):.0f}%")
    grid = torch.cat([torch.cat([t, torch.ones(3, 6, t.shape[2])], 1) for t in rows], 1)
    img = (grid.clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    img = np.kron(img, np.ones((2, 2, 1), dtype=np.uint8))
    label_rows(img, labels, img.shape[0] // 5).save(os.path.join(od, "strata_grid.png"))
    print("saved", os.path.join(od, "quint_eval.json"), "and strata_grid.png "
          "(rows = complexity quintiles, simplest on top; left 8 real, right 8 generated; % = share of generated)")


# =========================================================================================== 4d. summary over runs
def summary(a):
    """One table over every run in --out_root for this dataset: tokenizer gap, generator gap, gFID, KID, Q5 share.
    Runs with several DiT seeds are shown per seed and as mean."""
    import glob
    rows = []
    for d in sorted(glob.glob(os.path.join(a.out_root, f"{a.dataset}_*"))):
        if d.endswith("_smoke") or not os.path.exists(os.path.join(d, "gen_eval.json")):
            continue
        base = d.rsplit("_seed", 1)[0] if "_seed" in os.path.basename(d) else d
        G = json.load(open(os.path.join(d, "gen_eval.json")))
        rc = os.path.join(base, "recon_strata.json"); R = json.load(open(rc)) if os.path.exists(rc) else None
        qe = os.path.join(d, "quint_eval.json"); Q = json.load(open(qe)) if os.path.exists(qe) else None
        fi = os.path.join(base, "ifid.json"); I = json.load(open(fi)) if os.path.exists(fi) else None
        nan = float("nan")
        rows.append(dict(run=os.path.basename(base)[len(a.dataset) + 1:], seed=G.get("seed", 0), gfid=G["gfid"],
                         rfid=G.get("rfid_paper2", nan), ifid=I["ifid"] if I else nan,
                         kid=Q["kid_all"] if Q else nan,
                         tok_gap=(R["h_real_mean"] - R["h_recon_mean"]) if R else nan,
                         gen_gap=(R["h_recon_mean"] - G["h_gen_mean"]) if R else nan,
                         q5_rec=100 * R["Q5"]["share_recon"] if R else nan, q5_gen=100 * G["Q5"]["share_gen"],
                         kid_q1=Q["Q1"]["kid"] if Q else nan, kid_q5=Q["Q5"]["kid"] if Q else nan))
    if not rows:
        raise SystemExit(f"no gen_eval.json under {a.out_root}/{a.dataset}_*")
    cols = ["run", "seed", "gfid", "rfid", "ifid", "kid", "tok_gap", "gen_gap", "q5_rec", "q5_gen", "kid_q1", "kid_q5"]
    fmt = lambda v: f"{v:.3f}" if isinstance(v, float) and abs(v) < 1 else (f"{v:.1f}" if isinstance(v, float) else str(v))
    print("  ".join(f"{c:>18s}" if c == "run" else f"{c:>8s}" for c in cols))
    for r in rows:
        print("  ".join(f"{fmt(r[c]):>18s}" if c == "run" else f"{fmt(r[c]):>8s}" for c in cols))
    import csv
    f = os.path.join(a.out_root, f"summary_{a.dataset}.csv")
    with open(f, "w", newline="") as fh:
        w = csv.DictWriter(fh, cols); w.writeheader(); w.writerows(rows)
    print("tok_gap = mean h real - recon; gen_gap = mean h recon - generated (pre-registered threshold 0.01);"
          " q5 = % of images in the most complex real quintile; KID x1000.\nsaved", f)


# =========================================================================================== 5. iFID
@torch.no_grad()
def ifid(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    _, dec, vargs = load_vae(a, dev)
    z, norm = load_latents(a, "mean")
    tr = norm(z["train_mu"]).flatten(1).to(dev)
    te = norm(z["test_mu"]).flatten(1).to(dev)
    nn_idx = torch.cat([torch.cdist(te[s:s + 500], tr).argmin(1) for s in range(0, len(te), 500)])
    mid = 0.5 * (te + tr[nn_idx])                      # linear midpoint, as the iFID default
    X = []
    for s in range(0, len(mid), 250):
        X.append(to_01(dec(denorm(mid[s:s + 250].reshape(-1, 4, 8, 8), z, dev))).cpu())
    real = test_images(a, vargs)
    res = {"tag": a.tag, "dataset": a.dataset, "ifid": fid_between(real, torch.cat(X), dev), "n": len(real)}
    json.dump(res, open(os.path.join(out_dir(a), "ifid.json"), "w"), indent=1)
    print(f"{a.tag} on {a.dataset}: iFID {res['ifid']:.2f}  (midpoint to nearest training latent, n={len(real)})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["encode", "train", "sample", "quint", "summary", "ifid", "recon"])
    ap.add_argument("--dataset", required=True, choices=["cifar10", "celeba"])
    ap.add_argument("--root", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--runs", default="runs_p2"); ap.add_argument("--which", default="last")
    ap.add_argument("--out_root", default="runs_p3"); ap.add_argument("--n_eval", type=int, default=10000)
    ap.add_argument("--workers", type=int, default=2); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="tiny run that only checks the pipeline")
    # DiT and training (identical for every VAE)
    ap.add_argument("--width", type=int, default=384); ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--steps", type=int, default=100000); ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-4); ap.add_argument("--ema", type=float, default=0.9999)
    ap.add_argument("--latent", default="sample", choices=["sample", "mean"], help="train on z~q(z|x) or on mu")
    ap.add_argument("--amp", action="store_true", help="fp16 autocast (faster on the RTX 4000)")
    ap.add_argument("--log_every", type=int, default=500); ap.add_argument("--save_every", type=int, default=5000)
    # sampling
    ap.add_argument("--n_gen", type=int, default=10000); ap.add_argument("--gen_bs", type=int, default=500)
    ap.add_argument("--sample_steps", type=int, default=100)
    ap.add_argument("--q_n", type=int, default=300, help="quint: images per quintile for the equal-n FID")
    ap.add_argument("--q_rep", type=int, default=20, help="quint: random subsets for the equal-n FID")
    a = ap.parse_args()
    if a.smoke:
        a.log_every, a.save_every = 50, 100
    {"encode": encode, "train": train, "sample": sample, "quint": quint, "summary": summary, "ifid": ifid,
     "recon": recon}[a.cmd](a)
