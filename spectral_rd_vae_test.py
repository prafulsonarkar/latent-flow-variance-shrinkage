"""
spectral_rd_vae_test.py -- Idea A, decisive test on a trained VAE (run on the lab GPU).

Pre-registered question:
    Does an image's spectral slope alpha (plus its pixel variance), measured with NO training,
    predict that image's rate-distortion curve MSE_i(d) = a_i * d^-k_i under the VAE, and does
    allocating latent dims from that prediction beat (1) fixed d and (2) a variance-only allocator
    at the same mean d?

Pre-registered pass criteria (write these in the lab notebook BEFORE running):
    P1  Spearman rho(alpha, k_i)              >= 0.5   on the test half
    P2  alpha-allocator agg-PSNR gain over fixed d  >= +0.3 dB  at mean d = d_avg
    P3  alpha-allocator beats variance-only allocator on agg PSNR
    If P1 fails, stop: the idea does not transfer from linear theory to the VAE.

REQUIREMENT: the model must have ORDERED latents, i.e. be trained with nested dropout (or a
nested/Matryoshka objective), so "use the first d dims" is meaningful. A plain VAE fails this:
Paper 1's rate analysis found its dims are not ordered by index (Spearman ~ 0).

Usage:
    1. Implement `recon_fn(x, d)` for your model (examples at the bottom).
    2. python spectral_rd_vae_test.py --ckpt path/to/model.pt --d_grid 8 16 32 64 96 128 192 256 332
    3. Outputs results/spectral_rd_<tag>.npz and a printed report.
"""
import argparse, os
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

# ============================================================== spectral features (training-free)
def spectral_slope_batch(x, fmin=2, fmax=16):
    """x: [B, C, H, W] in [0,1]. Returns alpha per image (grayscale, Hann-windowed, radial fit)."""
    g = x.mean(1)                                                       # [B, H, W]
    B, H, W = g.shape
    hann = torch.outer(torch.hann_window(H, periodic=False), torch.hann_window(W, periodic=False)).to(x)
    g = (g - g.mean((1, 2), keepdim=True)) * hann
    P = torch.fft.fft2(g).abs() ** 2
    fy = torch.fft.fftfreq(H, device=x.device) * H
    fx = torch.fft.fftfreq(W, device=x.device) * W
    R = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2).round().long()
    f = torch.arange(fmin, fmax + 1, device=x.device)
    radial = torch.stack([(P * (R == r)).sum((1, 2)) / (R == r).sum() for r in f.tolist()], 1)
    lf, lp = torch.log(f.float()), torch.log(radial + 1e-12)
    lf_c = lf - lf.mean()
    slope = ((lp - lp.mean(1, keepdim=True)) * lf_c).sum(1) / (lf_c ** 2).sum()
    return (-slope).cpu().numpy()


def band_energies(x, cuts=(4, 8, 16)):
    """log energy above each radial cut (cycles/image). Sets the error SCALE a_i: by the n-width
    argument, MSE(d) is the energy left beyond the d-th mode, so tail energies predict its level."""
    g = x.mean(1)
    B, H, W = g.shape
    G = torch.fft.fft2(g - g.mean((1, 2), keepdim=True)).abs() ** 2 / (H * W) ** 2
    fy = torch.fft.fftfreq(H, device=x.device) * H
    fx = torch.fft.fftfreq(W, device=x.device) * W
    R = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    return torch.stack([torch.log((G * (R > c)).sum((1, 2)) + 1e-12) for c in cuts], 1).cpu().numpy()


# ============================================================== per-image R-D curves from the model
@torch.no_grad()
def collect(recon_fn, loader, d_grid, device):
    alpha, logvar, band, mse = [], [], [], []
    for x, *_ in loader:
        x = x.to(device)
        alpha.append(spectral_slope_batch(x))
        band.append(band_energies(x))
        logvar.append(torch.log(x.flatten(1).var(1) + 1e-12).cpu().numpy())
        mse.append(np.stack([((recon_fn(x, d).clamp(0, 1) - x) ** 2).flatten(1).mean(1).cpu().numpy()
                             for d in d_grid], 1))
    return np.concatenate(alpha), np.concatenate(logvar), np.concatenate(band), np.concatenate(mse)


# ============================================================== analysis
def fit_power(mse, d_grid, fit_idx):
    ld = np.log(np.asarray(d_grid)[fit_idx])
    out = np.array([np.polyfit(ld, np.log(m[fit_idx] + 1e-12), 1) for m in mse])
    return -out[:, 0], out[:, 1]


def lagrange_alloc(curves, d_grid, d_avg):
    """Operational R-D optimum over the available d values: argmin_d MSE_i(d) + lam*d."""
    d_grid = np.asarray(d_grid, float)
    def d_idx(lam):
        return np.argmin(curves + lam * d_grid[None, :], axis=1)
    lo, hi = 1e-14, 1e2
    for _ in range(100):
        mid = np.sqrt(lo * hi)
        if d_grid[d_idx(mid)].mean() > d_avg:
            lo = mid
        else:
            hi = mid
    return d_idx(np.sqrt(lo * hi))


def variance_alloc(logvar, d_grid, d_avg):
    """Naive baseline: d proportional to pixel variance, snapped to the grid, mean matched."""
    d_grid = np.asarray(d_grid, float)
    v = np.exp(logvar)
    lo, hi = 1e-6, 1e6
    for _ in range(100):
        s = np.sqrt(lo * hi)
        idx = np.abs(np.log(np.clip(s * v, d_grid[0], d_grid[-1]))[:, None] - np.log(d_grid)[None, :]).argmin(1)
        if d_grid[idx].mean() > d_avg:
            hi = s
        else:
            lo = s
    return idx


def report_row(name, mse_curves, idx, d_grid, alpha):
    m = mse_curves[np.arange(len(idx)), idx]
    psnr = 10 * np.log10(1 / m)
    q = np.digitize(alpha, np.quantile(alpha, [.2, .4, .6, .8]))
    qp = np.array([psnr[q == j].mean() for j in range(5)])
    print(f"  {name:10s} mean d {np.asarray(d_grid)[idx].mean():6.1f}  agg PSNR {10*np.log10(1/m.mean()):6.2f}"
          f"  mean PSNR {psnr.mean():6.2f}  alpha-quintile PSNR {np.round(qp, 2)}")
    return 10 * np.log10(1 / m.mean())


def analyse(alpha, logvar, band, mse, d_grid, d_avg, fit_lo, fit_hi, seed=0):
    d_grid = list(d_grid)
    fit_idx = [i for i, d in enumerate(d_grid) if fit_lo <= d <= fit_hi]
    k, la = fit_power(mse, d_grid, fit_idx)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(alpha))
    cal, te = perm[: len(perm) // 2], perm[len(perm) // 2:]

    c1, c0 = np.polyfit(alpha[cal], k[cal], 1)
    X = lambda ix: np.c_[np.ones(len(ix)), logvar[ix], alpha[ix], band[ix]]
    B = np.linalg.lstsq(X(cal), la[cal], rcond=None)[0]
    k_hat = np.clip(c0 + c1 * alpha[te], 0.02, None)
    la_hat = X(te) @ B          # scale from variance + slope + tail band energies

    rho_ak = spearmanr(alpha[te], k[te]).correlation
    print(f"\nn = {len(alpha)} (calibration {len(cal)}, test {len(te)}),  fit d in [{fit_lo}, {fit_hi}]")
    print(f"alpha = {alpha.mean():.2f} +- {alpha.std():.2f};  k = {k.mean():.3f} +- {k.std():.3f}")
    print(f"theory k = alpha/2 - 1 -> {np.mean(alpha/2-1):.3f};  calibrated k = {c0:.3f} + {c1:.3f}*alpha")
    print(f"P1  rho(alpha, k) on test         = {rho_ak:.3f}   (pass if >= 0.5)")
    print(f"    rho(k_hat, k)                 = {spearmanr(k_hat, k[te]).correlation:.3f}")
    print(f"    rho(log a_hat, log a)         = {spearmanr(la_hat, la[te]).correlation:.3f}")
    print(f"    rho(logvar, k) [baseline]     = {spearmanr(logvar[te], k[te]).correlation:.3f}")

    curves_te = mse[te]
    dg = np.asarray(d_grid, float)
    pred = np.exp(la_hat)[:, None] * dg[None, :] ** (-k_hat[:, None])
    fixed_idx = np.full(len(te), int(np.abs(dg - d_avg).argmin()))
    print(f"\nallocation at mean d = {d_avg} (fixed baseline uses d = {int(dg[fixed_idx[0]])})")
    p_fix = report_row("fixed", curves_te, fixed_idx, d_grid, alpha[te])
    p_var = report_row("variance", curves_te, variance_alloc(logvar[te], d_grid, d_avg), d_grid, alpha[te])
    p_alp = report_row("alpha", curves_te, lagrange_alloc(pred, d_grid, d_avg), d_grid, alpha[te])
    p_orc = report_row("oracle", curves_te, lagrange_alloc(curves_te, d_grid, d_avg), d_grid, alpha[te])
    print(f"P2  alpha - fixed    = {p_alp - p_fix:+.2f} dB  (pass if >= +0.3)")
    print(f"P3  alpha - variance = {p_alp - p_var:+.2f} dB  (pass if > 0)")
    print(f"    share of oracle gain captured = {(p_alp - p_fix) / max(p_orc - p_fix, 1e-9):.0%}")


# ============================================================== model hooks (edit for your model)
def make_recon_fn_flat(model):
    """Flat nested-dropout VAE: encode -> mu [B, 332]; keep first d dims, zero the rest; decode."""
    def recon(x, d):
        mu, _ = model.encode(x)
        mask = torch.zeros_like(mu); mask[:, :d] = 1
        return model.decode(mu * mask)
    return recon


def make_recon_fn_spatial(model):
    """Spatial nested VAE: mu [B, C, 8, 8]; d counts channels*64, keep first d//64 channels."""
    def recon(x, d):
        mu, _ = model.encode(x)
        c = max(1, d // (mu.shape[2] * mu.shape[3]))
        mask = torch.zeros_like(mu); mask[:, :c] = 1
        return model.decode(mu * mask)
    return recon


def cifar_loader(root, res=64, n=None, bs=256):
    import torchvision, torchvision.transforms as T
    tf = T.Compose([T.Resize(res, interpolation=T.InterpolationMode.BICUBIC), T.ToTensor()])
    ds = torchvision.datasets.CIFAR10(root, train=False, download=True, transform=tf)
    if n:
        ds = torch.utils.data.Subset(ds, range(n))
    return torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=False, num_workers=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=False)
    ap.add_argument("--data", default="./data")
    ap.add_argument("--tag", default="cifar10_64")
    ap.add_argument("--d_grid", type=int, nargs="+", default=[8, 16, 32, 64, 96, 128, 192, 256, 332])
    ap.add_argument("--d_avg", type=float, default=128)
    ap.add_argument("--fit_lo", type=int, default=16)
    ap.add_argument("--fit_hi", type=int, default=332)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--analyse_only", default=None, help="path to a saved .npz")
    a = ap.parse_args()

    if a.analyse_only:
        z = np.load(a.analyse_only)
        analyse(z["alpha"], z["logvar"], z["band"], z["mse"], z["d_grid"].tolist(), a.d_avg, a.fit_lo, a.fit_hi)
    else:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        # ---- load YOUR model here --------------------------------------------------------
        # from models import NestedVAE
        # model = NestedVAE(...); model.load_state_dict(torch.load(a.ckpt, map_location=dev)); model.eval().to(dev)
        # recon_fn = make_recon_fn_flat(model)
        raise SystemExit("Edit the model-loading block in __main__ for your checkpoint.")
        alpha, logvar, band, mse = collect(recon_fn, cifar_loader(a.data, n=a.n), a.d_grid, dev)
        os.makedirs("results", exist_ok=True)
        out = f"results/spectral_rd_{a.tag}.npz"
        np.savez(out, alpha=alpha, logvar=logvar, band=band, mse=mse, d_grid=np.array(a.d_grid))
        print("saved", out)
        analyse(alpha, logvar, band, mse, a.d_grid, a.d_avg, a.fit_lo, a.fit_hi)
