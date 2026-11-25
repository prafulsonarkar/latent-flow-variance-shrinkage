"""
p3_h_check.py -- two robustness checks on the Paper 3 complexity strata, run on files `sample` already saved.

1. Is h(x) fooled by low-contrast images?  h = share of (non-DC) spectral energy above 8 cycles/image is a RATIO, so a
   nearly flat image with faint noise can get a high h (seen in the strata grids: a flat brown image in A's Q4, a nearly
   white image in G's Q5). For every run this script reports
     - low_energy: % of generated images whose total non-DC energy is below the 5th percentile of the real test images
       (5% expected if matched), overall and inside Q4/Q5,
     - Q5 share three ways: h as in the paper; h with low-energy images (real and generated) left out;
       absolute high-frequency energy instead of the ratio (quintile cuts always from the real images).
   If the three Q5 shares tell the same story, the finding does not depend on the ratio.
   Baseline: the same low-energy % for REAL images that h puts in Q4/Q5 (the ratio affects real images too).
   Tokenizer vs generator gap under both measures: the VAE's reconstructions of the test images are made again
   (CPU is fine, ~1-2 min) and their Q5 share is computed with h and with absolute HF energy.
2. Is the KID trend across quintiles larger than its noise?  KID(Q5) - KID(Q1) on --reps paired random subsets of
   --m images each; the 2.5-97.5% range is conservative (subsets are smaller than the full quintiles).
Also writes h_check_grid.png: the 16 lowest-energy generated images that h puts in Q4/Q5 (to look at, by eye).

Run from D:\\new_vae_research after `sample` (needs gen_images.npz in each run folder; GPU optional, ~2-3 min per run):
  python p3_h_check.py --root D:\\new_vae_research\\data --tags A_small_mse B_strong_mse G_small_lpips_gan D_strong_gan_v2
Output: runs_p3/<ds>_<tag>/h_check.json, h_check_grid.png, and one summary table printed at the end.
"""
import argparse, json, os
import numpy as np
import torch

from nested_vae_spectral import get_datasets
from p3_latent_gen import inception_feats, kid_np, load_vae
from paper2_train import to_pm1, to_01


def spectra(x, cut=8):
    """(total non-DC energy, energy above `cut` cycles/image, ratio h) per image -- same FFT as hf_fraction."""
    E, H = [], []
    for s in range(0, len(x), 1000):
        g = x[s:s + 1000].mean(1)
        G = torch.fft.fft2(g - g.mean((1, 2), keepdim=True)).abs() ** 2
        n = g.shape[1]
        f = torch.fft.fftfreq(n) * n
        R = torch.sqrt(f[:, None] ** 2 + f[None, :] ** 2)
        E.append(G.sum((1, 2))); H.append((G * (R > cut)).sum((1, 2)))
    E, H = torch.cat(E).double().numpy(), torch.cat(H).double().numpy()
    return E, H, H / (E + 1e-12)


def shares(v_real, v_gen):
    cuts = np.quantile(v_real, [.2, .4, .6, .8])
    return np.digitize(v_real, cuts), np.digitize(v_gen, cuts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar10"); ap.add_argument("--root", required=True)
    ap.add_argument("--tags", nargs="+", required=True); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_root", default="runs_p3"); ap.add_argument("--n_eval", type=int, default=10000)
    ap.add_argument("--m", type=int, default=300); ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--runs", default="runs_p2"); ap.add_argument("--which", default="last")
    ap.add_argument("--no_kid", action="store_true", help="skip the KID part (no Inception features; much faster)")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    _, ev = get_datasets(a.dataset, a.root, 64, n_eval=a.n_eval)
    real = torch.stack([ev[i][0] for i in range(len(ev))])
    Er, Hr, hr = spectra(real)
    e_floor = np.quantile(Er, 0.05)
    qr_h = shares(hr, hr)[0]
    real_low_q45 = float(100 * (Er < e_floor)[qr_h >= 3].mean())
    Fr = None
    table = []
    for tag in a.tags:
        d = os.path.join(a.out_root, f"{a.dataset}_{tag}" + (f"_seed{a.seed}" if a.seed else ""))
        f = os.path.join(d, "gen_images.npz")
        if not os.path.exists(f):
            print(f"skip {tag}: {f} not found (run sample first)"); continue
        fake = torch.from_numpy(np.load(f)["x"]).float() / 255
        Eg, Hg, hg = spectra(fake)
        low = Eg < e_floor
        qr, qg = shares(hr, hg)
        keep_r, keep_g = Er >= e_floor, ~low
        qr2, qg2 = shares(hr[keep_r], hg[keep_g])
        _, qg3 = shares(Hr, Hg)
        res = {"tag": tag, "seed": a.seed, "energy_floor": float(e_floor),
               "low_energy_gen_pct": float(100 * low.mean()),
               "low_energy_in_Q4Q5_pct": float(100 * low[qg >= 3].mean()) if (qg >= 3).any() else float("nan"),
               "q5_share_h": float(100 * (qg == 4).mean()),
               "q5_share_h_no_low_energy": float(100 * (qg2 == 4).mean()),
               "q5_share_abs_hf": float(100 * (qg3 == 4).mean()),
               "shares_abs_hf": [float(100 * (qg3 == j).mean()) for j in range(5)],
               "real_low_energy_in_Q4Q5_pct": real_low_q45}
        # reconstructions of the test images (same as `p3_latent_gen.py recon`), to split tokenizer vs generator
        try:
            with torch.no_grad():
                enc, dec, _ = load_vae(argparse.Namespace(runs=a.runs, dataset=a.dataset, tag=tag, which=a.which), dev)
                rec = torch.cat([to_01(dec(enc(to_pm1(real[s:s + 250].to(dev)))[0])).cpu() for s in range(0, len(real), 250)])
            Ec, Hc, hc = spectra(rec)
            q5 = lambda vr, vx: float(100 * (shares(vr, vx)[1] == 4).mean())
            res.update({"q5_recon_h": q5(hr, hc), "q5_recon_abs_hf": q5(Hr, Hc),
                        "tok_gap_q5_h": 20 - q5(hr, hc), "gen_gap_q5_h": q5(hr, hc) - res["q5_share_h"],
                        "tok_gap_q5_abs_hf": 20 - q5(Hr, Hc), "gen_gap_q5_abs_hf": q5(Hr, Hc) - res["q5_share_abs_hf"]})
        except FileNotFoundError as e:
            print(f"  {tag}: no VAE checkpoint for the reconstruction part ({e})")
        # KID(Q5) - KID(Q1), paired subsets
        if a.no_kid:
            res.update({"kid_q5_minus_q1": float("nan"), "kid_q5_minus_q1_range95": [float("nan")] * 2, "kid_m": 0})
        elif Fr is None:
            print("Inception features of the real test images ...", flush=True)
            Fr = inception_feats(real, dev)
        if not a.no_kid:
            print(f"{tag}: Inception features of the generated images ...", flush=True)
            Fg = inception_feats(fake, dev)
            rng = np.random.default_rng(0)
            r1, r5, g1, g5 = (np.where(qr == 0)[0], np.where(qr == 4)[0], np.where(qg == 0)[0], np.where(qg == 4)[0])
            m = min(a.m, len(g1), len(g5))
            diffs = [kid_np(Fr[rng.choice(r5, m, replace=False)], Fg[rng.choice(g5, m, replace=False)])
                     - kid_np(Fr[rng.choice(r1, m, replace=False)], Fg[rng.choice(g1, m, replace=False)])
                     for _ in range(a.reps)] if m >= 50 else [float("nan")]
            res.update({"kid_q5_minus_q1": float(np.mean(diffs)), "kid_q5_minus_q1_range95": [float(np.percentile(diffs, 2.5)),
                        float(np.percentile(diffs, 97.5))], "kid_m": int(m)})
        json.dump(res, open(os.path.join(d, "h_check.json"), "w"), indent=1)
        # grid of the lowest-energy generated images that h calls complex
        idx = np.where(qg >= 3)[0]
        idx = idx[np.argsort(Eg[idx])][:16]
        if len(idx):
            from torchvision.utils import save_image
            up = torch.nn.functional.interpolate(fake[idx], scale_factor=3, mode="nearest")
            save_image(up, os.path.join(d, "h_check_grid.png"), nrow=8, padding=4, pad_value=1)
        table.append(res)
    print(f"\nreal energy floor (5th percentile) = {e_floor:.4g};  5% of generated images below it = matched")
    print(f"{'run':>20s} {'low-E gen %':>11s} {'low-E in Q4-5 %':>15s} {'Q5 h':>6s} {'Q5 h, no low-E':>15s}"
          f" {'Q5 abs HF':>10s} {'KID Q5-Q1 [95% range]':>26s}")
    for r in table:
        print(f"{r['tag']:>20s} {r['low_energy_gen_pct']:11.1f} {r['low_energy_in_Q4Q5_pct']:15.1f} {r['q5_share_h']:6.1f}"
              f" {r['q5_share_h_no_low_energy']:15.1f} {r['q5_share_abs_hf']:10.1f}"
              f"   {r['kid_q5_minus_q1']:6.1f} [{r['kid_q5_minus_q1_range95'][0]:5.1f}, {r['kid_q5_minus_q1_range95'][1]:5.1f}]")
    print(f"\nbaseline: {real_low_q45:.1f}% of the REAL images that h puts in Q4/Q5 are low-energy "
          "(compare with the 'low-E in Q4-5' column)")
    print(f"\nQ5 share (%), 20 = matched.  tokenizer gap = 20 - recon;  generator gap = recon - generated")
    print(f"{'run':>20s} {'recon h':>8s} {'gen h':>6s} {'tok gap h':>10s} {'gen gap h':>10s}   "
          f"{'recon absHF':>11s} {'gen absHF':>9s} {'tok gap':>8s} {'gen gap':>8s}")
    for r in table:
        if "q5_recon_h" in r:
            print(f"{r['tag']:>20s} {r['q5_recon_h']:8.1f} {r['q5_share_h']:6.1f} {r['tok_gap_q5_h']:10.1f} {r['gen_gap_q5_h']:10.1f}   "
                  f"{r['q5_recon_abs_hf']:11.1f} {r['q5_share_abs_hf']:9.1f} {r['tok_gap_q5_abs_hf']:8.1f} {r['gen_gap_q5_abs_hf']:8.1f}")
    print("Q5 columns: % of generated images in the most complex real quintile (20% = matched). KID x1000; a range that"
          " contains 0 means no detectable quality trend from simple to complex images.")


if __name__ == "__main__":
    main()
