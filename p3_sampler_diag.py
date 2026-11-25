"""
p3_sampler_diag.py -- mechanism test: why are the generated latents shrunk (variance ratio < 1)?

Same trained DiT (runs_p3/<ds>_<tag>[_seed<k>]/dit.pt), same unchanged decoder; only the SAMPLER or the WEIGHTS change.
For each configuration it reports the variance ratio var_gen / var_real along the real-covariance eigen-directions
(bias-free: eigenbasis from real third A, variance from real third B, null from real third C), the top-quintile share
S5 on h and on absolute HF energy, the low-energy share, and gFID.

Configurations (default all; pick with --configs):
  euler100   Euler, 100 steps, EMA weights          (the baseline used everywhere in Paper 3)
  euler200   Euler, 200 steps                        euler400   Euler, 400 steps
  heun50     Heun (2nd order), 50 steps = 100 network calls, Euler on the last step
  sto100     stochastic sampler, 100 steps (DDIM-eta generalised to alpha_t = 1 - t, sigma_t = t): predict
             x0 = x - t v and eps = x + (1 - t) v, then x_next = (1 - t') x0 + sqrt(t'^2 - c^2) eps + c z with
             c = eta * sqrt(sigma^2_{t|t'} t'^2 / t^2); eta = 1 (default) = DDPM ancestral sampling.
             Unit-tested on exact Gaussian velocities (see the smoke test notes).
  sto400     the same stochastic sampler with 400 steps (ancestral sampling needs more steps than Euler)
  raw100     Euler, 100 steps, RAW (non-EMA) weights
For every configuration the table also shows the SAMPLER-ONLY ratio: the same sampler run on a perfect (exact) velocity
field for Gaussian data with the real per-decile variances. observed / sampler-only = the MODEL-side shrinkage.
Convention: x_t = (1 - t) x0 + t eps, network predicts v = eps - x0, sampling from t = 1 (noise) to t = 0.

Prediction written BEFORE the run (12-13 Nov 2026), from a Gaussian toy check at our latent scales (per-direction
data sd 0.38-2.0): if Euler discretisation were the cause, euler400 would lift every decile ratio to >= 0.97.
Expected instead: ratios rise by only ~0.01-0.03 from euler100 to euler400, i.e. the shrinkage is mostly model-side.
Reading: ratios move toward 1 with more steps / heun -> sampler; sto100 removes it -> deterministic-sampler collapse
(Zhang & Zou 2508.16154); raw differs from EMA -> weight averaging; none moves -> learned velocity field (objective /
capacity).

Run from D:\\new_vae_research (GPU, ~2.5 h per setting with all seven configs; euler400 and sto400 ~45 min each):
  python p3_sampler_diag.py --dataset celeba  --root D:\\new_vae_research\\celeba64 --tag D_strong_gan --seed 1
  python p3_sampler_diag.py --dataset cifar10 --root D:\\new_vae_research\\data     --tag D_strong_gan_v2
Output: runs_p3_sd/<ds>_<tag>[_seed<k>]/<config>.json, latents_<config>.npy, summary.json, and a printed table.
"""
import argparse, json, os, time
import numpy as np
import torch

from p3_latent_gen import load_vae, load_latents, make_dit, denorm, fid_between, test_images
from p3_h_check import spectra
from paper2_train import to_01

CONFIGS = ["euler100", "euler200", "euler400", "heun50", "sto100", "sto400", "raw100"]


def sfx(a):
    return f"_seed{a.seed}" if a.seed else ""


@torch.no_grad()
def run_sampler(model, cfg, n, dev, seed, eta, amp, bs):
    kind = "".join(c for c in cfg if c.isalpha()).replace("raw", "euler")
    steps = int("".join(c for c in cfg if c.isdigit()))

    def v(x, t):
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp and dev == "cuda"):
            return model(x, torch.full((len(x),), float(t), device=dev)).float()

    out = []
    for s in range(0, n, bs):
        m = min(bs, n - s)
        g = torch.Generator(device=dev).manual_seed(seed + s)
        x = torch.randn(m, 4, 8, 8, device=dev, generator=g)          # same starting noise for every config
        ts = torch.linspace(1, 0, steps + 1).tolist()
        for k in range(steps):
            t, tn = ts[k], ts[k + 1]
            if kind == "euler":
                x = x + (tn - t) * v(x, t)
            elif kind == "heun":
                d = v(x, t)
                xe = x + (tn - t) * d
                x = xe if tn == 0 else x + (tn - t) * 0.5 * (d + v(xe, tn))
            elif kind == "sto":
                d = v(x, t)
                x0, e = x - t * d, x + (1 - t) * d
                if tn == 0:
                    x = x0
                else:
                    # DDIM-eta step for alpha_t = 1 - t, sigma_t = t (eta = 1: DDPM ancestral posterior variance)
                    a_t, a_n = 1 - t, 1 - tn
                    var_ts = t ** 2 - (a_t / a_n) ** 2 * tn ** 2          # sigma^2_{t|t'}
                    c = eta * np.sqrt(max(var_ts, 0.0) * tn ** 2 / t ** 2)
                    z = torch.randn(x.shape, device=dev, generator=g)
                    x = a_n * x0 + np.sqrt(max(tn ** 2 - c ** 2, 0.0)) * e + c * z
            else:
                raise ValueError(cfg)
        out.append(x.cpu())
    return torch.cat(out)


class GaussVelocity(torch.nn.Module):
    """Exact velocity for data ~ N(0, diag(sd^2)): what a PERFECT model would output (sampler-only error)."""
    def __init__(self, sd):
        super().__init__(); self.sd = sd

    def forward(self, x, t):
        t = t.view(-1, 1, 1, 1); s2 = self.sd ** 2
        return x * (t - (1 - t) * s2) / ((1 - t) ** 2 * s2 + t ** 2)


def sampler_only_ratio(cfg, decile_var, eta, n=4000):
    """Variance ratio each decile would show with a perfect velocity field: the part due to the sampler alone."""
    sd = torch.tensor(np.sqrt(decile_var), dtype=torch.float32).repeat_interleave(26)[:256].view(4, 8, 8)
    cfg_e = cfg.replace("raw", "euler")
    zz = run_sampler(GaussVelocity(sd), cfg_e, n, "cpu", 12345, eta, False, n)
    r = (zz.var(0) / sd ** 2).flatten()
    return [float(r[i * 26:(i + 1) * 26].mean()) for i in range(10)]


def ratio_deciles(R, Gn, rng):
    perm = rng.permutation(len(R)); n3 = len(R) // 3
    RA, RB, RC = R[perm[:n3]], R[perm[n3:2 * n3]], R[perm[2 * n3:3 * n3]]
    w, V = np.linalg.eigh(np.cov(RA, rowvar=False))
    V = V[:, np.argsort(w)[::-1]]
    pv = lambda X: ((X - X.mean(0)) @ V).var(0, ddof=1)
    vB = pv(RB)
    r, r0 = pv(Gn) / vB, pv(RC) / vB
    idx = np.array_split(np.arange(V.shape[1]), 10)
    return [float(r[i].mean()) for i in idx], [float(r0[i].mean()) for i in idx], [float(vB[i].mean()) for i in idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["cifar10", "celeba"]); ap.add_argument("--root", required=True)
    ap.add_argument("--tag", required=True); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--runs", default="runs_p2"); ap.add_argument("--which", default="last")
    ap.add_argument("--base_root", default="runs_p3"); ap.add_argument("--out", default="runs_p3_sd")
    ap.add_argument("--configs", nargs="+", default=CONFIGS)
    ap.add_argument("--n", type=int, default=10000); ap.add_argument("--bs", type=int, default=1000)
    ap.add_argument("--eta", type=float, default=1.0); ap.add_argument("--n_eval", type=int, default=10000)
    ap.add_argument("--workers", type=int, default=0); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    base = argparse.Namespace(**vars(a)); base.seed = 0; base.out_root = a.base_root
    ck = torch.load(os.path.join(a.base_root, f"{a.dataset}_{a.tag}{sfx(a)}", "dit.pt"), map_location=dev, weights_only=False)
    z, norm = load_latents(base, ck["args"]["latent"])
    model = make_dit(argparse.Namespace(**ck["args"])).to(dev).eval()
    amp = bool(ck["args"].get("amp", False))
    _, dec, vargs = load_vae(argparse.Namespace(runs=a.runs, dataset=a.dataset, tag=a.tag, which=a.which), dev)
    dec.eval()

    rng = np.random.default_rng(0)
    mu, lv = z["train_mu"].astype(np.float32), z["train_lv"].astype(np.float32)
    zs = mu + np.exp(0.5 * lv) * rng.standard_normal(mu.shape).astype(np.float32)
    R = norm(zs).flatten(1).double().numpy()
    real = test_images(base, vargs)
    Er, Hr, hr = spectra(real)
    ch, ce = np.quantile(hr, [.2, .4, .6, .8]), np.quantile(Hr, [.2, .4, .6, .8])
    efl = np.quantile(Er, 0.05)
    od = os.path.join(a.out, f"{a.dataset}_{a.tag}{sfx(a)}"); os.makedirs(od, exist_ok=True)
    rows = {}
    for cfg in a.configs:
        model.load_state_dict(ck["model"] if cfg.startswith("raw") else ck["ema"])
        t0 = time.time()
        zn = run_sampler(model, cfg, a.n, dev, 777 + 1000003 * a.seed, a.eta, amp, a.bs)
        sec = time.time() - t0
        np.save(os.path.join(od, f"latents_{cfg}.npy"), zn.numpy())
        ratios, null, dvar = ratio_deciles(R, zn.flatten(1).double().numpy(), np.random.default_rng(0))
        toy = sampler_only_ratio(cfg, dvar, a.eta)
        with torch.no_grad():
            fake = torch.cat([to_01(dec(denorm(zn[s:s + 250].to(dev), z, dev))).clamp(0, 1).cpu() for s in range(0, len(zn), 250)])
        E, H, h = spectra(fake)
        qh, qe = np.digitize(h, ch), np.digitize(H, ce)
        res = {"config": cfg, "eta": a.eta if cfg.startswith("sto") else None, "seconds": sec, "n": len(zn),
               "ratio_deciles": ratios, "null_deciles": null, "ratio_mean": float(np.mean(ratios)),
               "sampler_only_deciles": toy, "model_side_deciles": [r / q for r, q in zip(ratios, toy)],
               "S5_h": float(100 * (qh == 4).mean()), "S5_absHF": float(100 * (qe == 4).mean()),
               "lowE45": float(100 * (E < efl)[qh >= 3].mean()) if (qh >= 3).any() else float("nan"),
               "gfid": fid_between(real, fake, dev)}
        json.dump(res, open(os.path.join(od, f"{cfg}.json"), "w"), indent=1)
        rows[cfg] = res
        print(f"{cfg}: {sec:.0f}s, ratio mean {res['ratio_mean']:.3f}, S5 absHF {res['S5_absHF']:.1f}, gFID {res['gfid']:.2f}", flush=True)
    json.dump(rows, open(os.path.join(od, "summary.json"), "w"), indent=1)

    print(f"\n{a.dataset}_{a.tag}{sfx(a)}: variance ratio gen/real per decile (1 = largest real variance ... 10 = smallest)")
    print(f"{'config':>9s} " + " ".join(f"{i:>5d}" for i in range(1, 11)) + f" {'mean':>6s} {'S5 h':>6s} {'S5 aHF':>7s} {'lowE45':>7s} {'gFID':>6s} {'sec':>6s}")
    first = next(iter(rows.values()))
    print(f"{'null':>9s} " + " ".join(f"{x:5.3f}" for x in first["null_deciles"]))
    for c, r in rows.items():
        print(f"{c:>9s} " + " ".join(f"{x:5.3f}" for x in r["ratio_deciles"])
              + f" {r['ratio_mean']:6.3f} {r['S5_h']:6.1f} {r['S5_absHF']:7.1f} {r['lowE45']:7.1f} {r['gfid']:6.2f} {r['seconds']:6.0f}")
    print("\nsampler-only ratio (perfect Gaussian velocity, same sampler, same per-decile real variance) and the"
          " model-side part = observed / sampler-only:")
    for c, r in rows.items():
        print(f"{c:>9s} sampler " + " ".join(f"{x:5.3f}" for x in r["sampler_only_deciles"]))
        print(f"{'':>9s} model   " + " ".join(f"{x:5.3f}" for x in r["model_side_deciles"]))
    if "euler400" in rows:
        ok = all(0.97 <= x <= 1.03 for x in rows["euler400"]["ratio_deciles"])
        print("\npre-stated prediction: euler400 lifts every decile ratio to >= 0.97 only if Euler discretisation is the cause"
              f" -> {'YES, sampler-side' if ok else 'NO, mostly model-side'}")
    print(f"saved {od}")


if __name__ == "__main__":
    main()
