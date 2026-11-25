"""
p4_kid.py -- Paper 4: is the KID change from flip augmentation (and from flip + L0) larger than its noise?

The flip test (p3_ema_study.py fliptest) compares single gFID values. This script gives the uncertainty of the
differences so the paper can report "better by X [95% range]" instead of a bare inequality.

Arms (all at 100k steps, ema9999, Euler 100 steps, the SAME eval noise as `eval` and `maps`, seed 777 + 1000003*seed):
  noflip      runs_p3_ema/<setting>/snap_100000.pt, no map
  noflip_l0   the same samples after the L0 map (Gaussian OT, fitted on 50k other generated latents, as in `maps`)
  flip        runs_p3_flip/<setting>/snap_100000.pt, no map
  flip_l0     flip samples after their own L0 map
Comparisons (arm minus reference, KID x1000, lower = better):
  flip - noflip, flip_l0 - noflip, flip_l0 - noflip_l0
Every arm uses the same starting noise, so in each repetition the SAME image indices are drawn for both arms
(paired by noise) and the SAME real subset is used. Reported for all images and for the most complex quintile
(Q5 by h, as in p3_kid_diff.py, and Q5 by absolute HF energy, as in S5). 100 repetitions, subsets of 2000 / 500.

Fixed BEFORE the run (23 Nov 2026): flip "improves KID" on a setting if the 95% range of flip - noflip lies below 0
(all images). A range that contains 0 is reported as "no measurable change". No other reading is made.

Cost: per setting two models x (50k fit + 10k eval latents) + decoding + Inception features of 4 x 10k images,
about 45-60 min on the RTX 4000. Features are cached in runs_p4_kid/<setting>/feats.npz, so `--redo` is only needed
if a model changes.

Run from D:\\new_vae_research (venv active), one line per setting:
  python p4_kid.py --dataset cifar10 --root D:\\new_vae_research\\data --tag D_strong_gan_v2
  python p4_kid.py --dataset cifar10 --root D:\\new_vae_research\\data --tag D_strong_gan_v2 --seed 1
  python p4_kid.py --dataset celeba  --root <celeba root> --tag D_strong_gan
  python p4_kid.py --dataset celeba  --root <celeba root> --tag D_strong_gan --seed 1
Use the same --root you used for p3_ema_study.py on that dataset. `--smoke` runs it on a few hundred images.
"""
import argparse, json, os, time
import numpy as np
import torch

from p3_latent_gen import load_vae, load_latents, make_dit, denorm, test_images, inception_feats, kid_np
from p3_sampler_diag import run_sampler
from p3_h_check import spectra
from p3_ema_study import setting, base_ns, _ot
from paper2_train import to_01

ARMS = ["noflip", "noflip_l0", "flip", "flip_l0"]
COMPARE = [("flip", "noflip"), ("flip_l0", "noflip"), ("flip_l0", "noflip_l0")]


@torch.no_grad()
def build_feats(a, dev):
    od = os.path.join(a.out, setting(a)); os.makedirs(od, exist_ok=True)
    fp = os.path.join(od, "feats_smoke.npz" if a.smoke else "feats.npz")
    if os.path.exists(fp) and not a.redo:
        print(f"using cached features {fp}")
        return dict(np.load(fp))
    ck0 = torch.load(os.path.join(a.base_root, setting(a), "dit.pt"), map_location=dev, weights_only=False)
    ca = argparse.Namespace(**ck0["args"])
    z, norm = load_latents(base_ns(a), ca.latent)
    _, dec, vargs = load_vae(argparse.Namespace(runs=a.runs, dataset=a.dataset, tag=a.tag, which=a.which), dev)
    model = make_dit(ca).to(dev).eval()
    amp = bool(getattr(ca, "amp", False))
    rng = np.random.default_rng(0)                   # real posterior samples, exactly as in `maps`
    mu, lv = z["train_mu"].astype(np.float32), z["train_lv"].astype(np.float32)
    R = norm(mu + np.exp(0.5 * lv) * rng.standard_normal(mu.shape).astype(np.float32)).flatten(1).double().numpy()
    real = test_images(base_ns(a), vargs)
    if a.smoke:
        real = real[:400]
    print("Inception features of the real test images ...", flush=True)
    out = {"real": inception_feats(real, dev)}
    Er, Hr, hr = spectra(real)
    out["real_h"], out["real_H"] = hr, Hr
    n_fit, n_ev = (300, 400) if a.smoke else (a.n_fit, a.n)
    for run, folder in (("noflip", a.noflip_out), ("flip", a.flip_out)):
        p = os.path.join(folder, setting(a), f"snap_{a.step}.pt")
        if not os.path.exists(p):
            raise SystemExit(f"missing {p}")
        model.load_state_dict(torch.load(p, map_location=dev, weights_only=False)[a.weights])
        t0 = time.time()
        Zf = run_sampler(model, "euler100", n_fit, dev, 424242 + 1000003 * a.seed, 1.0, amp, a.bs)
        Ze = run_sampler(model, "euler100", n_ev, dev, 777 + 1000003 * a.seed, 1.0, amp, a.bs)
        F, G = Zf.flatten(1).double().numpy(), Ze.flatten(1).double().numpy()
        mg, mr, A, _, _ = _ot(F, R)
        for arm, Gk in ((run, G), (run + "_l0", (G - mg) @ A.T + mr)):
            zn = torch.from_numpy(Gk.astype(np.float32)).view(Ze.shape)
            fake = torch.cat([to_01(dec(denorm(zn[s:s + 250].to(dev), z, dev))).clamp(0, 1).cpu()
                              for s in range(0, len(zn), 250)])
            _, H, h = spectra(fake)
            out[arm], out[arm + "_h"], out[arm + "_H"] = inception_feats(fake, dev), h, H
            print(f"{arm}: {len(fake)} images, features done ({(time.time() - t0) / 60:.1f} min)", flush=True)
    np.savez(fp, **out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["cifar10", "celeba"]); ap.add_argument("--root", required=True)
    ap.add_argument("--tag", required=True); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--runs", default="runs_p2"); ap.add_argument("--which", default="last")
    ap.add_argument("--base_root", default="runs_p3")
    ap.add_argument("--noflip_out", default="runs_p3_ema"); ap.add_argument("--flip_out", default="runs_p3_flip")
    ap.add_argument("--out", default="runs_p4_kid")
    ap.add_argument("--step", type=int, default=100000); ap.add_argument("--weights", default="ema9999")
    ap.add_argument("--n", type=int, default=10000); ap.add_argument("--n_fit", type=int, default=50000)
    ap.add_argument("--bs", type=int, default=1000); ap.add_argument("--n_eval", type=int, default=10000)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--m", type=int, default=2000); ap.add_argument("--m_q5", type=int, default=500)
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--redo", action="store_true"); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    D = build_feats(a, dev)
    if a.smoke:
        a.m, a.m_q5, a.reps = 150, 50, 10
    cut_h = np.quantile(D["real_h"], [.2, .4, .6, .8]); cut_H = np.quantile(D["real_H"], [.2, .4, .6, .8])
    q5 = {"h": {k: np.where(np.digitize(D[k + "_h" if k != "real" else "real_h"], cut_h) == 4)[0] for k in ["real"] + ARMS},
          "aHF": {k: np.where(np.digitize(D[k + "_H" if k != "real" else "real_H"], cut_H) == 4)[0] for k in ["real"] + ARMS}}
    rng = np.random.default_rng(0)
    n_gen = len(D["noflip"])
    res = {"setting": setting(a), "step": a.step, "weights": a.weights, "reps": a.reps, "m": a.m, "m_q5": a.m_q5,
           "single": {}, "diff": {}}
    for k in ARMS:                                       # plain KID of each arm on all images (one value, for the table)
        res["single"][k] = kid_np(D["real"], D[k]) if a.smoke else kid_np(D["real"][:5000], D[k][:5000])
    for arm, ref in COMPARE:
        d = {"all": [], "q5_h": [], "q5_aHF": []}
        for _ in range(a.reps):
            Rr = D["real"][rng.choice(len(D["real"]), a.m, replace=False)]
            idx = rng.choice(n_gen, a.m, replace=False)             # same noise indices for both arms
            d["all"].append(kid_np(Rr, D[arm][idx]) - kid_np(Rr, D[ref][idx]))
            for key, Q in (("q5_h", q5["h"]), ("q5_aHF", q5["aHF"])):
                mq = min(a.m_q5, len(Q["real"]), len(Q[arm]), len(Q[ref]))
                if mq < 30:
                    continue
                R5 = D["real"][rng.choice(Q["real"], mq, replace=False)]
                d[key].append(kid_np(R5, D[arm][rng.choice(Q[arm], mq, replace=False)])
                              - kid_np(R5, D[ref][rng.choice(Q[ref], mq, replace=False)]))
        s = lambda v: [float(np.mean(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v else [float("nan")] * 3
        res["diff"][f"{arm}-{ref}"] = {k: s(v) for k, v in d.items()}
    od = os.path.join(a.out, setting(a))
    json.dump(res, open(os.path.join(od, "kid_smoke.json" if a.smoke else "kid.json"), "w"), indent=1)
    print(f"\n{setting(a)}  step {a.step} {a.weights}: KID x1000 (lower = better)")
    print("  single KID (5000 real vs 5000 generated): " + ", ".join(f"{k} {v:.2f}" for k, v in res["single"].items()))
    print(f"  difference, mean [95% range] over {a.reps} subsets (all: m={a.m}, paired by noise; Q5: m<={a.m_q5})")
    print(f"  {'comparison':>20s} {'all images':>24s} {'Q5 by h':>24s} {'Q5 by abs HF':>24s}")
    for name, v in res["diff"].items():
        print(f"  {name:>20s}" + "".join(f"   {x[0]:6.2f} [{x[1]:6.2f}, {x[2]:6.2f}]" for x in (v["all"], v["q5_h"], v["q5_aHF"])))
    f = res["diff"]["flip-noflip"]["all"]
    verdict = "flip IMPROVES KID" if f[2] < 0 else ("flip makes KID WORSE" if f[1] > 0 else "no measurable change from flip")
    print(f"  RULE (fixed 23 Nov 2026), all images: flip - noflip range [{f[1]:.2f}, {f[2]:.2f}] -> {verdict}")


if __name__ == "__main__":
    main()
