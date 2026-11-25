"""
p3_ema_study.py -- does weight averaging (EMA) cause the latent variance shrinkage, and does longer training remove it?

Background (13 Nov 2026, p3_sampler_diag.py): with RAW weights the generated-latent variance ratio is 0.99 on CIFAR-10 D
(no shrinkage, S5 abs HF 14.5) and 0.89 on CelebA D, against 0.88 / 0.84 with the EMA (decay 0.9999) used everywhere;
but raw weights give a worse gFID (30.9 vs 28.1 on CIFAR-10). The training loss is still falling at 50k steps.

`train`: resumes the existing DiT (runs_p3/<ds>_<tag>[_seed<k>]/dit.pt, 50k steps; the file itself is NOT modified)
and continues training to --steps (default 100k) while tracking THREE EMAs at once (decay 0.999, 0.9995, 0.9999; the
0.9999 one continues the saved EMA, the other two start from the raw weights at 50k). It saves snapshots (raw + three
EMAs) at --snap steps to runs_p3_ema/<ds>_<tag>[_seed<k>]/snap_<step>.pt. CIFAR-10 runs at ~4.9 it/s: 50k extra steps
~2.9 h.
`eval`: for every snapshot x weights (the 50k file gives raw and ema9999), samples 10k latents with Euler 100 steps from
the SAME starting noise, and reports the variance-ratio deciles (bias-free, as in p3_sampler_diag.py), S5 on h and abs
HF, the low-energy share and gFID, and the same after the iso scalar (s = sqrt(tr S_real / tr S_gen) with the mean moved
to the real mean, fitted on these 10k latents). ~5 min per row.
`table`: prints everything collected so far.

Questions it answers (fixed before the run, 13 Nov 2026):
  Q1. Is EMA the cause? -> at a fixed step, ratio and S5 abs HF rise as the EMA gets shorter (0.9999 -> 0.999 -> raw).
  Q2. Is it lag / underfitting? -> at a fixed EMA, ratio rises with training steps (50k -> 100k).
  Q3. Is there a setting that beats "ema9999 + iso" (the current best: CIFAR-10 S5 14.3, gFID 27.9) on BOTH S5 abs HF
      and gFID without any post-hoc map?

Run from D:\\new_vae_research:
  python p3_ema_study.py train --dataset cifar10 --root D:\\new_vae_research\\data --tag D_strong_gan_v2
  python p3_ema_study.py eval  --dataset cifar10 --root D:\\new_vae_research\\data --tag D_strong_gan_v2
  python p3_ema_study.py table --dataset cifar10 --root D:\\new_vae_research\\data --tag D_strong_gan_v2
`maps` (added 15 Nov 2026): l0 vs iso vs null at 50k and 100k (ema9999), ~15 min per step on the GPU; see maps().
  python p3_ema_study.py maps  --dataset cifar10 --root D:\\new_vae_research\\data --tag D_strong_gan_v2
"""
import argparse, copy, json, os, time
import numpy as np
import torch
import torch.nn as nn

from p3_latent_gen import load_vae, load_latents, make_dit, denorm, fid_between, test_images, atomic_save
from p3_sampler_diag import run_sampler, ratio_deciles
from p3_h_check import spectra
from paper2_train import to_01

DECAYS = {"ema999": 0.999, "ema9995": 0.9995, "ema9999": 0.9999}


def sfx(a):
    return f"_seed{a.seed}" if a.seed else ""


def setting(a):
    return f"{a.dataset}_{a.tag}{sfx(a)}"


def base_ns(a):
    b = argparse.Namespace(**vars(a)); b.seed = 0; b.out_root = a.base_root; b.smoke = False
    return b


def train(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    src = os.path.join(a.base_root, setting(a), "dit.pt")
    ck = torch.load(src, map_location=dev, weights_only=False)
    ca = argparse.Namespace(**ck["args"])
    od = os.path.join(a.out, setting(a)); os.makedirs(od, exist_ok=True)
    state_path = os.path.join(od, "state.pt")
    z, _ = load_latents(base_ns(a), ca.latent)
    MU = torch.from_numpy(z["train_mu"].astype(np.float32)).to(dev)
    LV = torch.from_numpy(z["train_lv"].astype(np.float32)).to(dev)
    if a.aug == "flip":                                  # step 2 baseline: horizontally flipped training images,
        fp = os.path.join(od, "latents_flip.npz")        # encoded with the same frozen VAE (doubles the data)
        if not os.path.exists(fp):
            from torch.utils.data import DataLoader
            from p3_latent_gen import datasets, to_pm1
            enc, _, vargs = load_vae(argparse.Namespace(runs=a.runs, dataset=a.dataset, tag=a.tag, which=a.which), dev)
            tr, _ = datasets(base_ns(a), vargs)
            FM, FL = [], []
            with torch.no_grad():
                for x, *_ in DataLoader(tr, 250, shuffle=False, num_workers=0):
                    mu_f, lv_f = enc(to_pm1(torch.flip(x, dims=[3]).to(dev)))
                    FM.append(mu_f.half().cpu()); FL.append(lv_f.half().cpu())
            FM, FL = torch.cat(FM).numpy(), torch.cat(FL).numpy()
            if len(FM) != len(MU):
                raise SystemExit(f"flipped set has {len(FM)} latents, original {len(MU)}: dataset order mismatch")
            np.savez(fp, mu=FM, lv=FL)
            eq = float(np.linalg.norm(FM.astype(np.float32) - z["train_mu"][..., ::-1].astype(np.float32))
                       / np.linalg.norm(z["train_mu"].astype(np.float32)))
            print(f"encoded {len(FM)} flipped training images; relative gap to latent-space flip {eq:.3f}", flush=True)
        f = np.load(fp)
        MU = torch.cat([MU, torch.from_numpy(f["mu"].astype(np.float32)).to(dev)])
        LV = torch.cat([LV, torch.from_numpy(f["lv"].astype(np.float32)).to(dev)])
        print(f"training on {len(MU)} latents (original + flipped)", flush=True)
    m_ = torch.from_numpy(z["ch_mean"]).to(dev)[None, :, None, None]
    s_ = torch.from_numpy(z["ch_std"]).to(dev)[None, :, None, None]
    model = make_dit(ca).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=ca.lr, weight_decay=0.0)
    emas = {k: make_dit(ca).to(dev).requires_grad_(False) for k in DECAYS}
    if os.path.exists(state_path):                       # resume this study
        st = torch.load(state_path, map_location=dev, weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        for k in emas:
            emas[k].load_state_dict(st[k])
        step, log = st["step"], st["log"]
        print(f"resumed the EMA study at step {step}", flush=True)
    else:
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        emas["ema9999"].load_state_dict(ck["ema"])
        emas["ema999"].load_state_dict(ck["model"]); emas["ema9995"].load_state_dict(ck["model"])
        step, log = ck["step"], []
        print(f"starting from {src} at step {step} (the file is not modified)", flush=True)
    amp = bool(getattr(ca, "amp", False)) and dev == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    model_p = [p.detach() for p in model.parameters()]
    ema_p = {k: list(m.parameters()) for k, m in emas.items()}
    snaps = sorted(set(a.snap))
    torch.manual_seed(1000 + step)
    model.train(); t0, acc = time.time(), 0.0
    steps = step + 20 if a.smoke else a.steps
    while step < steps:
        i = torch.randint(0, len(MU), (ca.bs if not a.smoke else 16,), device=dev)
        x0 = MU[i] + (torch.randn_like(MU[i]) * (0.5 * LV[i]).exp() if ca.latent == "sample" else 0)
        x0 = (x0 - m_) / s_
        eps, t = torch.randn_like(x0), torch.rand(len(x0), device=dev)
        xt = (1 - t)[:, None, None, None] * x0 + t[:, None, None, None] * eps
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            loss = ((model(xt, t).float() - (eps - x0)) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update()
        with torch.no_grad():
            for k, d in DECAYS.items():
                torch._foreach_lerp_(ema_p[k], model_p, 1 - d)
        acc += loss.item(); step += 1
        if step % a.log_every == 0 or step == steps:
            rec = {"step": step, "loss": round(acc / a.log_every, 5), "it_s": round(a.log_every / (time.time() - t0), 1)}
            log.append(rec); print(json.dumps(rec), flush=True); acc, t0 = 0.0, time.time()
        if step in snaps or step == steps or step % a.save_every == 0:
            st = {"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "log": log, "args": vars(ca),
                  **{k: m.state_dict() for k, m in emas.items()}}
            atomic_save(st, state_path)
            if step in snaps or step == steps:
                atomic_save({"raw": model.state_dict(), **{k: m.state_dict() for k, m in emas.items()},
                             "step": step, "args": vars(ca)}, os.path.join(od, f"snap_{step}.pt"))
                print(f"saved snapshot at step {step}", flush=True)
    json.dump(log, open(os.path.join(od, "log.json"), "w"), indent=1)


@torch.no_grad()
def evaluate(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    od = os.path.join(a.out, setting(a)); os.makedirs(od, exist_ok=True)
    ck0 = torch.load(os.path.join(a.base_root, setting(a), "dit.pt"), map_location=dev, weights_only=False)
    ca = argparse.Namespace(**ck0["args"])
    z, norm = load_latents(base_ns(a), ca.latent)
    _, dec, vargs = load_vae(argparse.Namespace(runs=a.runs, dataset=a.dataset, tag=a.tag, which=a.which), dev)
    model = make_dit(ca).to(dev).eval()
    amp = bool(getattr(ca, "amp", False))
    rng = np.random.default_rng(0)
    mu, lv = z["train_mu"].astype(np.float32), z["train_lv"].astype(np.float32)
    R = norm(mu + np.exp(0.5 * lv) * rng.standard_normal(mu.shape).astype(np.float32)).flatten(1).double().numpy()
    mr, tr_r = R.mean(0), np.trace(np.cov(R, rowvar=False))
    real = test_images(base_ns(a), vargs)
    Er, Hr, hr = spectra(real)
    ch, ce, efl = np.quantile(hr, [.2, .4, .6, .8]), np.quantile(Hr, [.2, .4, .6, .8]), np.quantile(Er, 0.05)

    def measure(zn):
        fake = torch.cat([to_01(dec(denorm(zn[s:s + 250].to(dev), z, dev))).clamp(0, 1).cpu() for s in range(0, len(zn), 250)])
        E, H, h = spectra(fake)
        qh, qe = np.digitize(h, ch), np.digitize(H, ce)
        return {"S5_h": float(100 * (qh == 4).mean()), "S5_absHF": float(100 * (qe == 4).mean()),
                "lowE45": float(100 * (E < efl)[qh >= 3].mean()) if (qh >= 3).any() else float("nan"),
                "gfid": fid_between(real, fake, dev)}

    jobs = [(ck0["step"], "raw", ck0["model"]), (ck0["step"], "ema9999", ck0["ema"])]
    for f in sorted(os.listdir(od)):
        if f.startswith("snap_") and f.endswith(".pt"):
            sn = torch.load(os.path.join(od, f), map_location=dev, weights_only=False)
            jobs += [(sn["step"], k, sn[k]) for k in ("raw", "ema999", "ema9995", "ema9999")]
    res_path = os.path.join(od, "eval.json")
    res = json.load(open(res_path)) if os.path.exists(res_path) else {}
    for step, w, sd in jobs:
        key = f"{step}_{w}"
        if key in res and not a.redo:
            continue
        model.load_state_dict(sd)
        zn = run_sampler(model, "euler100", 200 if a.smoke else a.n, dev, 777 + 1000003 * a.seed, 1.0, amp, a.bs)
        G = zn.flatten(1).double().numpy()
        ratios, _, _ = ratio_deciles(R, G, np.random.default_rng(0))
        r = {"step": step, "weights": w, "ratio_deciles": ratios, "ratio_mean": float(np.mean(ratios)), **measure(zn)}
        s = float(np.sqrt(tr_r / np.trace(np.cov(G, rowvar=False))))
        Gi = mr + s * (G - G.mean(0))
        iso = measure(torch.from_numpy(Gi.astype(np.float32)).view_as(zn))
        r.update({"iso_scale": s, **{f"iso_{k}": v for k, v in iso.items()}})
        res[key] = r
        json.dump(res, open(res_path, "w"), indent=1)
        print(f"{key}: ratio {r['ratio_mean']:.3f}, S5 absHF {r['S5_absHF']:.1f} (iso {r['iso_S5_absHF']:.1f}), "
              f"gFID {r['gfid']:.2f} (iso {r['iso_gfid']:.2f})", flush=True)
    table(a)


def _sqrtm_psd(C):
    w, V = np.linalg.eigh((C + C.T) / 2)
    return (V * np.sqrt(np.clip(w, 0, None))) @ V.T


def _ot(src, dst):
    """Closed-form Gaussian OT map src -> dst (same formula as p3_latent_transport.py)."""
    ms, md = src.mean(0), dst.mean(0)
    Cs, Cd = np.cov(src, rowvar=False), np.cov(dst, rowvar=False)
    Ss = _sqrtm_psd(Cs)
    w, V = np.linalg.eigh(Ss)
    Ssi = (V / np.clip(w, 1e-8, None)) @ V.T
    return ms, md, Ssi @ _sqrtm_psd(Ss @ Cd @ Ss) @ Ssi, Cs, Cd


@torch.no_grad()
def maps(a):
    """L0 (full-covariance Gaussian OT) vs iso (one scalar) vs null, fitted on 50k generated FIT latents and applied
    to 10k generated EVAL latents (different noise, the same eval noise as `eval`, so `base` must reproduce eval.json).

    Question (fixed 15 Nov 2026, BEFORE this run): the EMA study showed that with longer training the shrinkage
    becomes more direction-dependent (100k ema9999: largest directions 0.89, smallest 0.74; at 50k 0.92 / 0.84) and
    the iso fix recovers less (iso S5 abs HF 14.1 -> 12.9). If the remaining loss is direction-dependent, a map that
    uses directions should gain more over iso at 100k than at 50k. Direction-dependence counts as SHOWN if
      1. at 100k: S5 abs HF(l0) - S5 abs HF(iso) >= 1.0
      2. that gap at 100k is at least 0.5 larger than the same gap at 50k
      3. at 100k: LowE45(l0) <= LowE45(iso)
      4. the null control moves S5 abs HF by less than 1.0 at both steps
    """
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    od = os.path.join(a.out, setting(a)); os.makedirs(od, exist_ok=True)
    ck0 = torch.load(os.path.join(a.base_root, setting(a), "dit.pt"), map_location=dev, weights_only=False)
    ca = argparse.Namespace(**ck0["args"])
    z, norm = load_latents(base_ns(a), ca.latent)
    _, dec, vargs = load_vae(argparse.Namespace(runs=a.runs, dataset=a.dataset, tag=a.tag, which=a.which), dev)
    model = make_dit(ca).to(dev).eval()
    amp = bool(getattr(ca, "amp", False))
    rng = np.random.default_rng(0)
    mu, lv = z["train_mu"].astype(np.float32), z["train_lv"].astype(np.float32)
    R = norm(mu + np.exp(0.5 * lv) * rng.standard_normal(mu.shape).astype(np.float32)).flatten(1).double().numpy()
    perm = rng.permutation(len(R)); h2 = len(R) // 2
    m1, m2, A0, _, _ = _ot(R[perm[:h2]], R[perm[h2:]])
    real = test_images(base_ns(a), vargs)
    Er, Hr, hr = spectra(real)
    ch, ce, efl = np.quantile(hr, [.2, .4, .6, .8]), np.quantile(Hr, [.2, .4, .6, .8]), np.quantile(Er, 0.05)

    def measure(Gf, shape):
        zn = torch.from_numpy(Gf.astype(np.float32)).view(shape)
        fake = torch.cat([to_01(dec(denorm(zn[s:s + 250].to(dev), z, dev))).clamp(0, 1).cpu() for s in range(0, len(zn), 250)])
        E, H, hh = spectra(fake)
        qh, qe = np.digitize(hh, ch), np.digitize(H, ce)
        return {"S5_h": float(100 * (qh == 4).mean()), "S5_absHF": float(100 * (qe == 4).mean()),
                "lowE45": float(100 * (E < efl)[qh >= 3].mean()) if (qh >= 3).any() else float("nan"),
                "gfid": fid_between(real, fake, dev)}

    res_path = os.path.join(od, f"maps_{a.weights}.json")
    res = json.load(open(res_path)) if os.path.exists(res_path) else {}
    n_fit, n_ev = (300, 200) if a.smoke else (a.n_fit, a.n)
    for step in a.map_steps:
        key = str(step)
        if key in res and not a.redo:
            continue
        if step == ck0["step"]:
            sd = ck0["model"] if a.weights == "raw" else ck0["ema"]
            if a.weights not in ("raw", "ema9999"):
                raise SystemExit(f"the {step} file only has raw and ema9999 weights")
        else:
            p = os.path.join(od, f"snap_{step}.pt")
            if not os.path.exists(p):
                raise SystemExit(f"missing {p}: run `train` first")
            sd = torch.load(p, map_location=dev, weights_only=False)[a.weights]
        model.load_state_dict(sd)
        t0 = time.time()
        Zf = run_sampler(model, "euler100", n_fit, dev, 424242 + 1000003 * a.seed, 1.0, amp, a.bs)
        Ze = run_sampler(model, "euler100", n_ev, dev, 777 + 1000003 * a.seed, 1.0, amp, a.bs)
        F, G = Zf.flatten(1).double().numpy(), Ze.flatten(1).double().numpy()
        mg, mr, A, Cg, Cr = _ot(F, R)
        s = float(np.sqrt(np.trace(Cr) / np.trace(Cg)))
        arms = {"base": G, "iso": mr + s * (G - mg), "l0": (G - mg) @ A.T + mr, "null": (G - m1) @ A0.T + m2}
        r = {"step": step, "weights": a.weights, "iso_scale": s, "n_fit": n_fit, "n_eval": n_ev}
        for k, Gk in arms.items():
            r[k] = measure(Gk, Ze.shape)
        res[key] = r
        json.dump(res, open(res_path, "w"), indent=1)
        print(f"step {step} {a.weights}: base {r['base']['S5_absHF']:.1f}, iso {r['iso']['S5_absHF']:.1f}, "
              f"l0 {r['l0']['S5_absHF']:.1f}, null {r['null']['S5_absHF']:.1f}  ({(time.time() - t0) / 60:.1f} min)", flush=True)
    maps_table(a)


def maps_table(a):
    p = os.path.join(a.out, setting(a), f"maps_{a.weights}.json")
    if not os.path.exists(p):
        print("no maps json yet"); return
    res = json.load(open(p))
    print(f"\n{setting(a)}  weights {a.weights}: maps fitted on {next(iter(res.values()))['n_fit']} generated latents, "
          f"applied to {next(iter(res.values()))['n_eval']} others")
    print(f"{'step':>6s} {'arm':>5s} {'S5 h':>5s} {'S5 aHF':>7s} {'lowE45':>7s} {'gFID':>6s}")
    for r in sorted(res.values(), key=lambda r: r["step"]):
        for k in ("base", "iso", "l0", "null"):
            m = r[k]
            print(f"{r['step']:6d} {k:>5s} {m['S5_h']:5.1f} {m['S5_absHF']:7.1f} {m['lowE45']:7.1f} {m['gfid']:6.2f}"
                  + (f"   (s = {r['iso_scale']:.3f})" if k == "iso" else ""))
    steps = sorted(int(k) for k in res)
    if len(steps) < 2:
        print("need two steps (e.g. 50000 100000) for the test"); return
    lo, hi = res[str(steps[0])], res[str(steps[-1])]
    gap = lambda r: r["l0"]["S5_absHF"] - r["iso"]["S5_absHF"]
    checks = {f"at {steps[-1]}: l0 - iso on S5 abs HF >= 1.0 ({gap(hi):+.1f})": gap(hi) >= 1.0,
              f"gap at {steps[-1]} >= gap at {steps[0]} + 0.5 ({gap(hi):+.1f} vs {gap(lo):+.1f})": gap(hi) >= gap(lo) + 0.5,
              f"at {steps[-1]}: LowE45 l0 <= iso ({hi['l0']['lowE45']:.1f} vs {hi['iso']['lowE45']:.1f})":
                  hi["l0"]["lowE45"] <= hi["iso"]["lowE45"],
              "null moves S5 abs HF by < 1.0 at both steps ("
              + ", ".join(f"{r['null']['S5_absHF'] - r['base']['S5_absHF']:+.1f}" for r in (lo, hi)) + ")":
                  all(abs(r["null"]["S5_absHF"] - r["base"]["S5_absHF"]) < 1.0 for r in (lo, hi))}
    print("\ndirection-dependence test (fixed 15 Nov 2026, before this run):")
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'fail'}  {k}")
    print("  => " + ("SHOWN" if all(checks.values()) else "NOT SHOWN"))


GATE0_TABLE = {"cifar10": (0.058, 0.151), "celeba": (0.044, 0.214)}   # observed -ln(sd ratio) at 100k, dec1 / dec10


@torch.no_grad()
def gainmap(a):
    """Gate 0 + restart test + tails (Fable plan, fixed 17 Nov 2026 BEFORE the run).

    Real latents: train posterior samples split in thirds -> eigenbasis U (third A), real variances lam (third B);
    held-out TEST posterior samples for every model evaluation (gain map, loss map, restart test).
    1. Gain map: for each snapshot (ema weights) and 40 t-values, a_hat_k(t) = Cov((U'v_theta)_k, (U'x_t)_k) / Var((U'x_t)_k)
       vs the exact target a_k(t) = (t - (1-t) lam_k) / ((1-t)^2 lam_k + t^2) (also measured with the TRUE v as a check).
       Predicted ln(sd_gen / sd_real)_k = -int_0^1 (a_hat_k - a_k) dt, split into 5 t-bands.
    2. Loss map: held-out loss E[((U'(v_theta - v))_k)^2] per decile and t-band (H2: does the t > 0.8 loss rise?).
    3. Restart test (50k and 100k): start from REAL held-out latents noised to t0 in {1.0 (pure noise), 0.9, 0.6, 0.3, 0.1}
       and integrate the same 100-step Euler grid to t = 0; variance ratio per decile shows where the deficit appears.
    4. Tails (H3) and posterior-noise share (H1) per decile at 100k.
    Gate 0 (Fable): PASS if at 100k the predicted dec10 deficit is >= half of the observed one in GATE0_TABLE.
    """
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    od = os.path.join(a.out, setting(a)); os.makedirs(od, exist_ok=True)
    ck0 = torch.load(os.path.join(a.base_root, setting(a), "dit.pt"), map_location=dev, weights_only=False)
    ca = argparse.Namespace(**ck0["args"])
    z, norm = load_latents(base_ns(a), ca.latent)
    model = make_dit(ca).to(dev).eval()
    amp = bool(getattr(ca, "amp", False)) and dev == "cuda"
    rng = np.random.default_rng(0)
    mu, lv = z["train_mu"].astype(np.float32), z["train_lv"].astype(np.float32)
    R = norm(mu + np.exp(0.5 * lv) * rng.standard_normal(mu.shape).astype(np.float32)).flatten(1).double().numpy()
    perm = rng.permutation(len(R)); n3 = len(R) // 3
    RA, RB = R[perm[:n3]], R[perm[n3:2 * n3]]
    w, V = np.linalg.eigh(np.cov(RA, rowvar=False))
    U = V[:, np.argsort(w)[::-1]]
    mB = RB.mean(0)
    PB = (RB - mB) @ U
    lam = PB.var(0, ddof=1)
    idx = np.array_split(np.arange(U.shape[1]), 10)
    dec = lambda v: [float(np.mean(v[i])) for i in idx]
    tmu, tlv = z["test_mu"].astype(np.float32), z["test_lv"].astype(np.float32)
    n_te = 600 if a.smoke else min(len(tmu), a.n)
    T = norm(tmu[:n_te] + np.exp(0.5 * tlv[:n_te]) * np.random.default_rng(1).standard_normal(tmu[:n_te].shape)
             .astype(np.float32))                                             # held-out real latents, (n,4,8,8)
    Ut = torch.from_numpy(U).float().to(dev); mt = torch.from_numpy(mB).float().to(dev)
    lam_t = torch.from_numpy(lam).double()

    # H1 helper: posterior-noise variance along each direction (normalised units) and its share of real variance
    s_ = z["ch_std"].astype(np.float64)[None, :, None, None]
    nv = (np.exp(lv[:20000].astype(np.float64)) / s_ ** 2).mean(0).reshape(-1)  # diagonal noise variance per coordinate
    noise_share = (U ** 2).T @ nv / lam

    def v_of(x, t):
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            return model(x, torch.full((len(x),), float(t), device=dev)).float()

    def load(step):
        if step == ck0["step"]:
            return ck0["ema"] if a.weights == "ema9999" else ck0["model"]
        p = os.path.join(od, f"snap_{step}.pt")
        if not os.path.exists(p):
            raise SystemExit(f"missing {p}: run `train` first")
        return torch.load(p, map_location=dev, weights_only=False)[a.weights]

    n_t = 8 if a.smoke else a.n_t
    ts = (np.arange(n_t) + 0.5) / n_t                                     # midpoint rule on [0, 1]
    bands = np.digitize(ts, [0.2, 0.4, 0.6, 0.8])
    res_path = os.path.join(od, f"gainmap_{a.weights}.json")
    res = json.load(open(res_path)) if os.path.exists(res_path) else {}
    gen = torch.Generator(device=dev).manual_seed(2026)
    for step in a.map_steps:
        key = str(step)
        if key in res and not a.redo:
            continue
        model.load_state_dict(load(step)); t0c = time.time()
        ahat, atrue, loss = np.zeros((n_t, 256)), np.zeros((n_t, 256)), np.zeros((n_t, 256))
        for j, t in enumerate(ts):
            S = {k: torch.zeros(256, dtype=torch.float64) for k in ("x", "xx", "v", "vx", "u", "ux", "e2")}
            for s in range(0, n_te, a.bs):
                x0 = T[s:s + a.bs].to(dev)
                eps = torch.randn(x0.shape, device=dev, generator=gen)
                xt = (1 - t) * x0 + t * eps
                vt = v_of(xt, t)
                X = ((xt.flatten(1) - (1 - t) * mt) @ Ut).double().cpu()
                Vm = (vt.flatten(1) @ Ut).double().cpu()
                Vu = ((eps - x0).flatten(1) @ Ut).double().cpu()
                S["x"] += X.sum(0); S["xx"] += (X * X).sum(0); S["v"] += Vm.sum(0); S["vx"] += (Vm * X).sum(0)
                S["u"] += Vu.sum(0); S["ux"] += (Vu * X).sum(0); S["e2"] += ((Vm - Vu) ** 2).sum(0)
            n = n_te
            varx = S["xx"] / n - (S["x"] / n) ** 2
            ahat[j] = ((S["vx"] / n - S["v"] / n * S["x"] / n) / varx).numpy()
            atrue[j] = ((S["ux"] / n - S["u"] / n * S["x"] / n) / varx).numpy()
            loss[j] = (S["e2"] / n).numpy()
        a_exact = (ts[:, None] - (1 - ts[:, None]) * lam[None]) / ((1 - ts[:, None]) ** 2 * lam[None] + ts[:, None] ** 2)
        d = (ahat - a_exact) / n_t                                          # per-t contribution to the integral
        pred = -d.sum(0)                                                    # predicted ln(sd_gen / sd_real)
        tstar = np.sqrt(lam) / (1 + np.sqrt(lam))
        low = -(d * (ts[:, None] < tstar[None])).sum(0)                     # part of pred from t < t*_k
        r = {"step": step, "pred_ln_sd": dec(pred), "pred_low_t": dec(low), "pred_t_gt_0.8": dec(-(d * (ts[:, None] > 0.8)).sum(0)),
             "pred_band": [dec(-d[bands == b].sum(0)) for b in range(5)],
             "check_true_minus_exact": float(np.abs(atrue - a_exact).mean()),
             "loss_band": [dec(loss[bands == b].mean(0)) for b in range(5)], "pred_dir": pred.tolist()}
        if step in a.restart_steps:
            ratios = {}
            for t0 in (1.0, 0.9, 0.6, 0.3, 0.1):
                grid = [g for g in np.linspace(1, 0, 101) if g <= t0 + 1e-9]
                if t0 == 1.0:
                    grid = list(np.linspace(1, 0, 101))
                out = []
                for s in range(0, n_te, a.bs):
                    x0 = T[s:s + a.bs].to(dev)
                    x = (1 - grid[0]) * x0 + grid[0] * torch.randn(x0.shape, device=dev, generator=gen)
                    for k in range(len(grid) - 1):
                        x = x + (grid[k + 1] - grid[k]) * v_of(x, grid[k])
                    out.append(x.flatten(1).double().cpu())
                G = torch.cat(out).numpy()
                P = (G - G.mean(0)) @ U
                rr = P.var(0, ddof=1) / lam
                ratios[str(t0)] = dec(rr)
                if t0 == 1.0:
                    k_real = ((PB / PB.std(0)) ** 4).mean(0) - 3
                    Pz = P / P.std(0)
                    k_gen = (Pz ** 4).mean(0) - 3
                    q = lambda M, lo, hi: np.quantile(M, hi, 0) - np.quantile(M, lo, 0)
                    from scipy.stats import spearmanr
                    lnr = 0.5 * np.log(rr)
                    r["obs_ln_sd"] = dec(lnr)
                    r["tails"] = {"kurt_real": dec(k_real), "kurt_gen": dec(k_gen),
                                  "iqr_ratio": dec(q(P, .25, .75) / q(PB, .25, .75)),
                                  "tail_ratio": dec(q(P, .01, .99) / q(PB, .01, .99)),
                                  "spearman_lnratio_kurt": float(spearmanr(lnr, k_real)[0]),
                                  "spearman_lnratio_noise_share": float(spearmanr(lnr, noise_share)[0]),
                                  "spearman_lnratio_pred": float(spearmanr(lnr, pred)[0])}
            r["restart_ratio"] = ratios
        res[key] = r
        json.dump(res, open(res_path, "w"), indent=1)
        print(f"step {step}: predicted ln sd ratio dec1 {r['pred_ln_sd'][0]:+.3f}, dec10 {r['pred_ln_sd'][-1]:+.3f}"
              f"  ({(time.time() - t0c) / 60:.1f} min)", flush=True)
    json.dump({"lam_dec": dec(lam), "noise_share_dec": dec(noise_share)}, open(os.path.join(od, "gainmap_meta.json"), "w"))
    gainmap_table(a)


@torch.no_grad()
def energy(a):
    """Exact energy-rate decomposition (Fable, 17 Nov 2026), logged during normal 100-step Euler sampling.

    Per eigen-direction k, an Euler step x <- x + h v (h = -0.01) changes the variance exactly by
        dVar_k = 2 h Cov_gen(x_k, v_k) + h^2 Var_gen(v_k).
    The exact flow needs Cov(x_k, v_k) = c_k(t) = t - (1 - t) lam_k. The excess r_k = Cov_gen - c_k splits into
        gain     = Cov_real(x_k, v_theta,k) - c_k      (network error on REAL noised states; outcomes A/B)
        exposure = Cov_gen(x_k, v_k) - Cov_real(...)    (generated states differ from real ones; outcomes C/D)
    Contributions to (Var_gen / lam - 1) at t = 0 are reported per decile and per t-band; with the Euler h^2 term,
    the exact-path discretisation and the initial-noise term the sum must reproduce the observed ratio (check column).
    Rule fixed 17 Nov 2026 BEFORE the run: at the last step, dec10 is "gain-dominated" if |gain| >= 0.5 |gain + exposure|
    (then A if most of it lies at t < 0.4, else B); otherwise "exposure-dominated" (C if most of it lies at t >= 0.6,
    D if most lies at t < 0.4, else mixed).
    """
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    od = os.path.join(a.out, setting(a)); os.makedirs(od, exist_ok=True)
    ck0 = torch.load(os.path.join(a.base_root, setting(a), "dit.pt"), map_location=dev, weights_only=False)
    ca = argparse.Namespace(**ck0["args"])
    z, norm = load_latents(base_ns(a), ca.latent)
    model = make_dit(ca).to(dev).eval()
    amp = bool(getattr(ca, "amp", False)) and dev == "cuda"
    rng = np.random.default_rng(0)
    mu, lv = z["train_mu"].astype(np.float32), z["train_lv"].astype(np.float32)
    R = norm(mu + np.exp(0.5 * lv) * rng.standard_normal(mu.shape).astype(np.float32)).flatten(1).double().numpy()
    perm = rng.permutation(len(R)); n3 = len(R) // 3
    RA, RB = R[perm[:n3]], R[perm[n3:2 * n3]]
    w, V = np.linalg.eigh(np.cov(RA, rowvar=False))
    U = V[:, np.argsort(w)[::-1]]
    lam = ((RB - RB.mean(0)) @ U).var(0, ddof=1)
    idx = np.array_split(np.arange(U.shape[1]), 10)
    dec = lambda v: [float(np.mean(v[i])) for i in idx]
    tmu, tlv = z["test_mu"].astype(np.float32), z["test_lv"].astype(np.float32)
    n_te = 600 if a.smoke else min(len(tmu), a.n)
    T = norm(tmu[:n_te] + np.exp(0.5 * tlv[:n_te]) * np.random.default_rng(1).standard_normal(tmu[:n_te].shape)
             .astype(np.float32))
    Ut = torch.from_numpy(U).float().to(dev)

    def v_of(x, t):
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            return model(x, torch.full((len(x),), float(t), device=dev)).float()

    def cov_stats():
        return {k: torch.zeros(256, dtype=torch.float64) for k in ("x", "v", "xv", "vv")}

    def add(S, X, Vv):
        S["x"] += X.sum(0); S["v"] += Vv.sum(0); S["xv"] += (X * Vv).sum(0); S["vv"] += (Vv * Vv).sum(0)

    def cov(S, n):
        return ((S["xv"] / n - S["x"] / n * S["v"] / n).numpy(), (S["vv"] / n - (S["v"] / n) ** 2).numpy())

    grid = np.linspace(1, 0, 101)
    nst = len(grid) - 1
    res_path = os.path.join(od, f"energy_{a.weights}.json")
    res = json.load(open(res_path)) if os.path.exists(res_path) else {}
    for step in a.restart_steps:
        key = str(step)
        if key in res and not a.redo:
            continue
        if step == ck0["step"]:
            sd = ck0["ema"] if a.weights == "ema9999" else ck0["model"]
        else:
            p = os.path.join(od, f"snap_{step}.pt")
            if not os.path.exists(p):
                raise SystemExit(f"missing {p}: run `train` first")
            sd = torch.load(p, map_location=dev, weights_only=False)[a.weights]
        model.load_state_dict(sd); t0c = time.time()
        gen = torch.Generator(device=dev).manual_seed(777 + 1000003 * a.seed)
        Sg = [cov_stats() for _ in range(nst)]; Sr = [cov_stats() for _ in range(nst)]
        X0s, Xends = [], []
        for s in range(0, n_te, a.bs):                                   # generated trajectories
            x = torch.randn((min(a.bs, n_te - s), 4, 8, 8), device=dev, generator=gen)
            X0s.append((x.flatten(1) @ Ut).double().cpu())
            for k in range(nst):
                vt = v_of(x, grid[k])
                add(Sg[k], (x.flatten(1) @ Ut).double().cpu(), (vt.flatten(1) @ Ut).double().cpu())
                x = x + (grid[k + 1] - grid[k]) * vt
            Xends.append((x.flatten(1) @ Ut).double().cpu())
        for k in range(nst):                                              # real noised states at the same t
            t = grid[k]
            for s in range(0, n_te, a.bs):
                x0 = T[s:s + a.bs].to(dev)
                xt = (1 - t) * x0 + t * torch.randn(x0.shape, device=dev, generator=gen)
                add(Sr[k], (xt.flatten(1) @ Ut).double().cpu(), (v_of(xt, t).flatten(1) @ Ut).double().cpu())
        h = grid[1:] - grid[:-1]                                          # negative steps
        c = grid[:-1, None] - (1 - grid[:-1, None]) * lam[None]           # exact Cov(x, v) on the true path
        cg = np.stack([cov(S, n_te)[0] for S in Sg]); vg = np.stack([cov(S, n_te)[1] for S in Sg])
        cr = np.stack([cov(S, n_te)[0] for S in Sr])
        gain_c = 2 * h[:, None] * (cr - c) / lam                           # per-step contribution to Var/lam - 1
        expo_c = 2 * h[:, None] * (cg - cr) / lam
        euler_c = (h[:, None] ** 2) * vg / lam
        path_c = (2 * h[:, None] * c).sum(0) / lam - (lam - 1) / lam        # discretisation of the exact path
        X0 = torch.cat(X0s).numpy(); Xe = torch.cat(Xends).numpy()
        init_c = (X0.var(0, ddof=0) - 1) / lam
        obs = Xe.var(0, ddof=0) / lam - 1
        bands = np.digitize(grid[:-1], [0.2, 0.4, 0.6, 0.8 + 1e-9])        # band of the step's start time
        r = {"step": step, "obs": dec(obs), "gain": dec(gain_c.sum(0)), "exposure": dec(expo_c.sum(0)),
             "euler": dec(euler_c.sum(0)), "path": dec(path_c), "init": dec(init_c),
             "check": dec(obs - (gain_c.sum(0) + expo_c.sum(0) + euler_c.sum(0) + path_c + init_c)),
             "gain_band": [dec(gain_c[bands == b].sum(0)) for b in range(5)],
             "exposure_band": [dec(expo_c[bands == b].sum(0)) for b in range(5)]}
        res[key] = r
        json.dump(res, open(res_path, "w"), indent=1)
        print(f"step {step}: dec10 observed {r['obs'][-1]:+.3f} = gain {r['gain'][-1]:+.3f} + exposure "
              f"{r['exposure'][-1]:+.3f} + rest  ({(time.time() - t0c) / 60:.1f} min)", flush=True)
    energy_table(a)


def energy_table(a):
    p = os.path.join(a.out, setting(a), f"energy_{a.weights}.json")
    if not os.path.exists(p):
        print("no energy json yet"); return
    res = json.load(open(p))
    f = lambda v: " ".join(f"{x:+.3f}" for x in v)
    print(f"\n{setting(a)}  weights {a.weights}: contributions to (Var_gen / Var_real - 1) per decile (dec1 ... dec10)")
    for r in sorted(res.values(), key=lambda r: r["step"]):
        print(f"\n  step {r['step']}")
        for k in ("obs", "gain", "exposure", "euler", "path", "init", "check"):
            print(f"   {k:9s} {f(r[k])}")
        print("   dec10 by t-band [0-.2 .2-.4 .4-.6 .6-.8 .8-1]:  gain " + f([b[-1] for b in r["gain_band"]])
              + " | exposure " + f([b[-1] for b in r["exposure_band"]]))
    r = max(res.values(), key=lambda r: r["step"])
    g, e = r["gain"][-1], r["exposure"][-1]
    gb = [b[-1] for b in r["gain_band"]]; eb = [b[-1] for b in r["exposure_band"]]
    share = lambda b, lo, hi: sum(b[lo:hi]) / sum(b) if abs(sum(b)) > 1e-9 else float("nan")
    if abs(g) >= 0.5 * abs(g + e):
        out = "A (gain error, low t)" if share(gb, 0, 2) >= 0.5 else "B (gain error, not low t)"
    else:
        out = ("C (exposure, t >= 0.6)" if share(eb, 3, 5) >= 0.5 else
               "D (exposure, t < 0.4)" if share(eb, 0, 2) >= 0.5 else "mixed exposure")
    print(f"\nRULE (fixed 17 Nov 2026) at step {r['step']}, dec10: gain {g:+.3f}, exposure {e:+.3f} -> outcome {out}")


@torch.no_grad()
def overfit(a):
    """Is the low-t gain error a generalisation gap? (added 17 Nov 2026 after gainmap/energy gave outcome A)

    The held-out loss at t < 0.4 RISES from 50k to 100k (train loss falls). If the network is fitting the training
    latents at low noise, its gain is right on TRAIN states and too contractive on unseen states (held-out data and,
    equally, newly generated latents). For 50k and 100k: gain map + loss map on train posterior samples (the third not
    used for U / lam) vs held-out test samples; and a memorisation check: nearest-neighbour ratio d1/d2 of generated
    latents to the training set (memorised if d1/d2 < 1/3) and the gen->train vs test->train NN distance.
    Rule fixed 17 Nov 2026 BEFORE the run: "generalisation gap" if at 100k the dec10 predicted deficit on TRAIN states
    is <= half of that on TEST states. Then the M1 penalty computed on training batches cannot see the error.
    """
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    od = os.path.join(a.out, setting(a)); os.makedirs(od, exist_ok=True)
    ck0 = torch.load(os.path.join(a.base_root, setting(a), "dit.pt"), map_location=dev, weights_only=False)
    ca = argparse.Namespace(**ck0["args"])
    z, norm = load_latents(base_ns(a), ca.latent)
    model = make_dit(ca).to(dev).eval()
    amp = bool(getattr(ca, "amp", False)) and dev == "cuda"
    rng = np.random.default_rng(0)
    mu, lv = z["train_mu"].astype(np.float32), z["train_lv"].astype(np.float32)
    R = norm(mu + np.exp(0.5 * lv) * rng.standard_normal(mu.shape).astype(np.float32)).flatten(1).double().numpy()
    perm = rng.permutation(len(R)); n3 = len(R) // 3
    RA, RB, RC = R[perm[:n3]], R[perm[n3:2 * n3]], R[perm[2 * n3:3 * n3]]
    w, V = np.linalg.eigh(np.cov(RA, rowvar=False))
    U = V[:, np.argsort(w)[::-1]]
    lam = ((RB - RB.mean(0)) @ U).var(0, ddof=1)
    idx = np.array_split(np.arange(U.shape[1]), 10)
    dec = lambda v: [float(np.mean(v[i])) for i in idx]
    n_te = 600 if a.smoke else min(a.n, len(z["test_mu"]), len(RC))
    tmu, tlv = z["test_mu"][:n_te].astype(np.float32), z["test_lv"][:n_te].astype(np.float32)
    sets = {"train": torch.from_numpy(RC[:n_te].astype(np.float32)).view(-1, 4, 8, 8),
            "test": norm(tmu + np.exp(0.5 * tlv) * np.random.default_rng(1).standard_normal(tmu.shape).astype(np.float32))}
    Ut = torch.from_numpy(U).float().to(dev)
    Rtr = torch.from_numpy(R).float().to(dev)                              # full training set for nearest neighbours

    def v_of(x, t):
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            return model(x, torch.full((len(x),), float(t), device=dev)).float()

    def nn_ratio(Q):
        d1, d2 = [], []
        for s in range(0, len(Q), 500):
            D = torch.cdist(Q[s:s + 500], Rtr)
            v2 = D.topk(2, largest=False).values
            d1.append(v2[:, 0].cpu()); d2.append(v2[:, 1].cpu())
        d1, d2 = torch.cat(d1).numpy(), torch.cat(d2).numpy()
        return d1, d1 / np.maximum(d2, 1e-12)

    n_t = 8 if a.smoke else a.n_t
    ts = (np.arange(n_t) + 0.5) / n_t
    bands = np.digitize(ts, [0.2, 0.4, 0.6, 0.8])
    a_exact = (ts[:, None] - (1 - ts[:, None]) * lam[None]) / ((1 - ts[:, None]) ** 2 * lam[None] + ts[:, None] ** 2)
    res_path = os.path.join(od, f"overfit_{a.weights}.json")
    res = json.load(open(res_path)) if os.path.exists(res_path) else {}
    gen = torch.Generator(device=dev).manual_seed(4242)
    for step in a.restart_steps:
        key = str(step)
        if key in res and not a.redo:
            continue
        if step == ck0["step"]:
            sd = ck0["ema"] if a.weights == "ema9999" else ck0["model"]
        else:
            p = os.path.join(od, f"snap_{step}.pt")
            if not os.path.exists(p):
                raise SystemExit(f"missing {p}: run `train` first")
            sd = torch.load(p, map_location=dev, weights_only=False)[a.weights]
        model.load_state_dict(sd); t0c = time.time()
        r = {"step": step}
        for name, Xs in sets.items():
            ahat, loss = np.zeros((n_t, 256)), np.zeros((n_t, 256))
            for j, t in enumerate(ts):
                S = {k: torch.zeros(256, dtype=torch.float64) for k in ("x", "xx", "v", "vx", "e2")}
                for s in range(0, n_te, a.bs):
                    x0 = Xs[s:s + a.bs].to(dev)
                    eps = torch.randn(x0.shape, device=dev, generator=gen)
                    xt = (1 - t) * x0 + t * eps
                    vt = v_of(xt, t)
                    X = (xt.flatten(1) @ Ut).double().cpu(); Vm = (vt.flatten(1) @ Ut).double().cpu()
                    Vu = ((eps - x0).flatten(1) @ Ut).double().cpu()
                    S["x"] += X.sum(0); S["xx"] += (X * X).sum(0); S["v"] += Vm.sum(0); S["vx"] += (Vm * X).sum(0)
                    S["e2"] += ((Vm - Vu) ** 2).sum(0)
                varx = S["xx"] / n_te - (S["x"] / n_te) ** 2
                ahat[j] = ((S["vx"] / n_te - S["v"] / n_te * S["x"] / n_te) / varx).numpy()
                loss[j] = (S["e2"] / n_te).numpy()
            pred = -((ahat - a_exact) / n_t).sum(0)
            r[name] = {"pred_ln_sd": dec(pred), "loss_band": [dec(loss[bands == b].mean(0)) for b in range(5)]}
        G = run_sampler(model, "euler100", n_te, dev, 777 + 1000003 * a.seed, 1.0, amp, a.bs).flatten(1).to(dev)
        dg, rg = nn_ratio(G)
        dt_, rt = nn_ratio(sets["test"].flatten(1).to(dev))
        r["nn"] = {"gen_d1_median": float(np.median(dg)), "test_d1_median": float(np.median(dt_)),
                   "gen_memorised_pct": float(100 * (rg < 1 / 3).mean()), "test_memorised_pct": float(100 * (rt < 1 / 3).mean()),
                   "gen_ratio_median": float(np.median(rg)), "test_ratio_median": float(np.median(rt))}
        res[key] = r
        json.dump(res, open(res_path, "w"), indent=1)
        print(f"step {step}: dec10 predicted deficit train {r['train']['pred_ln_sd'][-1]:+.3f} vs test "
              f"{r['test']['pred_ln_sd'][-1]:+.3f}  ({(time.time() - t0c) / 60:.1f} min)", flush=True)
    overfit_table(a)


GATE1_ISO = {"cifar10": (12.9, 14.5), "celeba": (10.7, 15.5)}   # iso at 100k from `maps`: S5 abs HF, LowE45
REF_L0 = {"cifar10": (15.0, 12.5, 26.90), "celeba": (11.8, 13.8, 12.58)}


@torch.no_grad()
def calib(a):
    """Step 1 (17 Nov 2026): training-free HELD-OUT gain calibration of the velocity field.

    `overfit` showed the gain is right on training states and too contractive on unseen states (low t). Here the gain
    error delta_k(t) = a_hat_k(t) - a_k(t) is measured on HALF A of the held-out test posterior samples at every Euler
    grid time, and removed during sampling:  v <- v - U diag(delta(t)) U' (x - (1-t) m)   for t <= t_cut.
    Arms: base, calib (t_cut = 0.4), calib_all (t_cut = 1.0); iso on base for reference. Same starting noise as `eval`.
    Leakage note: half A of the test LATENTS sets 256 x (grid) gain numbers; metrics use all test IMAGES as reference.
    Gate 1 (fixed 17 Nov 2026 BEFORE the run), arm calib at 100k:
      every decile variance ratio in [0.93, 1.07]; S5 abs HF >= iso (CIFAR 12.9 / CelebA 10.7); LowE45 <= iso (14.5 / 15.5);
      gFID <= base + 0.5.
    """
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    od = os.path.join(a.out, setting(a)); os.makedirs(od, exist_ok=True)
    ck0 = torch.load(os.path.join(a.base_root, setting(a), "dit.pt"), map_location=dev, weights_only=False)
    ca = argparse.Namespace(**ck0["args"])
    z, norm = load_latents(base_ns(a), ca.latent)
    _, dec_, vargs = load_vae(argparse.Namespace(runs=a.runs, dataset=a.dataset, tag=a.tag, which=a.which), dev)
    model = make_dit(ca).to(dev).eval()
    amp = bool(getattr(ca, "amp", False)) and dev == "cuda"
    rng = np.random.default_rng(0)
    mu, lv = z["train_mu"].astype(np.float32), z["train_lv"].astype(np.float32)
    R = norm(mu + np.exp(0.5 * lv) * rng.standard_normal(mu.shape).astype(np.float32)).flatten(1).double().numpy()
    perm = rng.permutation(len(R)); n3 = len(R) // 3
    RA, RB = R[perm[:n3]], R[perm[n3:2 * n3]]
    w, V = np.linalg.eigh(np.cov(RA, rowvar=False))
    U = V[:, np.argsort(w)[::-1]]
    mB = RB.mean(0)
    lam = ((RB - mB) @ U).var(0, ddof=1)
    nt = len(z["test_mu"]); half = nt // 2
    n_cal = 300 if a.smoke else min(half, a.n)
    tmu, tlv = z["test_mu"][:n_cal].astype(np.float32), z["test_lv"][:n_cal].astype(np.float32)
    TA = norm(tmu + np.exp(0.5 * tlv) * np.random.default_rng(1).standard_normal(tmu.shape).astype(np.float32))
    Ut = torch.from_numpy(U).float().to(dev); mt = torch.from_numpy(mB).float().to(dev)
    real = test_images(base_ns(a), vargs)
    Er, Hr, hr = spectra(real)
    ch, ce, efl = np.quantile(hr, [.2, .4, .6, .8]), np.quantile(Hr, [.2, .4, .6, .8]), np.quantile(Er, 0.05)
    grid = np.linspace(1, 0, 101)
    n_ev = 200 if a.smoke else a.n

    def v_of(x, t):
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            return model(x, torch.full((len(x),), float(t), device=dev)).float()

    def measure(zn):
        fake = torch.cat([to_01(dec_(denorm(zn[s:s + 250].to(dev), z, dev))).clamp(0, 1).cpu() for s in range(0, len(zn), 250)])
        E, H, hh = spectra(fake)
        qh, qe = np.digitize(hh, ch), np.digitize(H, ce)
        G = zn.flatten(1).double().numpy()
        rr, r0, _ = ratio_deciles(R, G, np.random.default_rng(0))
        return {"ratio_dec": rr, "null_dec": r0, "S5_h": float(100 * (qh == 4).mean()),
                "S5_absHF": float(100 * (qe == 4).mean()),
                "lowE45": float(100 * (E < efl)[qh >= 3].mean()) if (qh >= 3).any() else float("nan"),
                "gfid": fid_between(real, fake, dev)}

    def sample(delta, t_cut):
        out = []
        for s in range(0, n_ev, a.bs):
            m = min(a.bs, n_ev - s)
            g = torch.Generator(device=dev).manual_seed(777 + 1000003 * a.seed + s)
            x = torch.randn(m, 4, 8, 8, device=dev, generator=g)               # same noise as run_sampler / eval
            for k in range(100):
                t, tn = grid[k], grid[k + 1]
                vt = v_of(x, t)
                if delta is not None and t <= t_cut + 1e-9:
                    P = (x.flatten(1) - (1 - t) * mt) @ Ut
                    vt = vt - ((P * delta[k]) @ Ut.T).view_as(vt)
                x = x + (tn - t) * vt
            out.append(x.cpu())
        return torch.cat(out)

    res_path = os.path.join(od, f"calib_{a.weights}.json")
    res = json.load(open(res_path)) if os.path.exists(res_path) else {}
    for step in a.restart_steps:
        key = str(step)
        if key in res and not a.redo:
            continue
        if step == ck0["step"]:
            sd = ck0["ema"] if a.weights == "ema9999" else ck0["model"]
        else:
            p = os.path.join(od, f"snap_{step}.pt")
            if not os.path.exists(p):
                raise SystemExit(f"missing {p}: run `train` first")
            sd = torch.load(p, map_location=dev, weights_only=False)[a.weights]
        model.load_state_dict(sd); t0c = time.time()
        gen = torch.Generator(device=dev).manual_seed(99)
        delta = torch.zeros(100, 256, device=dev)
        for k in range(100):                                               # gain error at every grid time, half A
            t = grid[k]
            S = {q: torch.zeros(256, dtype=torch.float64) for q in ("x", "xx", "v", "vx")}
            for s in range(0, n_cal, a.bs):
                x0 = TA[s:s + a.bs].to(dev)
                xt = (1 - t) * x0 + t * torch.randn(x0.shape, device=dev, generator=gen)
                X = ((xt.flatten(1) - (1 - t) * mt) @ Ut).double().cpu()
                Vm = (v_of(xt, t).flatten(1) @ Ut).double().cpu()
                S["x"] += X.sum(0); S["xx"] += (X * X).sum(0); S["v"] += Vm.sum(0); S["vx"] += (Vm * X).sum(0)
            varx = S["xx"] / n_cal - (S["x"] / n_cal) ** 2
            ahat = (S["vx"] / n_cal - S["v"] / n_cal * S["x"] / n_cal) / varx
            aex = torch.from_numpy((t - (1 - t) * lam) / ((1 - t) ** 2 * lam + t ** 2))
            delta[k] = (ahat - aex).float().to(dev)
        r = {"step": step, "n_cal": n_cal}
        for arm, (dl, tc) in {"base": (None, 0), "calib": (delta, a.t_cut), "calib_all": (delta, 1.0)}.items():
            Z = sample(dl, tc)
            r[arm] = measure(Z)
            if arm == "base":
                G = Z.flatten(1).double().numpy()
                s_iso = float(np.sqrt(np.trace(np.cov(R, rowvar=False)) / np.trace(np.cov(G, rowvar=False))))
                Gi = R.mean(0) + s_iso * (G - G.mean(0))
                r["iso_here"] = measure(torch.from_numpy(Gi.astype(np.float32)).view_as(Z)); r["iso_here"]["s"] = s_iso
            print(f"step {step} {arm}: S5 abs HF {r[arm]['S5_absHF']:.1f}, gFID {r[arm]['gfid']:.2f}, "
                  f"ratio dec1 {r[arm]['ratio_dec'][0]:.3f} dec10 {r[arm]['ratio_dec'][-1]:.3f}", flush=True)
        res[key] = r
        json.dump(res, open(res_path, "w"), indent=1)
        print(f"  ({(time.time() - t0c) / 60:.1f} min)", flush=True)
    calib_table(a)


def calib_table(a):
    p = os.path.join(a.out, setting(a), f"calib_{a.weights}.json")
    if not os.path.exists(p):
        print("no calib json yet"); return
    res = json.load(open(p))
    print(f"\n{setting(a)}  weights {a.weights}: held-out gain calibration (t_cut {a.t_cut} for arm calib)")
    print(f"{'step':>6s} {'arm':>9s} {'S5 h':>5s} {'S5 aHF':>6s} {'lowE45':>6s} {'gFID':>6s} | variance ratio dec1 ... dec10")
    for r in sorted(res.values(), key=lambda r: r["step"]):
        for arm in ("base", "iso_here", "calib", "calib_all"):
            m = r[arm]
            print(f"{r['step']:6d} {arm:>9s} {m['S5_h']:5.1f} {m['S5_absHF']:6.1f} {m['lowE45']:6.1f} {m['gfid']:6.2f} | "
                  + " ".join(f"{x:.3f}" for x in m["ratio_dec"]))
    l0 = REF_L0[a.dataset]
    print(f"   reference L0 at 100k (from `maps`): S5 aHF {l0[0]}, LowE45 {l0[1]}, gFID {l0[2]}")
    r = max(res.values(), key=lambda r: r["step"]); c, b = r["calib"], r["base"]
    iso_s5, iso_le = GATE1_ISO[a.dataset]
    checks = {f"every decile ratio in [0.93, 1.07] (min {min(c['ratio_dec']):.3f}, max {max(c['ratio_dec']):.3f})":
                  min(c["ratio_dec"]) >= 0.93 and max(c["ratio_dec"]) <= 1.07,
              f"S5 abs HF >= iso {iso_s5} ({c['S5_absHF']:.1f})": c["S5_absHF"] >= iso_s5,
              f"LowE45 <= iso {iso_le} ({c['lowE45']:.1f})": c["lowE45"] <= iso_le,
              f"gFID <= base + 0.5 ({c['gfid']:.2f} vs {b['gfid']:.2f})": c["gfid"] <= b["gfid"] + 0.5}
    print(f"\nGATE 1 (fixed 17 Nov 2026) at {r['step']}, arm calib:")
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'fail'}  {k}")
    print("  => " + ("PASS" if all(checks.values()) else "FAIL"))


FLIP_REF = {"cifar10": (0.104, 8.0, 27.06), "celeba": (0.144, 5.6, 14.18)}   # no-flip 100k ema9999: gap, S5, gFID


def fliptest(a):
    """Step 2 baseline test, fixed 17 Nov 2026 BEFORE the flip run. Reads runs_p3_flip (this --out) at 100k ema9999.
    Flip augmentation counts as FIXING the generalisation gap if
      1. held-out dec10 predicted deficit (`overfit`) <= half of the no-flip value (CIFAR 0.104, CelebA 0.144)
      2. S5 abs HF with no map (`eval`) >= no-flip + 2.0 (CIFAR 8.0, CelebA 5.6)
      3. gFID <= no-flip + 0.5 (CIFAR 27.06, CelebA 14.18)"""
    od = os.path.join(a.out, setting(a))
    of = json.load(open(os.path.join(od, f"overfit_{a.weights}.json")))["100000"]
    ev = json.load(open(os.path.join(od, "eval.json")))
    e = ev.get(f"100000_{a.weights}") or next(v for k, v in ev.items() if k.startswith("100000") and a.weights in k)
    gap0, s50, f0 = FLIP_REF[a.dataset]
    rd = os.path.join(a.ref_out, setting(a))           # the no-flip run of the SAME seed, if it exists
    try:
        gap0 = -json.load(open(os.path.join(rd, f"overfit_{a.weights}.json")))["100000"]["test"]["pred_ln_sd"][-1]
        er = json.load(open(os.path.join(rd, "eval.json")))[f"100000_{a.weights}"]
        s50, f0 = er["S5_absHF"], er["gfid"]
        print(f"reference (no flip, same seed) read from {rd}")
    except (FileNotFoundError, KeyError):
        print(f"reference: built-in values for the first seed (no no-flip results in {rd})")
    gap = -of["test"]["pred_ln_sd"][-1]
    s5 = e.get("S5_absHF", e.get("s5_ahf")); fid = e.get("gfid", e.get("gFID"))
    checks = {f"held-out dec10 deficit {gap:.3f} <= {gap0 / 2:.3f}": gap <= gap0 / 2,
              f"S5 abs HF {s5:.1f} >= {s50 + 2.0:.1f}": s5 >= s50 + 2.0,
              f"gFID {fid:.2f} <= {f0 + 0.5:.2f}": fid <= f0 + 0.5}
    print(f"\nFLIP TEST (fixed 17 Nov 2026), {setting(a)} at 100k {a.weights}  (train-state deficit "
          f"{-of['train']['pred_ln_sd'][-1]:+.3f})")
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'fail'}  {k}")
    print("  => " + ("flip augmentation FIXES the gap" if all(checks.values()) else "NOT fixed by flip augmentation"))


def overfit_table(a):
    p = os.path.join(a.out, setting(a), f"overfit_{a.weights}.json")
    if not os.path.exists(p):
        print("no overfit json yet"); return
    res = json.load(open(p))
    f = lambda v: " ".join(f"{x:+.3f}" for x in v)
    print(f"\n{setting(a)}  weights {a.weights}: gain-map prediction ln(sd_gen/sd_real), dec1 ... dec10")
    for r in sorted(res.values(), key=lambda r: r["step"]):
        for nm in ("train", "test"):
            print(f"  {r['step']:6d} {nm:5s} {f(r[nm]['pred_ln_sd'])}")
    print("\n  loss per t-band [0-.2 .2-.4 .4-.6 .6-.8 .8-1], dec10:")
    for r in sorted(res.values(), key=lambda r: r["step"]):
        for nm in ("train", "test"):
            print(f"  {r['step']:6d} {nm:5s} " + " ".join(f"{b[-1]:.4f}" for b in r[nm]["loss_band"]))
    print("\n  nearest neighbours in the training set (normalised latents):")
    for r in sorted(res.values(), key=lambda r: r["step"]):
        n = r["nn"]
        print(f"  {r['step']:6d} median d1: generated {n['gen_d1_median']:.3f} vs test {n['test_d1_median']:.3f} | "
              f"d1/d2 < 1/3: generated {n['gen_memorised_pct']:.2f}% vs test {n['test_memorised_pct']:.2f}% | "
              f"median d1/d2 {n['gen_ratio_median']:.3f} vs {n['test_ratio_median']:.3f}")
    r = max(res.values(), key=lambda r: r["step"])
    tr, te = -r["train"]["pred_ln_sd"][-1], -r["test"]["pred_ln_sd"][-1]
    gap = te > 0 and tr <= 0.5 * te
    print(f"\nRULE (fixed 17 Nov 2026) at {r['step']}: dec10 deficit train {tr:.3f} vs test {te:.3f} -> "
          + ("GENERALISATION GAP (train <= half of test)" if gap else "NOT a generalisation gap (error present on train states too)"))


def gainmap_table(a):
    od = os.path.join(a.out, setting(a))
    p = os.path.join(od, f"gainmap_{a.weights}.json")
    if not os.path.exists(p):
        print("no gainmap json yet"); return
    res = json.load(open(p)); meta = json.load(open(os.path.join(od, "gainmap_meta.json")))
    f = lambda v: " ".join(f"{x:+.3f}" for x in v)
    print(f"\n{setting(a)}  weights {a.weights}   (deciles: dec1 = largest real variance ... dec10 = smallest)")
    print("real variance       " + " ".join(f"{x:6.3f}" for x in meta["lam_dec"]))
    print("posterior-noise share" + " ".join(f"{x:6.3f}" for x in meta["noise_share_dec"]))
    print("\n1. GAIN MAP: predicted ln(sd_gen/sd_real) = -int (a_hat - a) dt   (negative = shrinkage)")
    for r in sorted(res.values(), key=lambda r: r["step"]):
        print(f"  {r['step']:6d} pred  {f(r['pred_ln_sd'])}")
        if "obs_ln_sd" in r:
            print(f"  {r['step']:6d} obs   {f(r['obs_ln_sd'])}   (restart from pure noise, same eigenbasis)")
        print(f"         check |a_true - a_exact| = {r['check_true_minus_exact']:.4f} (should be ~0)")
    print("\n   dec10 prediction split by t-band [0-.2 .2-.4 .4-.6 .6-.8 .8-1] and part from t < t*_k:")
    for r in sorted(res.values(), key=lambda r: r["step"]):
        print(f"  {r['step']:6d} " + " ".join(f"{b[-1]:+.3f}" for b in r["pred_band"]) + f"   t<t*: {r['pred_low_t'][-1]:+.3f}")
    print("\n2. LOSS MAP (held-out), dec1 / dec10, per t-band [0-.2 ... .8-1]:")
    for r in sorted(res.values(), key=lambda r: r["step"]):
        print(f"  {r['step']:6d} dec1 " + " ".join(f"{b[0]:.4f}" for b in r["loss_band"])
              + " | dec10 " + " ".join(f"{b[-1]:.4f}" for b in r["loss_band"]))
    rs = [r for r in res.values() if "restart_ratio" in r]
    if rs:
        print("\n3. RESTART TEST: variance ratio after starting from real held-out latents noised to t0 (dec1 dec5 dec10 mean)")
        for r in sorted(rs, key=lambda r: r["step"]):
            for t0, v in r["restart_ratio"].items():
                print(f"  {r['step']:6d} t0={float(t0):.1f}  {v[0]:.3f} {v[4]:.3f} {v[-1]:.3f}  mean {np.mean(v):.3f}")
        r = max(rs, key=lambda r: r["step"]); T = r["tails"]
        print(f"\n4. TAILS / NOISE at {r['step']}: kurt real {f(T['kurt_real'])}")
        print(f"   kurt gen  {f(T['kurt_gen'])}\n   IQR ratio {f(T['iqr_ratio'])}\n   tail(1-99%) ratio {f(T['tail_ratio'])}")
        print(f"   Spearman over 256 directions: ln ratio vs real kurtosis {T['spearman_lnratio_kurt']:+.2f}, "
              f"vs posterior-noise share {T['spearman_lnratio_noise_share']:+.2f}, vs gain prediction {T['spearman_lnratio_pred']:+.2f}")
    hi = max(res.values(), key=lambda r: r["step"])
    need = 0.5 * GATE0_TABLE[a.dataset][1]
    ok = -hi["pred_ln_sd"][-1] >= need
    print(f"\nGATE 0 (fixed 17 Nov 2026): at {hi['step']} predicted dec10 deficit {-hi['pred_ln_sd'][-1]:.3f} >= {need:.3f}"
          f" (half of the observed {GATE0_TABLE[a.dataset][1]:.3f})  =>  {'PASS' if ok else 'FAIL'}")
    if ok:
        lowp, high = hi["pred_low_t"][-1], hi["pred_t_gt_0.8"][-1]
        print(f"   location of the dec10 prediction: t < t*_k {lowp:+.3f}, t > 0.8 {high:+.3f} -> outcome "
              + ("A (low t)" if abs(lowp) >= abs(high) else "B (high t)"))
    elif rs:
        rr = hi.get("restart_ratio", {})
        if rr:
            full = np.log(rr["1.0"][-1])
            fr = {t0: np.log(v[-1]) / full if full < 0 else float("nan") for t0, v in rr.items()}
            print("   share of the dec10 deficit already present when restarting from t0: "
                  + ", ".join(f"t0={float(k):.1f}: {v:.2f}" for k, v in fr.items()))
            print("   guide: deficit only from t0 >= 0.6 -> outcome C; already large from t0 <= 0.3 -> outcome D;"
                  " strong negative Spearman with kurtosis -> E")


def table(a):
    p = os.path.join(a.out, setting(a), "eval.json")
    if not os.path.exists(p):
        print("no eval.json yet"); return
    res = json.load(open(p))
    order = {"raw": 0, "ema999": 1, "ema9995": 2, "ema9999": 3}
    print(f"\n{setting(a)}  (Euler 100 steps; same starting noise for every row)")
    print(f"{'step':>6s} {'weights':>8s} {'ratio':>6s} {'dec1':>5s} {'dec10':>6s} {'S5 h':>5s} {'S5 aHF':>7s} {'lowE45':>7s}"
          f" {'gFID':>6s} | {'iso s':>6s} {'S5 aHF':>7s} {'lowE45':>7s} {'gFID':>6s}")
    for r in sorted(res.values(), key=lambda r: (r["step"], order[r["weights"]])):
        print(f"{r['step']:6d} {r['weights']:>8s} {r['ratio_mean']:6.3f} {r['ratio_deciles'][0]:5.3f} {r['ratio_deciles'][-1]:6.3f}"
              f" {r['S5_h']:5.1f} {r['S5_absHF']:7.1f} {r['lowE45']:7.1f} {r['gfid']:6.2f} | {r['iso_scale']:6.3f}"
              f" {r['iso_S5_absHF']:7.1f} {r['iso_lowE45']:7.1f} {r['iso_gfid']:6.2f}")
    print("ratio = mean generated/real variance over the 10 eigen-deciles (1 = no shrinkage); dec1/dec10 = largest /"
          " smallest real-variance directions. iso columns: the same samples after the one-scalar variance fix.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "eval", "table", "maps", "maps_table", "gainmap", "gainmap_table",
                                    "energy", "energy_table", "overfit", "overfit_table", "calib", "calib_table", "fliptest"])
    ap.add_argument("--t_cut", type=float, default=0.4)
    ap.add_argument("--ref_out", default="runs_p3_ema", help="fliptest: folder of the no-flip run of the same seed")
    ap.add_argument("--aug", default="none", choices=["none", "flip"],
                    help="train only: 'flip' adds horizontally flipped training images (use with --out runs_p3_flip)")
    ap.add_argument("--gain_steps", type=int, nargs="+", default=[50000, 60000, 75000, 100000])
    ap.add_argument("--restart_steps", type=int, nargs="+", default=[50000, 100000])
    ap.add_argument("--n_t", type=int, default=40)
    ap.add_argument("--map_steps", type=int, nargs="+", default=[50000, 100000])
    ap.add_argument("--weights", default="ema9999", choices=["raw", "ema999", "ema9995", "ema9999"])
    ap.add_argument("--n_fit", type=int, default=50000)
    ap.add_argument("--dataset", required=True, choices=["cifar10", "celeba"]); ap.add_argument("--root", required=True)
    ap.add_argument("--tag", required=True); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--runs", default="runs_p2"); ap.add_argument("--which", default="last")
    ap.add_argument("--base_root", default="runs_p3"); ap.add_argument("--out", default="runs_p3_ema")
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--snap", type=int, nargs="+", default=[60000, 75000, 100000])
    ap.add_argument("--save_every", type=int, default=5000); ap.add_argument("--log_every", type=int, default=500)
    ap.add_argument("--n", type=int, default=10000); ap.add_argument("--bs", type=int, default=1000)
    ap.add_argument("--n_eval", type=int, default=10000); ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--redo", action="store_true"); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.cmd.startswith("gainmap"):
        a.map_steps = a.gain_steps
    {"train": train, "eval": evaluate, "table": table, "maps": maps, "maps_table": maps_table,
     "gainmap": gainmap, "gainmap_table": gainmap_table, "energy": energy, "energy_table": energy_table,
     "overfit": overfit, "overfit_table": overfit_table, "calib": calib, "calib_table": calib_table,
     "fliptest": fliptest}[a.cmd](a)
