# Training-Induced Variance Shrinkage in Latent Flow Matching

Code for the paper **"Training-Induced Variance Shrinkage in Latent Flow Matching: A Low-Noise Generalisation Gap and a
Simple Remedy"** by Praful Sonarkar, Pooja Jain, Richa Makhijani and Gaurav Sharma (submitted to *Pattern Analysis and
Applications*, 2026).

The code covers:
- continuing a latent flow-matching DiT from 50k to 100k steps with three EMAs, with or without re-encoded flip
  augmentation;
- the per-direction diagnostics: variance ratio per eigen-decile, the held-out gain map (exact gain target), the
  restart test, the exact energy decomposition of Euler sampling into gain and exposure, and the train vs held-out
  generalisation test with a nearest-neighbour memorisation check;
- the sampler diagnosis (Euler 100/200/400, Heun, stochastic, raw weights) with the sampler-only reference;
- the latent maps (iso, L0, null) and the KID differences with 95% ranges.

Every test has a decision rule that was written into the script, with its date, before the run that it judges.

## Files

| File | What it does |
|---|---|
| `p3_ema_study.py` | Main script: `train` (continue to 100k, `--aug flip`), `eval`, `maps`, `gainmap`, `energy`, `overfit`, `calib`, `fliptest`, and a `*_table` command for each |
| `p3_sampler_diag.py` | Sampler diagnosis with the sampler-only reference (exact Gaussian velocity) |
| `p4_kid.py` | KID differences flip vs no-flip, with and without L0, paired by starting noise, 95% ranges |
| `p4_collect.py` | Collects every number used in the paper into `p4_numbers.json` |
| `make_p4_figs.py` | Figures 1 and 2 of the paper |
| `p3_latent_gen.py` | Base DiT (`train` to 50k), latents, sampling, FID/KID helpers |
| `p3_h_check.py` | Complexity strata (absolute high-frequency energy) used for S5 and LowE45 |
| `paper2_train.py`, `paper2_decoder_gan.py`, `nested_vae_spectral.py` | The frozen autoencoder (model D) and the data loaders, from the companion autoencoder paper |
| `spectral_rd_vae_test.py` | Helper functions used by `nested_vae_spectral.py` |
## Requirements

Python 3.12, PyTorch with CUDA (all results were obtained on one 8 GB NVIDIA Quadro RTX 4000), and:

```
pip install torch torchvision numpy scipy matplotlib lpips torchmetrics torch-fidelity
```

## Data

CIFAR-10 and a 50,000-image subset of CelebA at 64x64, with the same folders and splits as the companion papers.
The datasets are not redistributed here.

## Pipeline

Run from the repository folder. Replace `<root>` with the data folder; use `--dataset celeba --tag D_strong_gan` for
CelebA and add `--seed 1` for the second DiT seed. A complete chain for one setting is about 14 hours on the GPU above.

```
# 0. autoencoder D and the base DiT (50k steps)
python paper2_train.py train --dataset cifar10 --root <root> --tag D_strong_gan_v2      # see the docstring for D's options
python p3_latent_gen.py encode --dataset cifar10 --root <root> --tag D_strong_gan_v2
python p3_latent_gen.py train  --dataset cifar10 --root <root> --tag D_strong_gan_v2 --steps 50000 --amp

# 1. continue to 100k without flips, then the diagnostics
python p3_ema_study.py train   --dataset cifar10 --root <root> --tag D_strong_gan_v2
python p3_ema_study.py eval    --dataset cifar10 --root <root> --tag D_strong_gan_v2
python p3_ema_study.py gainmap --dataset cifar10 --root <root> --tag D_strong_gan_v2
python p3_ema_study.py energy  --dataset cifar10 --root <root> --tag D_strong_gan_v2
python p3_ema_study.py overfit --dataset cifar10 --root <root> --tag D_strong_gan_v2
python p3_ema_study.py calib   --dataset cifar10 --root <root> --tag D_strong_gan_v2
python p3_ema_study.py maps    --dataset cifar10 --root <root> --tag D_strong_gan_v2 --map_steps 100000
python p3_sampler_diag.py      --dataset cifar10 --root <root> --tag D_strong_gan_v2

# 2. continue to 100k WITH flips, its diagnostics, and the flip test
python p3_ema_study.py train    --dataset cifar10 --root <root> --tag D_strong_gan_v2 --aug flip --out runs_p3_flip
python p3_ema_study.py overfit  --dataset cifar10 --root <root> --tag D_strong_gan_v2 --out runs_p3_flip --restart_steps 100000
python p3_ema_study.py eval     --dataset cifar10 --root <root> --tag D_strong_gan_v2 --out runs_p3_flip
python p3_ema_study.py maps     --dataset cifar10 --root <root> --tag D_strong_gan_v2 --out runs_p3_flip --map_steps 100000
python p3_ema_study.py fliptest --dataset cifar10 --root <root> --tag D_strong_gan_v2 --out runs_p3_flip

# 3. KID differences, numbers and figures
python p4_kid.py --dataset cifar10 --root <root> --tag D_strong_gan_v2
python p4_collect.py
python make_p4_figs.py
```

Every script has a `--smoke` option that runs the chain on a few images; its numbers mean nothing, it only checks the
code. Training resumes when the same command is run again. Model checkpoints are not included.

<!-- TODO (Praful): check every command above against what you ran, especially the paper2_train.py options for D and
     the p3_sampler_diag.py arguments, before making the repository public. -->

## Citation

```
P. Sonarkar, P. Jain, R. Makhijani, G. Sharma, Training-induced variance shrinkage in latent flow matching: a
low-noise generalisation gap and a simple remedy, submitted to Pattern Analysis and Applications (2026).
```

## License

<!-- TODO (Praful): choose a license with your supervisors (for example MIT) and add it when creating the repository. -->

## Contact

Praful Sonarkar, Indian Institute of Information Technology Nagpur: dtvi22cse001@iiitn.ac.in
