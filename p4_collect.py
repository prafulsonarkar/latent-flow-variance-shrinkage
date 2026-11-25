"""
p4_collect.py -- gather every number Paper 4 uses into ONE file (p4_numbers.json), so the draft tables can be filled
from the result files themselves instead of from copied console output. Reads only; changes nothing. Takes seconds.

Run from D:\\new_vae_research:   python p4_collect.py
Then send p4_numbers.json.
"""
import json, os

SETTINGS = ["cifar10_D_strong_gan_v2", "cifar10_D_strong_gan_v2_seed1", "celeba_D_strong_gan", "celeba_D_strong_gan_seed1"]
FILES = {
    "noflip": ("runs_p3_ema", ["eval.json", "maps_ema9999.json", "gainmap_ema9999.json", "gainmap_meta.json",
                               "energy_ema9999.json", "overfit_ema9999.json", "calib_ema9999.json"]),
    "flip": ("runs_p3_flip", ["eval.json", "maps_ema9999.json", "overfit_ema9999.json"]),
    "kid": ("runs_p4_kid", ["kid.json"]),
    "base50k": ("runs_p3", ["recon_strata.json", "gen_eval.json", "h_check.json"]),
}
DROP = {"pred_dir"}                       # per-direction arrays (256 numbers) are not needed in the paper


def clean(x):
    if isinstance(x, float):
        return round(x, 4)
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items() if k not in DROP}
    if isinstance(x, list):
        return [clean(v) for v in x]
    return x


out, missing = {}, []
for s in SETTINGS:
    out[s] = {}
    for part, (root, names) in FILES.items():
        for n in names:
            p = os.path.join(root, s, n)
            if os.path.exists(p):
                try:
                    out[s][f"{part}/{n}"] = clean(json.load(open(p)))
                except Exception as e:                       # a broken file should not stop the rest
                    out[s][f"{part}/{n}"] = f"could not read: {e}"
            else:
                missing.append(p)
json.dump(out, open("p4_numbers.json", "w"), indent=1)
print(f"wrote p4_numbers.json ({os.path.getsize('p4_numbers.json') // 1024} KB)")
print(f"{len(missing)} files not found (fine if a step was never run there):")
for p in missing:
    print("  ", p)
