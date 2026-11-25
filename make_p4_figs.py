"""Figures for Paper 4 (numbers from p4_numbers.json, 25 Nov 2026)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
steps = [50, 60, 75, 100]
NF = {"CIFAR-10 s0": ([.876, .861, .834, .802], [10.3, 9.9, 9.2, 8.0]),
      "CIFAR-10 s1": ([.865, .851, .826, .807], [9.9, 9.5, 8.7, 8.1]),
      "CelebA s0": ([.838, .826, .808, .780], [7.0, 6.9, 6.6, 5.6]),
      "CelebA s1": ([.836, .827, .808, .779], [7.0, 7.0, 6.6, 5.6])}
FL = {"CIFAR-10 s0": ([.876, .879, .875, .871], [10.3, 10.7, 10.8, 10.6]),
      "CIFAR-10 s1": ([.865, .868, .866, .870], [9.9, 10.1, 10.2, 10.5]),
      "CelebA s0": ([.838, .843, .847, .850], [7.0, 7.4, 7.6, 7.7]),
      "CelebA s1": ([.836, .846, .853, .849], [7.0, 7.4, 7.8, 7.4])}
col = {"CIFAR-10 s0": "#1f6fb4", "CIFAR-10 s1": "#7fb2e0", "CelebA s0": "#d9541e", "CelebA s1": "#f0a27a"}
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6))
for j, lab in enumerate(["Variance ratio (generated / real)", "S5 abs HF (%)"]):
    for k in NF:
        if NF[k][j] is not None:
            ax[j].plot(steps, NF[k][j], "-o", ms=3, color=col[k], label=f"{k}, no flip")
        ax[j].plot(steps, FL[k][j], "--s", ms=3, color=col[k], label=f"{k}, flip")
    ax[j].set_xlabel("Training step (thousands)"); ax[j].set_ylabel(lab); ax[j].set_xticks(steps)
ax[0].text(0.5, -0.38, "(a)", transform=ax[0].transAxes, ha="center")
ax[1].text(0.5, -0.38, "(b)", transform=ax[1].transAxes, ha="center")
h, l = ax[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=4, fontsize=7, frameon=False, bbox_to_anchor=(0.5, 1.12))
fig.tight_layout(); fig.savefig("Fig1.pdf", bbox_inches="tight")

dec = list(range(1, 11))
G = {"CIFAR-10 s1": {"train": [.010, .008, .008, .005, .006, .001, .002, .007, .006, .010],
                    "test": [-.037, -.054, -.063, -.070, -.073, -.079, -.080, -.087, -.097, -.097],
                    "flip test": [-.013, -.017, -.020, -.025, -.024, -.029, -.027, -.028, -.030, -.027]},
     "CelebA s0": {"train": [.015, .014, .007, .004, .004, .001, -.002, .002, .002, .012],
                   "test": [-.032, -.058, -.068, -.075, -.086, -.097, -.106, -.114, -.121, -.147],
                   "flip test": [-.007, -.015, -.019, -.024, -.027, -.031, -.033, -.032, -.032, -.039]}}
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.4), sharey=True)
for j, (k, d) in enumerate(G.items()):
    ax[j].axhline(0, color="0.6", lw=0.8)
    ax[j].plot(dec, d["train"], "-o", ms=3, color="#2a9d4b", label="training latents, no flip")
    ax[j].plot(dec, d["test"], "-o", ms=3, color="#c0392b", label="held-out latents, no flip")
    ax[j].plot(dec, d["flip test"], "--s", ms=3, color="#c0392b", alpha=0.6, label="held-out latents, flip")
    ax[j].set_xlabel("Eigen-decile (1 = largest variance)"); ax[j].set_xticks(dec)
    ax[j].text(0.5, -0.36, f"({'ab'[j]}) {k}", transform=ax[j].transAxes, ha="center")
ax[0].set_ylabel("Predicted ln(sd gen / sd real)")
h, l = ax[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=3, fontsize=7, frameon=False, bbox_to_anchor=(0.5, 1.08))
fig.tight_layout(); fig.savefig("Fig2.pdf", bbox_inches="tight")
print("ok")
