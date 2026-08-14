"""Geração reprodutível das figuras do artigo, com rótulos em INGLÊS.

O periódico-alvo é internacional; todas as figuras do manuscrito
(`docs/artigo/figures/`) são regeneradas aqui em inglês, a partir das fórmulas
(figuras conceituais) ou das evidências multi-semente versionadas em
`experiments/evidence/` (figuras de dados). Saída vetorial em PDF.

Figuras cobertas (as efetivamente referenciadas em `artigo.tex`):
  fig1_grounding, fig2_predicate_mlp, fig3a_tnorms, fig3b_implicacao,
  fig3c_gradiente_comparacao, fig4_loss_scheme, fig5_dag_prova_hinton,
  fig7_composta_vs_plana, fig8_tradeoff_gamma.

Uso: `uv run python experiments/plots/make_article_figures.py`
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "docs" / "artigo" / "figures"
EVID = ROOT / "experiments" / "evidence"

# paleta consistente com o manuscrito
STEEL = "#33475b"
ORANGE = "#c1660c"
GREEN = "#2f6b34"
RED = "#b0413e"
PURPLE = "#6d4a91"

plt.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "dejavuserif",
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "savefig.bbox": "tight",
    }
)


def _save(fig, name):
    out = FIG_DIR / name
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


def _load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------
# Fig. 3 (consolidada) — implication operators + gradient profile
# Substitui fig3a/3b/3c: a superfície de conjunção/disjunção (textbook) foi
# removida; mantêm-se as implicações (a,b) e o argumento de gradiente (c), que
# é o que sustenta a escolha da Product T-norm.
# --------------------------------------------------------------------------
def fig3_operadores():
    import matplotlib.gridspec as gridspec

    a = np.linspace(0, 1, 200)
    b = np.linspace(0, 1, 200)
    A, B = np.meshgrid(a, b)
    reich = 1 - A + A * B
    luka_surf = np.minimum(1, 1 - A + B)

    fig = plt.figure(figsize=(11, 8))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.05, 0.9], hspace=0.32, wspace=0.22)

    ax_a = fig.add_subplot(gs[0, 0])
    cf = ax_a.contourf(A, B, reich, levels=20, cmap="magma")
    ax_a.set_title(r"(a) Reichenbach S-implication: $1-a+ab$")
    ax_a.set_xlabel("$a$")
    ax_a.set_ylabel("$b$")
    fig.colorbar(cf, ax=ax_a)

    ax_b = fig.add_subplot(gs[0, 1])
    cf = ax_b.contourf(A, B, luka_surf, levels=20, cmap="magma")
    ax_b.set_title("(b) Łukasiewicz implication: " r"$\min(1,\,1-a+b)$")
    ax_b.set_xlabel("$a$")
    ax_b.set_ylabel("$b$")
    ax_b.annotate(
        "flat region\n(zero gradient in\nthe associated T-norm)",
        xy=(0.16, 0.16), xytext=(0.44, 0.24), fontsize=9, ha="left",
        arrowprops=dict(arrowstyle="->", color="black", lw=1),
    )
    fig.colorbar(cf, ax=ax_b)

    ax_c = fig.add_subplot(gs[1, :])
    x = np.linspace(0, 1, 500)
    bb = 0.1
    ax_c.axvspan(0, 1, color=ORANGE, alpha=0.06)
    ax_c.plot(x, np.full_like(x, bb), color=GREEN, lw=2.5,
              label=r"$\partial T_{\mathrm{prod}}/\partial a = b$ (constant)")
    ax_c.plot(x, np.where(x + bb - 1 > 0, 1.0, 0.0), color=ORANGE, lw=2.5,
              label=r"$\partial T_{\mathrm{Ł}}/\partial a$ (Łukasiewicz)")
    ax_c.text(0.40, 0.55, r"flat region: gradient $\equiv 0$ (e.g. $a=b=0.1$)",
              color=ORANGE, fontsize=10)
    ax_c.set_xlabel("$a$")
    ax_c.set_ylabel(r"$\partial(\cdot)/\partial a$ at $b=0.1$")
    ax_c.set_title(r"(c) Gradient flow: Product T-norm vs. Łukasiewicz ($b=0.1$)")
    ax_c.set_ylim(-0.05, 1.12)
    ax_c.legend(loc="upper left")

    _save(fig, "fig3_operadores.pdf")


# --------------------------------------------------------------------------
# Fig. 7 — composed (proof DAG) vs flat predictor, two domains, 5 seeds
# --------------------------------------------------------------------------
def fig7():
    kin = _load(EVID / "kinship_proof_dag_multiseed" / "kinship_report.json")["aggregate"]
    hin = _load(EVID / "hinton_family_multiseed" / "derived_report.json")["explainability"]

    domains = ["Synthetic kinship\n(transitive closure)", "Hinton (1986)\n(real family tree)"]
    mass_c = [kin["intermediate_mass_composed"]["mean"], hin["intermediate_mass_composed"]["mean"]]
    mass_c_e = [kin["intermediate_mass_composed"]["std"], hin["intermediate_mass_composed"]["std"]]
    del_c = [kin["deletion_delta_composed"]["mean"], hin["deletion_delta_composed"]["mean"]]
    del_c_e = [kin["deletion_delta_composed"]["std"], hin["deletion_delta_composed"]["std"]]

    x = np.arange(2)
    w = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

    # (a) gradient mass on intermediates
    ax = axes[0]
    ax.bar(x - w / 2, mass_c, w, yerr=mass_c_e, capsize=4, color=GREEN,
           label="Composed (proof DAG)")
    ax.bar(x + w / 2, [0, 0], w, color=RED, label="Flat predictor")
    for i, (m, e) in enumerate(zip(mass_c, mass_c_e)):
        ax.text(i - w / 2, m + e + 0.02, f"{m*100:.1f}%", ha="center", fontsize=10)
        ax.text(i + w / 2, 0.02, "0%", ha="center", fontsize=10)
    ax.set_ylabel("Gradient mass on intermediates")
    ax.set_title("(a) Gradient concentration")
    ax.set_xticks(x)
    ax.set_xticklabels(domains)
    ax.set_ylim(0, 0.62)
    ax.legend()

    # (b) causal deletion of intermediates
    ax = axes[1]
    ax.bar(x - w / 2, del_c, w, yerr=del_c_e, capsize=4, color=GREEN,
           label="Composed (proof DAG)")
    ax.bar(x + w / 2, [0, 0], w, color=RED, label="Flat predictor")
    for i, (m, e) in enumerate(zip(del_c, del_c_e)):
        ax.text(i - w / 2, m + e + 0.02, f"{m:.2f}", ha="center", fontsize=10)
        ax.text(i + w / 2, 0.02, "0.00", ha="center", fontsize=10)
    ax.set_ylabel(r"$\Delta$ truth degree after causal deletion")
    ax.set_title("(b) Causal deletion of intermediates")
    ax.set_xticks(x)
    ax.set_xticklabels(domains)
    ax.set_ylim(0, 1.05)
    ax.legend()

    fig.suptitle("Composed deduction (proof DAG) vs. flat prediction — two domains "
                 "(mean $\\pm$ s.d. over 5 seeds)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, "fig7_composta_vs_plana.pdf")


# --------------------------------------------------------------------------
# Fig. 8 — gamma vs sparsity vs fidelity trade-off (p=13)
# --------------------------------------------------------------------------
def fig8():
    sweep = _load(EVID / "fidelity_sparsity_p13" / "report.json")["part_b_gamma_sweep"]
    order = ["gamma_0.0001", "gamma_0.001", "gamma_0.01", "gamma_0.1"]
    gammas = [sweep[k]["gamma_l1"] for k in order]
    sparsity = [sweep[k]["sparsity"] for k in order]
    fidelity = [sweep[k]["fidelity_overall"]["mean"] for k in order]

    fig, ax1 = plt.subplots(figsize=(8, 4.8))
    ax2 = ax1.twinx()
    l1 = ax1.plot(gammas, sparsity, "-o", color=STEEL, lw=2, label="Sparsity")
    l2 = ax2.plot(gammas, fidelity, "--s", color=ORANGE, lw=2, label="Fidelity")
    ax1.set_xscale("log")
    ax1.set_xlabel(r"$\gamma$ ($L_1$ regularization)")
    ax1.set_ylabel("Structural sparsity", color=STEEL)
    ax2.set_ylabel("Axiomatic fidelity (overall)", color=ORANGE)
    ax1.set_ylim(0, 1)
    ax2.set_ylim(0.60, 1.0)

    ax1.axvline(1e-2, color="gray", ls=":", lw=1)
    s_cal = sweep["gamma_0.01"]["sparsity"] * 100
    f_cal = sweep["gamma_0.01"]["fidelity_overall"]["mean"] * 100
    ax1.text(1.1e-2, 0.10,
             f"$\\gamma=10^{{-2}}$\n(calibrated: {s_cal:.1f}% sparse,\nfidelity {f_cal:.1f}%)",
             color="gray", fontsize=9)
    ax1.set_title(r"Trade-off $\gamma \times$ sparsity $\times$ fidelity ($p=13$)")
    lines = l1 + l2
    ax1.legend(lines, [ln.get_label() for ln in lines], loc="center right")
    fig.tight_layout()
    _save(fig, "fig8_tradeoff_gamma.pdf")


# --------------------------------------------------------------------------
# schematic helpers
# --------------------------------------------------------------------------
def _box(ax, xy, w, h, text, fc, ec, fs=11, tc="black"):
    x, y = xy
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.03",
                                fc=fc, ec=ec, lw=1.6))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc)


def _arrow(ax, p0, p1, color=STEEL):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=16,
                                 color=color, lw=1.6, shrinkA=2, shrinkB=2))


# --------------------------------------------------------------------------
# Fig. 1 — symbol grounding
# --------------------------------------------------------------------------
def fig1():
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")

    labels = ["$c_1$", "$c_2$", "$c_3$", r"$\ldots$", "$c_k$"]
    ys = np.linspace(4.1, 0.5, 5)
    for lab, y in zip(labels, ys):
        _box(ax, (0.7, y - 0.28), 0.7, 0.56, lab, "#eef2f7", STEEL, fs=12)
    ax.text(1.05, -0.15, "symbolic constants", ha="center", fontsize=12)

    _arrow(ax, (2.0, 2.3), (4.2, 2.3))
    ax.text(3.1, 2.55, "grounding", ha="center", fontsize=12, style="italic")

    ax.add_patch(FancyBboxPatch((4.6, 0.6), 4.8, 3.6,
                                boxstyle="round,pad=0.02,rounding_size=0.05",
                                fc="#fdf1df", ec=ORANGE, lw=1.8))
    ax.text(6.9, 3.9, r"$\mathbb{R}^d$", ha="center", fontsize=14)
    pts = [(5.6, 2.6), (6.5, 2.65), (7.4, 2.7), (8.4, 1.9), (8.3, 1.3)]
    names = [r"$\ldots$", "$e_{c_3}$", "$e_{c_1}$", "$e_{c_k}$", "$e_{c_2}$"]
    for (px, py), nm in zip(pts, names):
        if nm != r"$\ldots$":
            ax.plot(px, py, "o", color=ORANGE, ms=10)
        ax.text(px + 0.18, py + 0.12, nm, fontsize=12)
    ax.text(7.0, 0.15, "trainable latent vectors", ha="center", fontsize=12)
    _save(fig, "fig1_grounding.pdf")


# --------------------------------------------------------------------------
# Fig. 2 — neural predicate as MLP (Add, modular-addition domain)
# --------------------------------------------------------------------------
def fig2():
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3)
    ax.axis("off")
    blocks = [
        (r"$[e_a\|e_b\|e_c]$" + "\n(24)", "#fdf1df", ORANGE),
        ("Linear\n24\nReLU", "#eef2f7", STEEL),
        ("Linear\n24\nReLU", "#eef2f7", STEEL),
        ("Linear\n1", "#eef2f7", STEEL),
        (r"Sigmoid" + "\n" + r"$f_P \in [0,1]$", "#e6f0e6", GREEN),
    ]
    x = 0.3
    w, h = 1.65, 1.5
    centers = []
    for text, fc, ec in blocks:
        _box(ax, (x, 0.75), w, h, text, fc, ec, fs=11)
        centers.append(x + w)
        x += w + 0.28
    for cx in centers[:-1]:
        _arrow(ax, (cx, 1.5), (cx + 0.28, 1.5))
    _save(fig, "fig2_predicate_mlp.pdf")


# --------------------------------------------------------------------------
# Fig. 4 — conceptual loss-landscape schematic (illustrative, not measured)
# --------------------------------------------------------------------------
def fig4():
    rng = np.random.default_rng(0)
    x = np.linspace(-1, 1, 400)
    # left: rugged landscape with spurious minima (illustrative)
    rugged = (1.6 * x**2 + 0.35 * np.sin(6 * x) + 0.18 * np.sin(13 * x)
              + 0.12 * np.cos(21 * x) + 1.0)
    # right: smooth convex bowl
    smooth = 3.15 * x**2

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(x, rugged, color=ORANGE, lw=2)
    axes[0].fill_between(x, rugged, rugged.max() + 0.3, color=ORANGE, alpha=0.08)
    axes[0].set_title(r"Without $\mathcal{L}_{\mathrm{semantic}}$ (schematic)")
    axes[1].plot(x, smooth, color=GREEN, lw=2)
    axes[1].fill_between(x, smooth, smooth.max() + 0.3, color=GREEN, alpha=0.08)
    axes[1].set_title(r"With $\mathcal{L}_{\mathrm{semantic}}$ (schematic)")
    for ax in axes:
        ax.set_xlabel(r"$\theta$ (illustrative projection)")
        ax.set_ylabel(r"$\mathcal{L}_{\mathrm{total}}$ (illustrative)")
        ax.set_xticks([])
    fig.suptitle("Conceptual loss-surface schematic — not directly measured",
                 fontsize=12, style="italic")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, "fig4_loss_scheme.pdf")


# --------------------------------------------------------------------------
# Fig. 5 — proof DAG for uncle(arthur, colin) [Hinton domain]
# --------------------------------------------------------------------------
def fig5():
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis("off")

    ax.text(6, 7.4,
            r"$\mathrm{uncle}(x,y) = \bigvee_z\ \mathrm{brother}(x,z)\ \wedge\ "
            r"(\mathrm{father}(z,y)\ \vee\ \mathrm{mother}(z,y))$",
            ha="center", fontsize=13)

    _box(ax, (0.6, 5.2), 2.6, 1.1, "brother(arthur,\nvictoria)", "#fdf1df", ORANGE, fs=11)
    _box(ax, (0.6, 1.1), 2.6, 1.0, "father(victoria,\ncolin)", "#fdf1df", ORANGE, fs=11)
    _box(ax, (0.6, 0.0), 2.6, 1.0, "mother(victoria,\ncolin)", "#fdf1df", ORANGE, fs=11)
    _box(ax, (3.9, 1.15), 1.7, 1.0, r"$\vee$" + "\n(parent)", "#efe6f7", PURPLE, fs=12)
    _box(ax, (5.2, 3.1), 1.7, 1.1, r"$\wedge$" + "\n(victoria)", "#f7e0e0", RED, fs=12)
    _box(ax, (8.4, 3.1), 2.8, 1.1, "uncle(arthur,\ncolin)", "#e6f0e6", GREEN, fs=11)

    _arrow(ax, (3.2, 5.5), (5.4, 4.0))       # brother -> AND
    _arrow(ax, (5.5, 2.15), (6.0, 3.1))      # OR(parent) -> AND
    _arrow(ax, (3.2, 1.6), (3.9, 1.6))       # father -> OR
    _arrow(ax, (3.2, 0.5), (3.95, 1.35))     # mother -> OR
    _arrow(ax, (6.9, 3.65), (8.4, 3.65))     # AND -> uncle

    ax.text(3.6, 4.7, "intermediate $z^{*}=$ victoria", color=RED, fontsize=11)
    ax.text(7.0, 1.7,
            r"gradient mass on the intermediate ($\|\nabla_{e_{victoria}}\|_1$)"
            "\nand its causal deletion $\\Delta$:\naggregate over 5 seeds in Table 3",
            color=RED, fontsize=9.5, ha="left")
    ax.text(6, -0.5,
            "predicted truth degree (composed DLG); equivalent flat predictor: "
            r"mass$/\Delta = 0$ (no intermediate)",
            ha="center", fontsize=9, style="italic", color="#555555")
    _save(fig, "fig5_dag_prova_hinton.pdf")


def main():
    fig1()
    fig2()
    fig3_operadores()
    fig4()
    fig5()
    fig7()
    fig8()
    print("done.")


if __name__ == "__main__":
    main()
