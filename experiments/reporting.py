"""Geração de evidências por rodada: tabelas + gráficos, preservados em disco.

Todo experimento (piloto, sweep, escala cheia) deve passar por aqui depois de
rodar: cada chamada cria um diretório versionável em `experiments/evidence/`
com a configuração, o resultado bruto (curvas por época incluídas), uma tabela
de métricas em Markdown e os gráficos de curvas de treino em PNG (visualização
rápida) e PDF (vetorial, exigência do parecer para figuras de periódico).

Rótulos dos gráficos em inglês: o parecer (Seção 6/8) aponta que figuras em
português precisarão ser refeitas para um periódico internacional; gerar já em
inglês evita retrabalho.
"""

import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EVIDENCE_ROOT = Path(__file__).parent / "evidence"

_CURVE_KEYS = ["loss", "l_data", "l_semantic", "l1_penalty"]


def _is_multiseed(result: Dict) -> bool:
    return "runs" in result and "aggregate" in result


def _runs(result: Dict) -> List[Dict]:
    return result["runs"] if _is_multiseed(result) else [result]


def _fmt(value, digits=4) -> str:
    if value is None:
        return "—"
    if isinstance(value, dict):  # aggregate {'mean','std','n'}
        if value.get("mean") is None:
            return "—"
        return f"{value['mean']:.{digits}g} ± {value.get('std', 0) or 0:.{digits}g} (n={value['n']})"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def _summary_rows(result: Dict) -> List[tuple]:
    if _is_multiseed(result):
        agg = result["aggregate"]
        return [
            ("Seeds", agg["seeds"]),
            ("T_g", agg["t_g"]),
            ("Final val accuracy", agg["final_val_accuracy"]),
            ("Test accuracy", agg["test_accuracy"]),
            ("Final L1 penalty", agg["final_l1_penalty"]),
        ]
    return [
        ("Seed", result.get("seed")),
        ("T_g", result.get("t_g")),
        ("Final val accuracy", result.get("final_val_accuracy")),
        ("Test accuracy", result.get("test_accuracy")),
        ("Final L1 penalty", result.get("final_l1_penalty")),
    ]


def _write_summary_md(path: Path, name: str, result: Dict, config: Optional[Dict]):
    lines = [f"# {name}", ""]
    if config:
        lines += ["## Config", "", "```json", json.dumps(config, indent=2), "```", ""]
    lines += ["## Metrics", "", "| Metric | Value |", "|---|---|"]
    for label, value in _summary_rows(result):
        lines.append(f"| {label} | {_fmt(value)} |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _plot_run_curves(run: Dict, out_base: Path, title: str):
    logs = run.get("epoch_logs", [])
    if not logs:
        return
    epochs = list(range(1, len(logs) + 1))

    fig, ax_loss = plt.subplots(figsize=(7, 4.5))
    ax_acc = ax_loss.twinx()

    for key in _CURVE_KEYS:
        values = [l.get(key) for l in logs]
        if any(v is not None for v in values):
            if key == "l1_penalty":
                # escala diferente das demais perdas; normaliza pelo valor inicial
                first = next((v for v in values if v), None)
                if first:
                    values = [v / first if v is not None else None for v in values]
                    ax_loss.plot(epochs, values, label="l1_penalty (relative)", alpha=0.7)
                continue
            ax_loss.plot(epochs, values, label=key, alpha=0.8)

    val = [l.get("val_accuracy") for l in logs]
    if any(v is not None for v in val):
        ax_acc.plot(epochs, val, label="val_accuracy", color="black", linewidth=2)
        ax_acc.set_ylabel("Validation accuracy")
        ax_acc.set_ylim(0, 1)

    ax_loss.set_xscale("log")
    ax_loss.set_xlabel("Epoch (log scale)")
    ax_loss.set_ylabel("Loss components")
    ax_loss.set_title(title)

    handles1, labels1 = ax_loss.get_legend_handles_labels()
    handles2, labels2 = ax_acc.get_legend_handles_labels()
    ax_loss.legend(handles1 + handles2, labels1 + labels2, fontsize=8, loc="center left")

    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".png"), dpi=150)
    fig.savefig(out_base.with_suffix(".pdf"))
    plt.close(fig)


def save_run_report(
    name: str,
    result: Dict,
    config: Optional[Dict] = None,
    output_root: Optional[Path] = None,
    timestamp: bool = True,
) -> Path:
    """Salva a evidência completa de uma rodada (single-seed ou multiseed)."""
    root = Path(output_root) if output_root else EVIDENCE_ROOT
    dir_name = (
        f"{name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}" if timestamp else name
    )
    out_dir = root / dir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    if config is not None:
        with open(out_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

    _write_summary_md(out_dir / "summary.md", name, result, config)

    for run in _runs(result):
        seed = run.get("seed", 0)
        _plot_run_curves(run, out_dir / f"curves_seed{seed}", f"{name} (seed {seed})")

    return out_dir


def _mean_val_curve(result: Dict) -> List[Optional[float]]:
    """Curva de val_accuracy média entre seeds (single-seed vira a própria curva)."""
    runs = _runs(result)
    curves = []
    for run in runs:
        curves.append([l.get("val_accuracy") for l in run.get("epoch_logs", [])])
    if not curves:
        return []
    n = min(len(c) for c in curves)
    mean_curve = []
    for i in range(n):
        vals = [c[i] for c in curves if c[i] is not None]
        mean_curve.append(statistics.fmean(vals) if vals else None)
    return mean_curve


def save_comparison_report(
    name: str,
    results: Dict[str, Dict],
    configs: Optional[Dict[str, Dict]] = None,
    output_root: Optional[Path] = None,
    timestamp: bool = True,
) -> Path:
    """Evidência comparativa entre arquiteturas: curvas sobrepostas + tabela +
    barras agrupadas (substituindo o radar da Fig. 8, conforme o parecer)."""
    root = Path(output_root) if output_root else EVIDENCE_ROOT
    dir_name = (
        f"{name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}" if timestamp else name
    )
    out_dir = root / dir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- tabela comparativa ---
    lines = [f"# {name} — comparison", "", "| Architecture | T_g | Final val acc | Test acc | Final L1 |", "|---|---|---|---|---|"]
    for arch, result in results.items():
        row = dict(_summary_rows(result))
        lines.append(
            f"| {arch} | {_fmt(row.get('T_g'))} | {_fmt(row.get('Final val accuracy'))} | "
            f"{_fmt(row.get('Test accuracy'))} | {_fmt(row.get('Final L1 penalty'))} |"
        )
    lines.append("")
    (out_dir / "comparison.md").write_text("\n".join(lines), encoding="utf-8")

    if configs is not None:
        with open(out_dir / "configs.json", "w", encoding="utf-8") as f:
            json.dump(configs, f, indent=2)

    # --- curvas de validação sobrepostas (estilo Fig. 7 do artigo) ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for arch, result in results.items():
        curve = _mean_val_curve(result)
        if curve:
            epochs = list(range(1, len(curve) + 1))
            ax.plot(epochs, curve, label=arch, alpha=0.85)
    ax.axhline(0.95, linestyle="--", color="gray", linewidth=1, label="τ = 0.95")
    ax.set_xscale("log")
    ax.set_xlabel("Epoch (log scale)")
    ax.set_ylabel("Validation accuracy (mean across seeds)")
    ax.set_ylim(0, 1)
    ax.set_title(name)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "val_curves.png", dpi=150)
    fig.savefig(out_dir / "val_curves.pdf")
    plt.close(fig)

    # --- barras agrupadas de acurácia final (substitui o radar) ---
    archs = list(results.keys())

    def _scalar(value):
        if isinstance(value, dict):
            return value.get("mean")
        return value

    val_accs = [_scalar(dict(_summary_rows(r)).get("Final val accuracy")) for r in results.values()]
    test_accs = [_scalar(dict(_summary_rows(r)).get("Test accuracy")) for r in results.values()]

    x = range(len(archs))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([i - width / 2 for i in x], [v or 0 for v in val_accs], width, label="Validation")
    ax.bar([i + width / 2 for i in x], [v or 0 for v in test_accs], width, label="Test")
    ax.set_xticks(list(x))
    ax.set_xticklabels(archs, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.set_title(f"{name} — final accuracy")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "final_accuracy_bars.png", dpi=150)
    fig.savefig(out_dir / "final_accuracy_bars.pdf")
    plt.close(fig)

    return out_dir
