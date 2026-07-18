"""Gera retroativamente os diretórios de evidência (tabelas + gráficos) para
resultados brutos já existentes em pilot_results/ e sweep_results/.

Uso: `uv run python experiments/generate_evidence.py`
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.reporting import save_comparison_report, save_run_report

MODULAR = Path(__file__).parent / "modular_addition"


def _load(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    generated = []

    # --- piloto curto (multiseed por arquitetura) ---
    pilot = MODULAR / "pilot_results"
    short_names = ["dlg", "mlp_baseline", "ltn_baseline"]
    short = {n: _load(pilot / f"{n}.json") for n in short_names if (pilot / f"{n}.json").exists()}
    if short:
        for arch, result in short.items():
            generated.append(save_run_report(f"pilot200_{arch}", result, timestamp=False))
        generated.append(
            save_comparison_report("pilot200_comparison", short, timestamp=False)
        )

    # --- rodada longa ---
    long_names = ["dlg_long", "mlp_baseline_long"]
    long_ = {n: _load(pilot / f"{n}.json") for n in long_names if (pilot / f"{n}.json").exists()}
    if long_:
        for arch, result in long_.items():
            generated.append(save_run_report(f"pilot3000_{arch}", result, timestamp=False))
        generated.append(
            save_comparison_report("pilot3000_comparison", long_, timestamp=False)
        )

    # --- sweep (single-seed por config) ---
    sweep = MODULAR / "sweep_results"
    if sweep.exists():
        summary_path = sweep / "summary.json"
        summary = _load(summary_path) if summary_path.exists() else {}
        sweep_results = {}
        for config_file in sorted(sweep.glob("*.json")):
            if config_file.name == "summary.json":
                continue
            result = _load(config_file)
            config = summary.get(config_file.stem, {}).get("config")
            sweep_results[config_file.stem] = result
            generated.append(
                save_run_report(
                    f"sweep_{config_file.stem}", result, config=config, timestamp=False
                )
            )
        if sweep_results:
            generated.append(
                save_comparison_report("sweep_comparison", sweep_results, timestamp=False)
            )

    for path in generated:
        print("evidence:", path)


if __name__ == "__main__":
    main()
