"""Experimento central do artigo reposicionado: explicabilidade e consistência
lógica na inferência composta via DAG de prova (domínio de parentesco).

Protocolo:
1. Treina os predicados `parent` e `ancestor_flat` (embeddings compartilhados)
   sobre os fatos do domínio. `ancestor` composto NUNCA é treinado — é derivado
   em tempo de consulta pela fórmula de prova (Product T-norm sobre o DAG).
2. **Consistência lógica / transferência de raciocínio**: mede se a inferência
   composta discrimina pares ancestral verdadeiros de falsos usando apenas o
   predicado base aprendido — raciocínio dedutivo emergindo da composição, sem
   supervisão direta do conceito derivado.
3. **Explicabilidade arquitetural**: sobre as consultas com encadeamento
   (caminho >= 2), compara a inferência composta com o predicado plano em:
   - fração da massa de gradiente nos intermediários do caminho (a "assinatura"
     do raciocínio — estruturalmente zero no plano);
   - efeito de deletar os intermediários sobre o grau de verdade previsto
     (dependência causal do caminho de prova);
   - concentração no fecho (constantes da consulta + intermediários).

Evidência salva via experiments/reporting + JSON dedicado.

Uso: `uv run python experiments/kinship/run_kinship.py`
"""

import json
import logging
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.neurosym.tensor.backend import set_backend

set_backend("numpy")
logging.disable(logging.CRITICAL)

import random

from experiments.kinship.domain import (
    ancestor_proof_formula,
    build_kinship_grounding_env,
    generate_kinship,
)
from experiments.reporting import EVIDENCE_ROOT
from src.neurosym.explainability.metrics import (
    compute_influences,
    concentration,
    deletion_curve,
)
from src.neurosym.interpreter import Interpreter
from src.neurosym.logic import Atom, Constant
from src.neurosym.module.module import Linear, ReLU, Sequential, Sigmoid
from src.neurosym.training.optimizer import AdamW
from src.neurosym.training.trainer import Trainer

SEED = 0
EMBEDDING_DIM = 8
HIDDEN = 16
EPOCHS = 500
PROOF_DEPTH = 3  # cobre os caminhos mais longos do domínio (2 intermediários)


def _predicate(in_features: int) -> Sequential:
    return Sequential(
        Linear(in_features, HIDDEN), ReLU(), Linear(HIDDEN, 1), Sigmoid()
    )


def _truth(interpreter: Interpreter, formula) -> float:
    result = interpreter.eval_formula(formula, {})
    return result._flatten(result.data)[0]


def _agg(values):
    if not values:
        return {"mean": None, "std": None, "n": 0}
    return {
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "n": len(values),
    }


def main():
    random.seed(SEED)
    import numpy as np

    np.random.seed(SEED)

    # negatives_per_positive alto o bastante para cobrir TODOS os não-arcos:
    # com Product t-conorm, o OR da fórmula de prova agrega ~|E| ramos; ramos
    # de pares nunca vistos no treino pontuam ~0.5 e saturam a disjunção
    # (1 - prod(1-x_i) -> 1). A composição dedutiva exige o predicado base
    # calibrado em todo o domínio — limitação metodológica a registrar no texto.
    data = generate_kinship(seed=SEED, negatives_per_positive=20)
    grounding_env = build_kinship_grounding_env(data.entities, EMBEDDING_DIM, SEED)
    predicate_map = {
        "parent": _predicate(2 * EMBEDDING_DIM),
        "ancestor_flat": _predicate(2 * EMBEDDING_DIM),
    }
    interpreter = Interpreter(predicate_map, grounding_env)

    all_params = list(grounding_env.values())
    for model in predicate_map.values():
        all_params.extend(model.parameters())

    trainer = Trainer(
        interpreter,
        AdamW(all_params, lr=1e-2, weight_decay=1e-3),
        epochs=EPOCHS,
    )

    print(f"=== treinando parent + ancestor_flat ({EPOCHS} épocas) ===", flush=True)
    t0 = time.time()
    trainer.fit(rules=[], facts=data.parent_facts + data.ancestor_flat_facts)
    print(f"  treino: {time.time() - t0:.1f}s", flush=True)

    parent_train_acc = trainer.evaluate_accuracy(data.parent_facts)
    flat_train_acc = trainer.evaluate_accuracy(data.ancestor_flat_facts)
    print(f"  acc parent: {parent_train_acc:.3f} | acc ancestor_flat: {flat_train_acc:.3f}", flush=True)

    # --- 2. consistência lógica: a composição discrimina sem treino direto? ---
    positive_pairs = [(x, z) for x, z, _ in data.chained_queries]
    ancestor_set = set(positive_pairs) | set(data.edges)
    negative_pairs = [
        (x, z)
        for x in data.entities
        for z in data.entities
        if x != z and (x, z) not in ancestor_set
    ][: len(positive_pairs) * 2]

    composed_pos = [
        _truth(interpreter, ancestor_proof_formula(x, z, data.entities, PROOF_DEPTH))
        for x, z in positive_pairs
    ]
    composed_neg = [
        _truth(interpreter, ancestor_proof_formula(x, z, data.entities, PROOF_DEPTH))
        for x, z in negative_pairs
    ]
    flat_pos = [
        _truth(interpreter, Atom("ancestor_flat", [Constant(x), Constant(z)]))
        for x, z in positive_pairs
    ]
    flat_neg = [
        _truth(interpreter, Atom("ancestor_flat", [Constant(x), Constant(z)]))
        for x, z in negative_pairs
    ]

    consistency = {
        "composed_truth_positives": _agg(composed_pos),
        "composed_truth_negatives": _agg(composed_neg),
        "flat_truth_positives": _agg(flat_pos),
        "flat_truth_negatives": _agg(flat_neg),
        "note": (
            "composed nunca foi treinado em pares ancestral; discriminação "
            "positiva/negativa emerge da composição do predicado parent aprendido"
        ),
    }
    print("  consistência:", json.dumps({k: v for k, v in consistency.items() if k != 'note'}, indent=2), flush=True)

    # --- 3. explicabilidade arquitetural sobre consultas encadeadas ---
    per_query = []
    for x, z, intermediates in data.chained_queries:
        composed = ancestor_proof_formula(x, z, data.entities, PROOF_DEPTH)
        flat = Atom("ancestor_flat", [Constant(x), Constant(z)])
        closure = {x, z} | intermediates

        inf_composed = compute_influences(interpreter, composed)
        inf_flat = compute_influences(interpreter, flat)

        total_c = sum(inf_composed.values())
        total_f = sum(inf_flat.values())
        mid_mass_composed = (
            sum(inf_composed[m] for m in intermediates) / total_c if total_c else None
        )
        mid_mass_flat = (
            sum(inf_flat[m] for m in intermediates) / total_f if total_f else None
        )

        # dependência causal: deletar SÓ os intermediários
        order = sorted(intermediates)
        del_composed = deletion_curve(interpreter, composed, order)
        del_flat = deletion_curve(interpreter, flat, order)

        per_query.append(
            {
                "query": f"ancestor({x},{z})",
                "intermediates": sorted(intermediates),
                "intermediate_mass_composed": mid_mass_composed,
                "intermediate_mass_flat": mid_mass_flat,
                "concentration_composed": concentration(inf_composed, closure),
                "concentration_flat": concentration(inf_flat, closure),
                "deletion_delta_composed": del_composed[0] - del_composed[-1],
                "deletion_delta_flat": del_flat[0] - del_flat[-1],
            }
        )

    explainability = {
        "n_queries": len(per_query),
        "intermediate_mass_composed": _agg(
            [q["intermediate_mass_composed"] for q in per_query]
        ),
        "intermediate_mass_flat": _agg([q["intermediate_mass_flat"] for q in per_query]),
        "concentration_composed": _agg([q["concentration_composed"] for q in per_query]),
        "concentration_flat": _agg([q["concentration_flat"] for q in per_query]),
        "deletion_delta_composed": _agg([q["deletion_delta_composed"] for q in per_query]),
        "deletion_delta_flat": _agg([q["deletion_delta_flat"] for q in per_query]),
        "per_query": per_query,
    }
    print("  explicabilidade:", json.dumps({k: v for k, v in explainability.items() if k != 'per_query'}, indent=2), flush=True)

    # --- evidência ---
    out_dir = EVIDENCE_ROOT / "kinship_proof_dag"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "seed": SEED,
            "embedding_dim": EMBEDDING_DIM,
            "hidden": HIDDEN,
            "epochs": EPOCHS,
            "proof_depth": PROOF_DEPTH,
            "edges": data.edges,
        },
        "train_accuracy": {
            "parent": parent_train_acc,
            "ancestor_flat": flat_train_acc,
        },
        "logical_consistency": consistency,
        "explainability": explainability,
    }
    with open(out_dir / "kinship_report.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    lines = [
        "# Kinship — proof-DAG composed inference vs flat predicate",
        "",
        "| Metric | Composed (proof DAG) | Flat predicate |",
        "|---|---|---|",
    ]

    def _cell(stats):
        return (
            f"{stats['mean']:.4f} ± {stats['std']:.4f}"
            if stats["mean"] is not None
            else "—"
        )

    lines.append(
        f"| Truth on true ancestor pairs | {_cell(consistency['composed_truth_positives'])} | {_cell(consistency['flat_truth_positives'])} |"
    )
    lines.append(
        f"| Truth on false ancestor pairs | {_cell(consistency['composed_truth_negatives'])} | {_cell(consistency['flat_truth_negatives'])} |"
    )
    lines.append(
        f"| Gradient mass on path intermediates | {_cell(explainability['intermediate_mass_composed'])} | {_cell(explainability['intermediate_mass_flat'])} |"
    )
    lines.append(
        f"| Concentration on closure | {_cell(explainability['concentration_composed'])} | {_cell(explainability['concentration_flat'])} |"
    )
    lines.append(
        f"| Δ truth after deleting intermediates | {_cell(explainability['deletion_delta_composed'])} | {_cell(explainability['deletion_delta_flat'])} |"
    )
    lines += [
        "",
        "Composed inference is never trained on ancestor pairs: discrimination "
        "and path-dependence emerge from composing the learned parent predicate "
        "through the proof DAG (Product T-norms).",
        "",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("evidence:", out_dir, flush=True)


if __name__ == "__main__":
    main()
