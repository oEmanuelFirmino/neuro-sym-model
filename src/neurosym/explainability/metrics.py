"""Métricas quantitativas de fidelidade da explicabilidade (item M5 do parecer).

O artigo alega "explicabilidade intrínseca" ilustrada por um único mapa de
calor; o parecer exige medi-la sobre N consultas. Este módulo implementa os
dois protocolos pedidos:

1. **Concentração do gradiente**: para uma consulta, que fração da massa total
   de gradiente (norma L1 sobre os embeddings de todas as constantes do
   domínio) cai sobre as constantes relevantes à consulta (seu "fecho
   transitivo" — no domínio de Adição Modular, as próprias constantes da
   tripla). Perto de 1.0 = explicação concentrada onde deveria.

2. **Deletion/Insertion**: zera-se progressivamente os embeddings na ordem de
   influência indicada pelo gradiente e mede-se a degradação do grau de
   verdade previsto (deletion), ou parte-se de tudo zerado e restaura-se na
   mesma ordem medindo a recuperação (insertion). A comparação com uma ordem
   aleatória é o teste de fidelidade: se a explicação por gradiente é fiel,
   deletar na ordem do gradiente degrada mais rápido (AUC menor) e inserir
   recupera mais rápido (AUC maior) do que na ordem aleatória.
"""

import copy
import random
import statistics
from typing import Dict, Iterable, List, Optional, Tuple

from src.neurosym.interpreter import Interpreter
from src.neurosym.logic import Formula


def compute_influences(interpreter: Interpreter, formula: Formula) -> Dict[str, float]:
    """Norma L1 do gradiente do grau de verdade em relação a cada embedding."""
    for tensor in interpreter.grounding_env.values():
        tensor.requires_grad = True
        tensor.zero_grad()

    truth = interpreter.eval_formula(formula, {}).sum()
    truth.backward()

    influences = {}
    for name, tensor in interpreter.grounding_env.items():
        if tensor.grad is not None:
            influences[name] = sum(
                abs(g) for g in tensor.grad._flatten(tensor.grad.data)
            )
        else:
            influences[name] = 0.0
    return influences


def concentration(
    influences: Dict[str, float], relevant: Iterable[str]
) -> Optional[float]:
    """Fração da massa de gradiente sobre as constantes relevantes à consulta."""
    total = sum(influences.values())
    if total <= 0:
        return None
    relevant_set = set(relevant)
    return sum(v for k, v in influences.items() if k in relevant_set) / total


def _eval_truth(interpreter: Interpreter, formula: Formula) -> float:
    result = interpreter.eval_formula(formula, {})
    return result._flatten(result.data)[0]


def deletion_curve(
    interpreter: Interpreter, formula: Formula, order: List[str]
) -> List[float]:
    """Grau de verdade após zerar cumulativamente os embeddings em `order`.

    O primeiro ponto é o valor sem nenhuma deleção; restaura tudo ao final.
    """
    saved = {
        name: copy.deepcopy(interpreter.grounding_env[name].data) for name in order
    }
    try:
        curve = [_eval_truth(interpreter, formula)]
        for name in order:
            tensor = interpreter.grounding_env[name]
            tensor.data = tensor.backend.apply_recursive(
                tensor.data, None, lambda _: 0.0
            )
            curve.append(_eval_truth(interpreter, formula))
        return curve
    finally:
        for name, data in saved.items():
            interpreter.grounding_env[name].data = data


def insertion_curve(
    interpreter: Interpreter, formula: Formula, order: List[str]
) -> List[float]:
    """Grau de verdade partindo de todos os embeddings de `order` zerados e
    restaurando-os cumulativamente; restaura tudo ao final."""
    saved = {
        name: copy.deepcopy(interpreter.grounding_env[name].data) for name in order
    }
    try:
        for name in order:
            tensor = interpreter.grounding_env[name]
            tensor.data = tensor.backend.apply_recursive(
                tensor.data, None, lambda _: 0.0
            )
        curve = [_eval_truth(interpreter, formula)]
        for name in order:
            interpreter.grounding_env[name].data = copy.deepcopy(saved[name])
            curve.append(_eval_truth(interpreter, formula))
        return curve
    finally:
        for name, data in saved.items():
            interpreter.grounding_env[name].data = data


def curve_auc(curve: List[float]) -> float:
    """AUC normalizada (média dos pontos) de uma curva de deletion/insertion."""
    return statistics.fmean(curve) if curve else 0.0


def evaluate_queries(
    interpreter: Interpreter,
    queries: List[Tuple[Formula, Iterable[str]]],
    seed: int = 0,
) -> Dict:
    """Roda os protocolos de fidelidade sobre N consultas e agrega.

    Para cada consulta `(formula, constantes_relevantes)`:
      - concentração do gradiente nas constantes relevantes;
      - AUC de deletion na ordem do gradiente vs. ordem aleatória;
      - AUC de insertion na ordem do gradiente vs. ordem aleatória.

    Fidelidade esperada de uma boa explicação: deletion_auc_gradient <
    deletion_auc_random e insertion_auc_gradient > insertion_auc_random.
    """
    rng = random.Random(seed)
    per_query = []

    for formula, relevant in queries:
        influences = compute_influences(interpreter, formula)
        order_gradient = sorted(influences, key=influences.get, reverse=True)
        order_random = list(influences.keys())
        rng.shuffle(order_random)

        record = {
            "query": repr(formula),
            "concentration": concentration(influences, relevant),
            "deletion_auc_gradient": curve_auc(
                deletion_curve(interpreter, formula, order_gradient)
            ),
            "deletion_auc_random": curve_auc(
                deletion_curve(interpreter, formula, order_random)
            ),
            "insertion_auc_gradient": curve_auc(
                insertion_curve(interpreter, formula, order_gradient)
            ),
            "insertion_auc_random": curve_auc(
                insertion_curve(interpreter, formula, order_random)
            ),
        }
        per_query.append(record)

    def _agg(key):
        values = [q[key] for q in per_query if q[key] is not None]
        if not values:
            return {"mean": None, "std": None, "n": 0}
        return {
            "mean": statistics.fmean(values),
            "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
            "n": len(values),
        }

    return {
        "n_queries": len(per_query),
        "concentration": _agg("concentration"),
        "deletion_auc_gradient": _agg("deletion_auc_gradient"),
        "deletion_auc_random": _agg("deletion_auc_random"),
        "insertion_auc_gradient": _agg("insertion_auc_gradient"),
        "insertion_auc_random": _agg("insertion_auc_random"),
        "per_query": per_query,
    }
