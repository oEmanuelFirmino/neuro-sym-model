"""O núcleo do batching: agrupa uma `List[Formula]` (todas totalmente grounded,
i.e. só `Constant` nas folhas -- o caso real de uso hoje, ver
`experiments/modular_addition/axioms.py`) por "assinatura" estrutural (mesmo
predicado/aridade/forma de árvore And/Or/Implies/Not) e empilha os índices de
constante de cada folha num único `LongTensor[B, L]` por grupo.

Isso é computado UMA VEZ antes do loop de épocas (facts/rules não mudam de época
para época) -- o custo por época passa a ser só o gather + forward/backward, não
mais o agrupamento em si.

`Forall`/`Exists` (quantificação sobre um domínio) ficam de fora desta fase: os
axiomas reais do projeto já são instanciados por exemplo de treino, não expressos
via `Forall` (ver docstring de `axioms.py`), então o alvo de batching que importa
de verdade é este loop de fatos/regras grounded.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

from src.neurosym.logic import And, Atom, Constant, Formula, Implies, Not, Or

Signature = Any  # tupla aninhada hasheável descrevendo a forma da árvore


def signature(formula: Formula) -> Signature:
    if isinstance(formula, Atom):
        return ("Atom", formula.predicate_name, len(formula.terms))
    if isinstance(formula, Not):
        return ("Not", signature(formula.formula))
    if isinstance(formula, And):
        return ("And", signature(formula.left), signature(formula.right))
    if isinstance(formula, Or):
        return ("Or", signature(formula.left), signature(formula.right))
    if isinstance(formula, Implies):
        return ("Implies", signature(formula.antecedent), signature(formula.consequent))
    raise NotImplementedError(
        f"torch_engine.compile não suporta fórmulas do tipo {type(formula).__name__} "
        "nesta fase (Forall/Exists ficam para uma fase futura de batching)."
    )


def leaves(formula: Formula) -> List[str]:
    """Nomes das constantes nas folhas, na mesma ordem de travessia usada por
    `BatchedInterpreter._eval` -- as duas funções DEVEM percorrer a árvore na
    mesma ordem, senão as colunas de `leaf_idx` acabam ligadas ao termo errado."""
    if isinstance(formula, Atom):
        names = []
        for term in formula.terms:
            if not isinstance(term, Constant):
                raise TypeError(
                    "torch_engine espera fórmulas totalmente grounded (só Constant "
                    f"nas folhas); encontrado {type(term).__name__} em {formula}."
                )
            names.append(term.name)
        return names
    if isinstance(formula, Not):
        return leaves(formula.formula)
    if isinstance(formula, And) or isinstance(formula, Or):
        return leaves(formula.left) + leaves(formula.right)
    if isinstance(formula, Implies):
        return leaves(formula.antecedent) + leaves(formula.consequent)
    raise NotImplementedError(f"Tipo de fórmula não suportado: {type(formula).__name__}")


@dataclass
class CompiledGroup:
    signature: Signature
    template: Formula
    leaf_idx: torch.Tensor  # LongTensor[B, L]
    targets: Optional[torch.Tensor] = None  # Tensor[B], só quando compilado com alvos


def compile_formulas(
    formulas: List[Formula],
    vocab: Dict[str, int],
    targets: Optional[List[float]] = None,
) -> List[CompiledGroup]:
    """Agrupa `formulas` por assinatura estrutural. Chame separadamente para
    `facts` (com `targets`) e para `rules` (sem `targets`) -- um Atom de fato e um
    Atom de axioma têm a mesma assinatura e não podem cair no mesmo grupo, senão
    l_data/l_semantic se misturam."""
    if targets is not None and len(targets) != len(formulas):
        raise ValueError("targets deve ter o mesmo tamanho de formulas.")

    groups: Dict[Signature, Dict[str, list]] = {}
    for i, formula in enumerate(formulas):
        sig = signature(formula)
        leaf_names = leaves(formula)
        bucket = groups.setdefault(sig, {"template": formula, "rows": [], "target_rows": []})
        try:
            row = [vocab[name] for name in leaf_names]
        except KeyError as e:
            raise KeyError(f"Constante desconhecida no grounding: {e}") from e
        bucket["rows"].append(row)
        if targets is not None:
            bucket["target_rows"].append(targets[i])

    compiled = []
    for sig, bucket in groups.items():
        leaf_idx = torch.tensor(bucket["rows"], dtype=torch.long)
        group_targets = (
            torch.tensor(bucket["target_rows"], dtype=torch.float32)
            if targets is not None
            else None
        )
        compiled.append(CompiledGroup(sig, bucket["template"], leaf_idx, group_targets))
    return compiled
