"""Axiomas de comutatividade pro grafo de conhecimento sintético
(`experiments/torch_scale/dataset.py`): cada relação `Rj(e1,e2,e3) :=
(e1+e2+j) mod num_entities == e3` é comutativa em (e1,e2) (soma é comutativa),
então `Rj(a,b,c) -> Rj(b,a,c)` é um axioma válido pra qualquer fato positivo de
treino -- mesma ideia de `experiments/modular_addition/axioms.py`'s
`commutativity_axioms`, generalizada pras várias relações do domínio sintético.

Só se aplica a Atoms de aridade 3 (exatamente 2 argumentos de entrada + 1 de
saída) -- pra aridade maior, "comutatividade" exigiria considerar todas as
permutações dos argumentos de entrada, fora do escopo do que estamos testando
agora (o sweep de hiperparâmetros do grokking usa arity=3).
"""

from typing import List, Sequence

from experiments.torch_scale.dataset import Fact
from src.neurosym.logic import Atom, Constant, Formula, Implies


def build_commutativity_axioms(facts: List[Fact], relations: Sequence[str]) -> List[Formula]:
    """`Rj(a,b,c) -> Rj(b,a,c)` para cada fato positivo de treino de uma relação
    em `relations`. Instanciado por exemplo de treino (não via `Forall`), mesmo
    raciocínio de `experiments/modular_addition/axioms.py`."""
    relations_set = set(relations)
    axioms: List[Formula] = []
    for atom, target in facts:
        if target < 0.5:
            continue
        if atom.predicate_name not in relations_set:
            continue
        if len(atom.terms) != 3:
            continue
        a, b, c = (term.name for term in atom.terms)
        axioms.append(
            Implies(
                Atom(atom.predicate_name, [Constant(a), Constant(b), Constant(c)]),
                Atom(atom.predicate_name, [Constant(b), Constant(a), Constant(c)]),
            )
        )
    return axioms
