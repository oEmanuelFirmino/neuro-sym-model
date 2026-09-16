"""Gerador de domínio sintético domain-agnostic, para validar o `torch_engine`
batched numa escala próxima de um cenário real (milhares+ de entidades/fatos).

Kinship/Hinton Family são árvores de família pequenas e fixas -- não escalam.
Modular Addition (`experiments/modular_addition/dataset.py`) escala via `p`, mas
fica presa à relação de soma. Este gerador generaliza a mesma ideia: uma relação
por "deslocamento" `j`,
`Rj(e_1,...,e_{arity-1}, e_arity) := (soma(e_1..e_{arity-1}) + j) mod
num_entities == e_arity`, com `num_entities`/`num_relations`/`arity` livres. É um
domínio de *benchmark de escala*, não uma nova alegação científica sobre DLG --
a validação real do método continua em Modular Addition/Kinship/Hinton, na
escala que cada um suporta.

Atenção ao espaço de entrada: com `arity-1` argumentos de entrada, só existem
`num_entities**(arity-1)` tuplas de entrada distintas por relação -- `arity=2`
(função unária) permite no máximo `num_entities` fatos positivos distintos por
relação, o que é pouco para uma validação de "milhares de fatos". O padrão
`arity=3` (2 entradas + 1 saída, como `Add(a,b,c)` da Adição Modular) dá
`num_entities**2`, com espaço suficiente para dezenas de milhares de fatos já
com poucas centenas de entidades.

Reaproveita o padrão de negative sampling e de split por tupla-de-entrada (não
por fato solto, para não vazar a mesma tupla entre train/val/test com rótulos
diferentes) de `experiments/modular_addition/dataset.py`'s
`_sample_negative_cs`/`generate_split`.
"""

import random
from dataclasses import dataclass
from typing import List, Tuple

from src.neurosym.logic import Atom, Constant

Fact = Tuple[Atom, float]


@dataclass
class SyntheticRelationalData:
    num_entities: int
    arity: int
    relations: List[str]
    train_facts: List[Fact]
    val_facts: List[Fact]
    test_facts: List[Fact]


def _relation_name(j: int) -> str:
    return f"R{j}"


def _correct_last_arg(args: Tuple[int, ...], j: int, num_entities: int) -> int:
    return (sum(args) + j) % num_entities


def _sample_negatives(
    correct: int, num_entities: int, rng: random.Random, k: int
) -> List[int]:
    available = num_entities - 1
    if available <= 0:
        return []
    negatives = set()
    while len(negatives) < min(k, available):
        candidate = rng.randrange(num_entities)
        if candidate != correct:
            negatives.add(candidate)
    return list(negatives)


def _sample_unique_input_tuples(
    num_entities: int, arity: int, count: int, rng: random.Random
) -> List[Tuple[int, ...]]:
    """Amostra `count` tuplas de entrada DISTINTAS (rejeitando repetidas), para
    que o split train/val/test abaixo (uma fatia contígua desta lista) nunca
    tenha a mesma tupla em duas partições com rótulos diferentes -- ao contrário
    de sortear cada tupla independentemente (com reposição), que garante
    colisões pelo paradoxo do aniversário em num_positive_facts realistas e
    vazaria a mesma tupla entre splits."""
    input_size = arity - 1
    max_possible = num_entities**input_size
    if count > max_possible:
        raise ValueError(
            f"num_positive_facts={count} excede o número de tuplas de entrada "
            f"distintas possíveis para arity={arity}/num_entities={num_entities} "
            f"({max_possible})."
        )
    seen = set()
    while len(seen) < count:
        seen.add(tuple(rng.randrange(num_entities) for _ in range(input_size)))
    return list(seen)


def _facts_for_tuples(
    relation_name: str,
    input_tuples: List[Tuple[int, ...]],
    j: int,
    num_entities: int,
    rng: random.Random,
    negatives_per_positive: int,
) -> List[Fact]:
    facts: List[Fact] = []
    for args in input_tuples:
        correct = _correct_last_arg(args, j, num_entities)
        all_args = [str(a) for a in args] + [str(correct)]
        facts.append(
            (Atom(relation_name, [Constant(a) for a in all_args]), 1.0)
        )
        for neg in _sample_negatives(correct, num_entities, rng, negatives_per_positive):
            neg_args = [str(a) for a in args] + [str(neg)]
            facts.append(
                (Atom(relation_name, [Constant(a) for a in neg_args]), 0.0)
            )
    return facts


def generate_synthetic_domain(
    num_entities: int,
    num_relations: int = 1,
    arity: int = 3,
    num_positive_facts: int = 10_000,
    negatives_per_positive: int = 1,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    seed: int = 0,
) -> SyntheticRelationalData:
    if arity < 2:
        raise ValueError("arity deve ser >= 2 (>=1 argumento de entrada + 1 de saída).")
    if not (0 < train_frac < 1) or not (0 < val_frac < 1) or train_frac + val_frac >= 1:
        raise ValueError("train_frac + val_frac deve ser < 1 (o restante vira teste).")

    rng = random.Random(seed)
    relations = [_relation_name(j) for j in range(num_relations)]
    train_facts: List[Fact] = []
    val_facts: List[Fact] = []
    test_facts: List[Fact] = []

    for j, name in enumerate(relations):
        input_tuples = _sample_unique_input_tuples(num_entities, arity, num_positive_facts, rng)
        rng.shuffle(input_tuples)
        n_total = len(input_tuples)
        n_train = int(n_total * train_frac)
        n_val = int(n_total * val_frac)

        train_facts += _facts_for_tuples(
            name, input_tuples[:n_train], j, num_entities, rng, negatives_per_positive
        )
        val_facts += _facts_for_tuples(
            name, input_tuples[n_train : n_train + n_val], j, num_entities, rng, negatives_per_positive
        )
        test_facts += _facts_for_tuples(
            name, input_tuples[n_train + n_val :], j, num_entities, rng, negatives_per_positive
        )

    return SyntheticRelationalData(
        num_entities=num_entities,
        arity=arity,
        relations=relations,
        train_facts=train_facts,
        val_facts=val_facts,
        test_facts=test_facts,
    )


def entity_names(num_entities: int) -> List[str]:
    return [str(i) for i in range(num_entities)]
