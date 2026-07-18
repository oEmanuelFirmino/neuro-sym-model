"""Dataset e grounding do domínio de Adição Modular (Z_p) -- Fase 3 do plano
de correções do artigo (docs/plano-correcoes-artigo.md).
"""

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

from src.neurosym.logic import Atom, Constant
from src.neurosym.tensor import Tensor

Fact = Tuple[Atom, float]


@dataclass
class ModularAdditionData:
    p: int
    train_pairs: List[Tuple[int, int]]
    val_pairs: List[Tuple[int, int]]
    test_pairs: List[Tuple[int, int]]
    train_facts: List[Fact]
    val_facts: List[Fact]
    test_facts: List[Fact]


def build_grounding_env(p: int, embedding_dim: int, seed: int) -> Dict[str, Tensor]:
    """Embeddings treináveis para as constantes {0, ..., p-1}.

    Usa `random.uniform` (seedável) em vez do `hash()` de string usado por
    `KnowledgeBaseLoader.load_domain` -- aquele não é fixado por
    `random.seed()`/`np.random.seed()` e quebraria a reprodutibilidade do
    runner multi-seed (gap sinalizado na Fase 2, ver docs/plano-correcoes-artigo.md).
    """
    rng = random.Random(seed)
    return {
        str(i): Tensor(
            [[rng.uniform(-1.0, 1.0) for _ in range(embedding_dim)]],
            requires_grad=True,
        )
        for i in range(p)
    }


def _add_mod(a: int, b: int, p: int) -> int:
    return (a + b) % p


def _sample_negative_cs(
    a: int, b: int, p: int, rng: random.Random, k: int
) -> List[int]:
    """Amostra até k valores de c incorretos para o par (a,b) (nunca o correto)."""
    correct = _add_mod(a, b, p)
    available = p - 1
    if available <= 0:
        return []
    negatives = set()
    while len(negatives) < min(k, available):
        candidate = rng.randrange(p)
        if candidate != correct:
            negatives.add(candidate)
    return list(negatives)


def _facts_for_pairs(
    pairs: List[Tuple[int, int]],
    p: int,
    rng: random.Random,
    negatives_per_positive: int,
) -> List[Fact]:
    """Um fato positivo (c correto, rótulo 1.0) e `negatives_per_positive`
    fatos negativos (c incorreto, rótulo 0.0) por par -- sem negativos, um
    preditor trivial que sempre responde "verdadeiro" teria acurácia perfeita.
    """
    facts: List[Fact] = []
    for a, b in pairs:
        correct = _add_mod(a, b, p)
        facts.append(
            (Atom("Add", [Constant(str(a)), Constant(str(b)), Constant(str(correct))]), 1.0)
        )
        for neg_c in _sample_negative_cs(a, b, p, rng, negatives_per_positive):
            facts.append(
                (Atom("Add", [Constant(str(a)), Constant(str(b)), Constant(str(neg_c))]), 0.0)
            )
    return facts


def generate_split(
    p: int,
    seed: int,
    train_frac: float = 0.30,
    val_frac: float = 0.35,
    negatives_per_positive: int = 1,
) -> ModularAdditionData:
    """Gera as p² triplas corretas (a,b,(a+b) mod p) e as particiona em
    treino/validação/teste genuinamente separados.

    O artigo original usa 30% treino / 70% "validação", mas seleciona o
    modelo nesse mesmo conjunto de 70% sem holdout de teste real -- exatamente
    o que o item m1 do parecer do orientador aponta como impreciso. Aqui a
    fração de treino pequena é preservada por padrão (é o que induz o efeito
    de grokking), mas o restante é dividido em validação (usada para
    acompanhar a curva/T_g durante o treino) e teste (nunca visto até a
    avaliação final).
    """
    if not (0 < train_frac < 1) or not (0 < val_frac < 1) or train_frac + val_frac >= 1:
        raise ValueError("train_frac + val_frac deve ser < 1 (o restante vira teste).")

    rng = random.Random(seed)
    all_pairs = [(a, b) for a in range(p) for b in range(p)]
    rng.shuffle(all_pairs)

    n_total = len(all_pairs)
    n_train = int(n_total * train_frac)
    n_val = int(n_total * val_frac)

    train_pairs = all_pairs[:n_train]
    val_pairs = all_pairs[n_train : n_train + n_val]
    test_pairs = all_pairs[n_train + n_val :]

    train_facts = _facts_for_pairs(train_pairs, p, rng, negatives_per_positive)
    val_facts = _facts_for_pairs(val_pairs, p, rng, negatives_per_positive)
    test_facts = _facts_for_pairs(test_pairs, p, rng, negatives_per_positive)

    return ModularAdditionData(
        p=p,
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        test_pairs=test_pairs,
        train_facts=train_facts,
        val_facts=val_facts,
        test_facts=test_facts,
    )
