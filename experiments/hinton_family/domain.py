"""Dataset de parentesco de Hinton (1986) — segundo domínio real (M2).

Árvore genealógica inglesa de Hinton, "Learning Distributed Representations
of Concepts" (1986): 12 pessoas, 5 casais, 3 gerações. Benchmark relacional
canônico do qual descendem os kinship benchmarks usados por NTP e ∂ILP.

Divisão de papéis alinhada à tese reposicionada do artigo:
- 8 predicados base TREINADOS de fatos: father, mother, husband, wife,
  brother, sister, son, daughter (com cobertura completa de negativos — lição
  da saturação da t-conorm registrada na Fase 4 do plano);
- 4 predicados derivados NUNCA treinados: uncle, aunt, nephew, niece,
  avaliados em tempo de consulta pela composição da regra com Product T-norms
  (DAG de prova), ex.: uncle(x,y) = ⋁_z [brother(x,z) ∧ (father(z,y) ∨ mother(z,y))].

Simplificação registrada: usamos parentesco consanguíneo estrito (tio =
irmão de progenitor), sem tios por afinidade — documentar no artigo.
"""

import random
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

from src.neurosym.logic import And, Atom, Constant, Formula, Or
from src.neurosym.tensor import Tensor

Fact = Tuple[Atom, float]

# --- árvore inglesa de Hinton (1986) ---
COUPLES = [  # (marido, esposa)
    ("christopher", "penelope"),
    ("andrew", "christine"),
    ("arthur", "margaret"),
    ("james", "victoria"),
    ("charles", "jennifer"),
]
CHILDREN = {  # (pai, mãe) -> filhos
    ("christopher", "penelope"): ["arthur", "victoria"],
    ("andrew", "christine"): ["james", "jennifer"],
    ("james", "victoria"): ["colin", "charlotte"],
}
MALES = {"christopher", "andrew", "arthur", "james", "charles", "colin"}
FEMALES = {"penelope", "christine", "margaret", "victoria", "jennifer", "charlotte"}

BASE_RELATIONS = [
    "father", "mother", "husband", "wife", "brother", "sister", "son", "daughter",
]
DERIVED_RELATIONS = ["uncle", "aunt", "nephew", "niece"]


def _gold_base() -> Dict[str, Set[Tuple[str, str]]]:
    """Deriva os pares verdadeiros de cada relação base da estrutura da árvore."""
    gold: Dict[str, Set[Tuple[str, str]]] = {r: set() for r in BASE_RELATIONS}

    for husband, wife in COUPLES:
        gold["husband"].add((husband, wife))
        gold["wife"].add((wife, husband))

    siblings: List[Tuple[str, str]] = []
    for (father, mother), kids in CHILDREN.items():
        for kid in kids:
            gold["father"].add((father, kid))
            gold["mother"].add((mother, kid))
            if kid in MALES:
                gold["son"].add((kid, father))
                gold["son"].add((kid, mother))
            else:
                gold["daughter"].add((kid, father))
                gold["daughter"].add((kid, mother))
        for a in kids:
            for b in kids:
                if a != b:
                    siblings.append((a, b))

    for a, b in siblings:
        if a in MALES:
            gold["brother"].add((a, b))
        else:
            gold["sister"].add((a, b))

    return gold


def _gold_derived() -> Dict[str, Set[Tuple[str, str, str]]]:
    """Pares derivados verdadeiros com o intermediário z do caminho de prova:
    uncle(x,y) via z: brother(x,z) ∧ parent(z,y); análogos para os demais."""
    base = _gold_base()
    parent = base["father"] | base["mother"]
    sibling = base["brother"] | base["sister"]

    gold: Dict[str, Set[Tuple[str, str, str]]] = {r: set() for r in DERIVED_RELATIONS}
    for x, z in sibling:
        for z2, y in parent:
            if z == z2:
                if x in MALES:
                    gold["uncle"].add((x, y, z))
                else:
                    gold["aunt"].add((x, y, z))
    for x, z in base["son"]:
        for z2, y in sibling:
            if z == z2:
                gold["nephew"].add((x, y, z))
    for x, z in base["daughter"]:
        for z2, y in sibling:
            if z == z2:
                gold["niece"].add((x, y, z))
    return gold


@dataclass
class FamilyData:
    entities: List[str]
    base_facts: List[Fact]           # cobertura completa: todo par rotulado
    flat_derived_facts: List[Fact]   # para os baselines planos treinados
    derived_gold: Dict[str, Set[Tuple[str, str, str]]]


def generate_family(seed: int = 0) -> FamilyData:
    rng = random.Random(seed)
    entities = sorted(MALES | FEMALES)
    gold_base = _gold_base()
    gold_derived = _gold_derived()

    base_facts: List[Fact] = []
    for relation in BASE_RELATIONS:
        positives = gold_base[relation]
        for x in entities:
            for y in entities:
                if x == y:
                    continue
                label = 1.0 if (x, y) in positives else 0.0
                base_facts.append(
                    (Atom(relation, [Constant(x), Constant(y)]), label)
                )

    flat_derived_facts: List[Fact] = []
    for relation in DERIVED_RELATIONS:
        positives = {(x, y) for x, y, _ in gold_derived[relation]}
        for x in entities:
            for y in entities:
                if x == y:
                    continue
                label = 1.0 if (x, y) in positives else 0.0
                flat_derived_facts.append(
                    (Atom(f"{relation}_flat", [Constant(x), Constant(y)]), label)
                )

    rng.shuffle(base_facts)
    rng.shuffle(flat_derived_facts)
    return FamilyData(
        entities=entities,
        base_facts=base_facts,
        flat_derived_facts=flat_derived_facts,
        derived_gold=gold_derived,
    )


def build_family_grounding_env(
    entities: List[str], embedding_dim: int, seed: int
) -> Dict[str, Tensor]:
    rng = random.Random(seed)
    return {
        name: Tensor(
            [[rng.uniform(-1.0, 1.0) for _ in range(embedding_dim)]],
            requires_grad=True,
        )
        for name in entities
    }


def _parent_or(z: str, y: str) -> Formula:
    return Or(
        Atom("father", [Constant(z), Constant(y)]),
        Atom("mother", [Constant(z), Constant(y)]),
    )


def _sibling_or(z: str, y: str) -> Formula:
    return Or(
        Atom("brother", [Constant(z), Constant(y)]),
        Atom("sister", [Constant(z), Constant(y)]),
    )


def derived_proof_formula(
    relation: str, x: str, y: str, entities: List[str]
) -> Formula:
    """Fórmula de prova grounded do predicado derivado (DAG de prova).

    uncle(x,y)  = ⋁_z brother(x,z) ∧ parent(z,y)
    aunt(x,y)   = ⋁_z sister(x,z)  ∧ parent(z,y)
    nephew(x,y) = ⋁_z son(x,z)      ∧ sibling(z,y)
    niece(x,y)  = ⋁_z daughter(x,z) ∧ sibling(z,y)
    """
    first_atom = {
        "uncle": "brother",
        "aunt": "sister",
        "nephew": "son",
        "niece": "daughter",
    }[relation]
    second = _parent_or if relation in ("uncle", "aunt") else _sibling_or

    branches: List[Formula] = []
    for z in entities:
        if z in (x, y):
            continue
        branches.append(
            And(Atom(first_atom, [Constant(x), Constant(z)]), second(z, y))
        )

    formula = branches[0]
    for branch in branches[1:]:
        formula = Or(formula, branch)
    return formula
