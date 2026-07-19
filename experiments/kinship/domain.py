"""Domínio de parentesco/ancestral com inferência composta via DAG de prova.

Este é o segundo domínio (Fase 4), promovido a experimento central pelo
redirecionamento estratégico do artigo (explicabilidade + consistência lógica):

- O predicado base `parent(x,y)` é aprendido de fatos (arestas de árvores
  genealógicas pequenas).
- O predicado derivado `ancestor(x,z)` NÃO é uma rede treinada: é avaliado em
  tempo de consulta compondo dinamicamente a fórmula de prova

      ancestor_d(x,z) = parent(x,z) ∨ ⋁_m [ parent(x,m) ∧ ancestor_{d-1}(m,z) ]

  com `And`/`Or`/`Atom` existentes, avaliada pelo Interpreter com Product
  T-norms. Isso materializa a "execução dinâmica via DAG" do artigo: o grafo
  computacional da consulta percorre as entidades intermediárias do caminho,
  então o gradiente da resposta alcança o fecho transitivo de verdade — e a
  explicabilidade estrutural vira uma propriedade mensurável da arquitetura.

- Para contraste, um predicado plano `ancestor_flat(x,z)` (MLP treinado
  diretamente em pares ancestral/não-ancestral, sem composição) responde às
  mesmas consultas sem estrutura: o gradiente só toca x e z, nunca os
  intermediários. A diferença entre os dois é a evidência arquitetural.
"""

import random
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

from src.neurosym.logic import And, Atom, Constant, Formula, Or
from src.neurosym.tensor import Tensor

Fact = Tuple[Atom, float]
Edge = Tuple[str, str]

# Três linhagens de profundidade 4 + uma linhagem curta de distratores.
DEFAULT_EDGES: List[Edge] = [
    ("a0", "a1"), ("a1", "a2"), ("a2", "a3"),
    ("b0", "b1"), ("b1", "b2"), ("b2", "b3"),
    ("c0", "c1"),
]


@dataclass
class KinshipData:
    entities: List[str]
    edges: List[Edge]
    parent_facts: List[Fact]
    ancestor_flat_facts: List[Fact]
    # pares (x, z, intermediários) com caminho de comprimento >= 2
    chained_queries: List[Tuple[str, str, Set[str]]] = field(default_factory=list)


def _closure(edges: List[Edge]) -> Dict[str, Set[str]]:
    """Fecho transitivo: para cada x, o conjunto de descendentes alcançáveis."""
    children: Dict[str, Set[str]] = {}
    for parent, child in edges:
        children.setdefault(parent, set()).add(child)

    def reach(x: str, seen: Set[str]) -> Set[str]:
        result = set()
        for c in children.get(x, ()):  # DFS simples; grafos pequenos e acíclicos
            if c not in seen:
                result.add(c)
                result |= reach(c, seen | {c})
        return result

    return {x: reach(x, {x}) for x in children}


def _path_intermediates(edges: List[Edge], x: str, z: str) -> Set[str]:
    """Entidades intermediárias em algum caminho x -> ... -> z."""
    children: Dict[str, Set[str]] = {}
    for parent, child in edges:
        children.setdefault(parent, set()).add(child)

    intermediates: Set[str] = set()

    def dfs(node: str, acc: List[str]):
        if node == z:
            intermediates.update(acc)
            return
        for c in children.get(node, ()):
            dfs(c, acc + [c] if c != z else acc)

    dfs(x, [])
    intermediates.discard(x)
    intermediates.discard(z)
    return intermediates


def generate_kinship(
    edges: List[Edge] = None,
    negatives_per_positive: int = 2,
    seed: int = 0,
) -> KinshipData:
    edges = list(edges) if edges is not None else list(DEFAULT_EDGES)
    rng = random.Random(seed)

    entities = sorted({e for pair in edges for e in pair})
    edge_set = set(edges)
    closure = _closure(edges)

    # --- fatos do predicado base parent ---
    parent_facts: List[Fact] = [
        (Atom("parent", [Constant(x), Constant(y)]), 1.0) for x, y in edges
    ]
    non_edges = [
        (x, y)
        for x in entities
        for y in entities
        if x != y and (x, y) not in edge_set
    ]
    rng.shuffle(non_edges)
    for x, y in non_edges[: len(edges) * negatives_per_positive]:
        parent_facts.append((Atom("parent", [Constant(x), Constant(y)]), 0.0))

    # --- fatos do predicado plano ancestor_flat (baseline sem estrutura) ---
    ancestor_pairs = [(x, z) for x, descendants in closure.items() for z in descendants]
    ancestor_set = set(ancestor_pairs)
    ancestor_flat_facts: List[Fact] = [
        (Atom("ancestor_flat", [Constant(x), Constant(z)]), 1.0)
        for x, z in ancestor_pairs
    ]
    non_ancestor = [
        (x, z)
        for x in entities
        for z in entities
        if x != z and (x, z) not in ancestor_set
    ]
    rng.shuffle(non_ancestor)
    for x, z in non_ancestor[: len(ancestor_pairs) * negatives_per_positive]:
        ancestor_flat_facts.append(
            (Atom("ancestor_flat", [Constant(x), Constant(z)]), 0.0)
        )

    # --- consultas com encadeamento (caminho >= 2, i.e., há intermediários) ---
    chained = []
    for x, z in ancestor_pairs:
        intermediates = _path_intermediates(edges, x, z)
        if intermediates:
            chained.append((x, z, intermediates))

    return KinshipData(
        entities=entities,
        edges=edges,
        parent_facts=parent_facts,
        ancestor_flat_facts=ancestor_flat_facts,
        chained_queries=chained,
    )


def build_kinship_grounding_env(
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


def ancestor_proof_formula(
    x: str, z: str, entities: List[str], depth: int
) -> Formula:
    """Constrói dinamicamente a fórmula de prova de ancestor(x,z) até `depth`.

    ancestor_1(x,z) = parent(x,z)
    ancestor_d(x,z) = parent(x,z) ∨ ⋁_{m ≠ x,z} [ parent(x,m) ∧ ancestor_{d-1}(m,z) ]

    A fórmula resultante é avaliada pelo Interpreter comum (Product T-norm),
    formando o DAG de prova pelo qual o gradiente flui até os intermediários.
    """
    base: Formula = Atom("parent", [Constant(x), Constant(z)])
    if depth <= 1:
        return base

    formula = base
    for m in entities:
        if m == x or m == z:
            continue
        branch = And(
            Atom("parent", [Constant(x), Constant(m)]),
            ancestor_proof_formula(m, z, entities, depth - 1),
        )
        formula = Or(formula, branch)
    return formula
