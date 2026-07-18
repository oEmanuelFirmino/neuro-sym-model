"""Axiomas algébricos do domínio de Adição Modular (artigo, Seção 5.2.1).

Os axiomas são instanciados (grounded) diretamente sobre os pares de treino,
em vez de quantificados com `Forall` sobre todo o domínio. Com p=97, um
`Forall` de duas variáveis livres exigiria iterar 97² combinações por
avaliação (o `Interpreter.eval_formula` atual não vetoriza quantificadores),
o que é proibitivo dado que as regras são reavaliadas por completo a cada
época. Instanciar por par de treino é computacionalmente tratável e é a
leitura mais direta do funcional L_semantic do artigo (Seção 4.4: média sobre
o conjunto de axiomas A) quando A é entendido como o conjunto de instâncias
grounded do batch de treino.
"""

from typing import List, Tuple

from src.neurosym.logic import Atom, Constant, Formula, Implies


def commutativity_axioms(pairs: List[Tuple[int, int]], p: int) -> List[Formula]:
    """Add(a,b,c) -> Add(b,a,c) para cada par de treino (a,b), c = (a+b) mod p."""
    axioms: List[Formula] = []
    for a, b in pairs:
        c = (a + b) % p
        axioms.append(
            Implies(
                Atom("Add", [Constant(str(a)), Constant(str(b)), Constant(str(c))]),
                Atom("Add", [Constant(str(b)), Constant(str(a)), Constant(str(c))]),
            )
        )
    return axioms


def identity_axioms(p: int) -> List[Formula]:
    """Add(a,0,a) para todo a no domínio -- barato (O(p)), mantido como Forall
    implícito via enumeração direta (não depende do split de treino)."""
    return [
        Atom("Add", [Constant(str(a)), Constant("0"), Constant(str(a))])
        for a in range(p)
    ]
