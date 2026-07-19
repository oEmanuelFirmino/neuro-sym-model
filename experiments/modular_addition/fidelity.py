"""Fidelidade axiomática medida em instanciações não vistas no treino.

A "Fidelidade (%)" da Tabela 4 do artigo, medida de verdade: grau médio de
satisfação dos axiomas do domínio (comutatividade e identidade) instanciados
sobre pares de validação/teste — que o modelo nunca viu como axioma de treino.

Para comparabilidade entre arquiteturas (item m3 do parecer), a satisfação da
implicação é sempre computada com a semântica de Reichenbach/Product
(I(a,b) = 1 - a + a·b) sobre os graus de verdade previstos, independentemente
da T-norm usada no treino de cada modelo — a semântica de medição é a régua,
não o mecanismo.
"""

import statistics
from typing import Dict, List, Tuple

from src.neurosym.interpreter import Interpreter
from src.neurosym.logic import Atom, Constant


def _truth(interpreter: Interpreter, formula) -> float:
    result = interpreter.eval_formula(formula, {})
    return result._flatten(result.data)[0]


def _add_atom(a: int, b: int, c: int) -> Atom:
    return Atom("Add", [Constant(str(a)), Constant(str(b)), Constant(str(c))])


def axiom_fidelity(
    interpreter: Interpreter, pairs: List[Tuple[int, int]], p: int
) -> Dict:
    """Satisfação média dos axiomas sobre os `pairs` fornecidos (held-out).

    - comutatividade: I(t(Add(a,b,c)), t(Add(b,a,c))), c = (a+b) mod p;
    - identidade: t(Add(a,0,a)) para todo a do domínio.
    """
    commutativity = []
    for a, b in pairs:
        c = (a + b) % p
        t_ab = _truth(interpreter, _add_atom(a, b, c))
        t_ba = _truth(interpreter, _add_atom(b, a, c))
        commutativity.append(1.0 - t_ab + t_ab * t_ba)

    identity = [_truth(interpreter, _add_atom(a, 0, a)) for a in range(p)]

    def _agg(values):
        return {
            "mean": statistics.fmean(values) if values else None,
            "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
            "n": len(values),
        }

    overall = commutativity + identity
    return {
        "commutativity": _agg(commutativity),
        "identity": _agg(identity),
        "overall": _agg(overall),
    }
