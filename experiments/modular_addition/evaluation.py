"""Métrica de avaliação específica do domínio de Adição Modular.

`Trainer.evaluate_accuracy` genérica compara um único fato contra um limiar
binário -- insuficiente aqui, porque isso permitiria a um preditor trivial
("sempre responda >= 0.5") acertar todo fato positivo sem ter aprendido a
função de adição. A métrica correta para uma relação Add(a,b,c) é: dado (a,b),
o candidato c com maior grau de verdade dentre todos os p candidatos é o c
correto? (equivalente à acurácia usada na literatura de grokking para a
formulação como classificação, adaptada à formulação relacional do artigo.)
"""

from typing import Callable, List, Optional

from experiments.modular_addition.dataset import Fact
from src.neurosym.interpreter import Interpreter
from src.neurosym.logic import Atom, Constant


def make_argmax_accuracy_fn(
    interpreter: Interpreter, p: int
) -> Callable[[List[Fact]], Optional[float]]:
    def accuracy_fn(facts: List[Fact]) -> Optional[float]:
        positive_facts = [(formula, target) for formula, target in facts if target >= 0.5]
        if not positive_facts:
            return None

        correct = 0
        for formula, _ in positive_facts:
            a_name, b_name, c_true_name = (term.name for term in formula.terms)
            best_c, best_value = None, float("-inf")
            for c in range(p):
                candidate = Atom(
                    "Add", [Constant(a_name), Constant(b_name), Constant(str(c))]
                )
                predicted = interpreter.eval_formula(candidate, {})
                value = predicted._flatten(predicted.data)[0]
                if value > best_value:
                    best_value, best_c = value, c
            if str(best_c) == c_true_name:
                correct += 1

        return correct / len(positive_facts)

    return accuracy_fn
