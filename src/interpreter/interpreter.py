from typing import Dict, List
from src.tensor.tensor import Tensor
from src.logic.logic import (
    Formula,
    Atom,
    And,
    Or,
    Implies,
    Forall,
    Not,
    Constant,
    Variable,
    Term,
)
from src.module.module import Module

PredicateMap = Dict[str, Module]
GroundingEnv = Dict[str, Tensor]


class Interpreter:
    def __init__(self, predicate_map: PredicateMap, grounding_env: GroundingEnv):
        self.predicate_map = predicate_map
        self.grounding_env = grounding_env
        self.domain = list(grounding_env.keys())

    def eval_formula(self, formula: Formula, current_env: GroundingEnv) -> Tensor:
        if isinstance(formula, Atom):
            term_embeddings = [
                self._eval_term(term, current_env) for term in formula.terms
            ]

            if len(term_embeddings) > 1:
                input_tensor = Tensor.concatenate(term_embeddings, axis=1)
            else:
                input_tensor = term_embeddings[0]

            predicate_module = self.predicate_map[formula.predicate_name]
            return predicate_module(input_tensor)

        elif isinstance(formula, And):
            left_val = self.eval_formula(formula.left, current_env)
            right_val = self.eval_formula(formula.right, current_env)
            return left_val * right_val

        elif isinstance(formula, Or):
            left_val = self.eval_formula(formula.left, current_env)
            right_val = self.eval_formula(formula.right, current_env)
            return left_val + right_val - (left_val * right_val)

        elif isinstance(formula, Implies):
            antecedent_val = self.eval_formula(formula.antecedent, current_env)
            consequent_val = self.eval_formula(formula.consequent, current_env)
            return Tensor(1.0) - antecedent_val + (antecedent_val * consequent_val)

        elif isinstance(formula, Not):
            return Tensor(1.0) - self.eval_formula(formula.formula, current_env)

        elif isinstance(formula, Forall):
            truth_values = []
            for const_name in self.domain:
                temp_env = current_env.copy()
                temp_env[formula.variable.name] = self.grounding_env[const_name]
                truth_values.append(self.eval_formula(formula.formula, temp_env))

            stacked_tensor = Tensor.concatenate(truth_values, axis=0)
            return stacked_tensor.min()

        else:
            raise NotImplementedError(
                f"Avaliação para o tipo de fórmula {type(formula).__name__} não implementada."
            )

    def _eval_term(self, term: Term, current_env: GroundingEnv) -> Tensor:
        if isinstance(term, Constant):
            return self.grounding_env[term.name]
        elif isinstance(term, Variable):
            return current_env[term.name]
        else:
            raise TypeError(f"Tipo de termo desconhecido: {type(term).__name__}")
