from typing import Dict, Optional, Union
from src.neurosym.tensor import Tensor
from src.neurosym.logic import (
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
from src.neurosym.module import Module
from .fuzzy_operators import get_operator, DEFAULT_OPERATORS, FuzzyOperator

PredicateMap = Dict[str, Module]
GroundingEnv = Dict[str, Tensor]


DEFAULT_QUANTIFIERS = {"forall": "min"}


class Interpreter:
    def __init__(
        self,
        predicate_map: PredicateMap,
        grounding_env: GroundingEnv,
        operator_config: Optional[Dict[str, str]] = None,
        quantifier_config: Optional[Dict[str, Union[str, Dict]]] = None,
    ):
        self.predicate_map = predicate_map
        self.grounding_env = grounding_env
        self.domain = list(grounding_env.keys())

        op_config = DEFAULT_OPERATORS.copy()
        if operator_config:
            op_config.update(operator_config)
        self.op_and: FuzzyOperator = get_operator(op_config["and"])
        self.op_or: FuzzyOperator = get_operator(op_config["or"])
        self.op_implies: FuzzyOperator = get_operator(op_config["implies"])

        q_config = DEFAULT_QUANTIFIERS.copy()
        if quantifier_config:
            q_config.update(quantifier_config)
        self.quantifier_forall = q_config["forall"]

    def _aggregate_forall(self, tensor: Tensor) -> Tensor:
        """Aplica a função de agregação configurada para Forall."""
        if (
            isinstance(self.quantifier_forall, dict)
            and "p_mean" in self.quantifier_forall
        ):
            p_value = self.quantifier_forall["p_mean"]
            return tensor.p_mean(p=p_value)
        elif self.quantifier_forall == "mean":
            return tensor.mean()
        elif self.quantifier_forall == "min":
            return tensor.min()
        else:
            raise ValueError(f"Agregador Forall desconhecido: {self.quantifier_forall}")

    def eval_formula(self, formula: Formula, current_env: GroundingEnv) -> Tensor:
        if isinstance(formula, Atom):
            term_embeddings = [
                self._eval_term(term, current_env) for term in formula.terms
            ]
            input_tensor = (
                Tensor.concatenate(term_embeddings, axis=1)
                if len(term_embeddings) > 1
                else term_embeddings[0]
            )
            predicate_module = self.predicate_map[formula.predicate_name]
            return predicate_module(input_tensor)

        elif isinstance(formula, And):
            left_val = self.eval_formula(formula.left, current_env)
            right_val = self.eval_formula(formula.right, current_env)
            return self.op_and(left_val, right_val)

        elif isinstance(formula, Or):
            left_val = self.eval_formula(formula.left, current_env)
            right_val = self.eval_formula(formula.right, current_env)
            return self.op_or(left_val, right_val)

        elif isinstance(formula, Implies):
            antecedent_val = self.eval_formula(formula.antecedent, current_env)
            consequent_val = self.eval_formula(formula.consequent, current_env)
            return self.op_implies(antecedent_val, consequent_val)

        elif isinstance(formula, Not):
            return Tensor(1.0) - self.eval_formula(formula.formula, current_env)

        elif isinstance(formula, Forall):
            truth_values = []
            for const_name in self.domain:
                temp_env = current_env.copy()
                temp_env[formula.variable.name] = self.grounding_env[const_name]
                truth_values.append(self.eval_formula(formula.formula, temp_env))

            if not truth_values:
                return Tensor(1.0)

            stacked_tensor = Tensor.concatenate(truth_values, axis=0)

            return self._aggregate_forall(stacked_tensor)

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
