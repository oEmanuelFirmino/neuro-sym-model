"""Equivalente batched de `src/neurosym/interpreter/interpreter.py`: mesma
recursão sobre a AST de `Formula`, mas cada "valor" que flui pela recursão é um
`torch.Tensor` de shape `(B,)` (um lote inteiro de instâncias grounded) em vez de
um escalar por chamada."""

from typing import Dict, List, Optional

import torch
from torch import nn

from src.neurosym.logic import And, Atom, Formula, Implies, Not, Or

from . import operators
from .compile import CompiledGroup
from .grounding import BatchedGroundingEnv
from .operators import DEFAULT_OPERATORS, FuzzyOperator, get_operator

PredicateMap = Dict[str, nn.Module]


class BatchedInterpreter:
    def __init__(
        self,
        predicate_map: PredicateMap,
        grounding_env: BatchedGroundingEnv,
        operator_config: Optional[Dict[str, str]] = None,
    ):
        self.predicate_map = predicate_map
        self.grounding_env = grounding_env

        op_config = DEFAULT_OPERATORS.copy()
        if operator_config:
            op_config.update(operator_config)
        self.op_and: FuzzyOperator = get_operator(op_config["and"])
        self.op_or: FuzzyOperator = get_operator(op_config["or"])
        self.op_implies: FuzzyOperator = get_operator(op_config["implies"])

    def eval_group(self, group: CompiledGroup) -> torch.Tensor:
        cursor = [0]  # int mutável (lista de 1) para avançar pela recursão
        return self._eval(group.template, group.leaf_idx, cursor)

    def _eval(self, formula: Formula, leaf_idx: torch.Tensor, cursor: List[int]) -> torch.Tensor:
        if isinstance(formula, Atom):
            n_terms = len(formula.terms)
            cols = leaf_idx[:, cursor[0] : cursor[0] + n_terms]
            cursor[0] += n_terms
            embeds = [self.grounding_env.embed(cols[:, i]) for i in range(n_terms)]
            x = embeds[0] if n_terms == 1 else torch.cat(embeds, dim=1)
            predicate_module = self.predicate_map[formula.predicate_name]
            return predicate_module(x).squeeze(-1)

        if isinstance(formula, Not):
            return operators.logical_not(self._eval(formula.formula, leaf_idx, cursor))

        if isinstance(formula, And):
            left = self._eval(formula.left, leaf_idx, cursor)
            right = self._eval(formula.right, leaf_idx, cursor)
            return self.op_and(left, right)

        if isinstance(formula, Or):
            left = self._eval(formula.left, leaf_idx, cursor)
            right = self._eval(formula.right, leaf_idx, cursor)
            return self.op_or(left, right)

        if isinstance(formula, Implies):
            antecedent = self._eval(formula.antecedent, leaf_idx, cursor)
            consequent = self._eval(formula.consequent, leaf_idx, cursor)
            return self.op_implies(antecedent, consequent)

        raise NotImplementedError(
            f"BatchedInterpreter não suporta fórmulas do tipo {type(formula).__name__} "
            "nesta fase."
        )
