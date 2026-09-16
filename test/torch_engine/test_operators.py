"""Os operadores fuzzy batched (`torch_engine.operators`) devem reproduzir
exatamente as fórmulas do motor original (`src.neurosym.interpreter.fuzzy_operators`)
-- só muda o tipo do operando (escalar Tensor vs. lote torch.Tensor)."""

import pytest

torch = pytest.importorskip("torch")

import numpy as np

from src.neurosym.interpreter import fuzzy_operators as old_ops
from src.neurosym.tensor import Tensor
from src.neurosym.torch_engine import operators as new_ops

BINARY_OPERATOR_PAIRS = [
    (old_ops.product_tnorm, new_ops.product_and),
    (old_ops.product_tconorm, new_ops.product_or),
    (old_ops.product_implication, new_ops.product_implies),
    (old_ops.lukasiewicz_tnorm, new_ops.lukasiewicz_and),
    (old_ops.lukasiewicz_tconorm, new_ops.lukasiewicz_or),
    (old_ops.lukasiewicz_implication, new_ops.lukasiewicz_implies),
]


class TestOperatorParity:
    def test_binary_operators_match_old_engine(self):
        grid = np.linspace(0.0, 1.0, 11)
        a_grid, b_grid = np.meshgrid(grid, grid)
        a_vals, b_vals = a_grid.flatten(), b_grid.flatten()

        for old_op, new_op in BINARY_OPERATOR_PAIRS:
            old_results = [
                old_op(Tensor(float(a)), Tensor(float(b))).data
                for a, b in zip(a_vals, b_vals)
            ]
            new_results = (
                new_op(
                    torch.tensor(a_vals, dtype=torch.float64),
                    torch.tensor(b_vals, dtype=torch.float64),
                )
                .numpy()
                .tolist()
            )
            assert np.allclose(old_results, new_results, atol=1e-9), old_op.__name__

    def test_not_matches_old_engine(self):
        a_vals = np.linspace(0.0, 1.0, 11)
        old_results = [(Tensor(1.0) - Tensor(float(a))).data for a in a_vals]
        new_results = new_ops.logical_not(
            torch.tensor(a_vals, dtype=torch.float64)
        ).numpy()
        assert np.allclose(old_results, new_results, atol=1e-9)

    def test_get_operator_matches_default_config(self):
        assert new_ops.get_operator("product_and") is new_ops.product_and
        assert new_ops.DEFAULT_OPERATORS == {
            "and": "product_and",
            "or": "product_or",
            "implies": "product_implies",
        }

    def test_get_operator_unknown_raises(self):
        with pytest.raises(ValueError):
            new_ops.get_operator("not_an_operator")
