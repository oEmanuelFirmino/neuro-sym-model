import pytest

torch = pytest.importorskip("torch")

from src.neurosym.logic import Atom, Constant, Forall, Implies, Variable
from src.neurosym.torch_engine.compile import compile_formulas, leaves, signature


def test_signature_groups_by_predicate_and_arity():
    f1 = Atom("Add", [Constant("1"), Constant("2"), Constant("3")])
    f2 = Atom("Add", [Constant("4"), Constant("5"), Constant("6")])
    f3 = Atom("Other", [Constant("1")])
    assert signature(f1) == signature(f2)
    assert signature(f1) != signature(f3)


def test_signature_distinguishes_tree_shape():
    implies = Implies(Atom("Add", [Constant("1")]), Atom("Add", [Constant("2")]))
    atom = Atom("Add", [Constant("1")])
    assert signature(implies) != signature(atom)


def test_leaves_order_matches_left_to_right_traversal():
    f = Implies(
        Atom("Add", [Constant("1"), Constant("2")]),
        Atom("Add", [Constant("3"), Constant("4")]),
    )
    assert leaves(f) == ["1", "2", "3", "4"]


def test_leaves_rejects_unbound_variable():
    x = Variable("x")
    with pytest.raises(TypeError):
        leaves(Atom("Add", [x]))


def test_unsupported_formula_type_raises():
    x = Variable("x")
    f = Forall(x, Atom("Add", [x]))
    with pytest.raises(NotImplementedError):
        signature(f)


def test_compile_formulas_groups_and_builds_leaf_idx():
    vocab = {str(i): i for i in range(10)}
    formulas = [
        Atom("Add", [Constant("1"), Constant("2")]),
        Atom("Add", [Constant("3"), Constant("4")]),
        Atom("Other", [Constant("5")]),
    ]
    groups = compile_formulas(formulas, vocab)
    assert len(groups) == 2

    add_group = next(g for g in groups if g.signature[1] == "Add")
    other_group = next(g for g in groups if g.signature[1] == "Other")
    assert add_group.leaf_idx.tolist() == [[1, 2], [3, 4]]
    assert other_group.leaf_idx.tolist() == [[5]]
    assert add_group.targets is None


def test_compile_formulas_with_targets():
    vocab = {str(i): i for i in range(10)}
    formulas = [
        Atom("Add", [Constant("1"), Constant("2")]),
        Atom("Add", [Constant("3"), Constant("4")]),
    ]
    groups = compile_formulas(formulas, vocab, targets=[1.0, 0.0])
    assert groups[0].targets.tolist() == [1.0, 0.0]


def test_compile_formulas_mismatched_targets_length_raises():
    vocab = {str(i): i for i in range(10)}
    formulas = [Atom("Add", [Constant("1")])]
    with pytest.raises(ValueError):
        compile_formulas(formulas, vocab, targets=[1.0, 0.0])


def test_compile_formulas_unknown_constant_raises():
    vocab = {"1": 0}
    formulas = [Atom("Add", [Constant("999")])]
    with pytest.raises(KeyError):
        compile_formulas(formulas, vocab)


def test_facts_and_rules_share_signature_must_be_compiled_separately():
    """Um Atom de fato e um Atom de axioma têm a mesma assinatura -- compilar os
    dois JUNTOS produz um único grupo misturando as duas coisas, o que quebraria
    a separação l_data/l_semantic do BatchedTrainer. Por isso `BatchedTrainer`
    sempre chama `compile_formulas` duas vezes (uma para facts, outra para
    rules) -- este teste documenta por quê."""
    vocab = {str(i): i for i in range(10)}
    fact = Atom("Add", [Constant("1"), Constant("2")])
    axiom = Atom("Add", [Constant("3"), Constant("4")])
    mixed_groups = compile_formulas([fact, axiom], vocab)
    assert len(mixed_groups) == 1
    assert mixed_groups[0].leaf_idx.shape[0] == 2
