import sys
import logging
import pytest

try:
    from experiments.modular_addition.dataset import (
        generate_split,
        build_grounding_env,
    )
    from experiments.modular_addition.axioms import (
        commutativity_axioms,
        identity_axioms,
    )
    from experiments.modular_addition.evaluation import make_argmax_accuracy_fn
    from src.neurosym.tensor.tensor import Tensor
    from src.neurosym.module.module import Module
    from src.neurosym.logic.logic import Atom, Constant, Implies
    from src.neurosym.interpreter.interpreter import Interpreter
except ImportError:
    pytest.fail(
        "❌ Erro ao importar módulos do domínio de Adição Modular.", pytrace=False
    )


class ModularAdditionTestFormatter:
    def __init__(self, log_level=logging.INFO):
        self.logger = logging.getLogger("ModularAdditionTest")
        self.logger.setLevel(log_level)
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setLevel(log_level)
            formatter = logging.Formatter(
                "%(asctime)s | %(name)s | %(message)s", datefmt="%H:%M:%S"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)


@pytest.fixture
def formatter():
    return ModularAdditionTestFormatter()


class TestGenerateSplit:
    def test_covers_all_pairs_without_overlap(self, formatter):
        p = 11
        data = generate_split(p, seed=0, train_frac=0.3, val_frac=0.35)

        train_set = set(data.train_pairs)
        val_set = set(data.val_pairs)
        test_set = set(data.test_pairs)

        assert len(train_set) + len(val_set) + len(test_set) == p * p
        assert train_set.isdisjoint(val_set)
        assert train_set.isdisjoint(test_set)
        assert val_set.isdisjoint(test_set)
        formatter.logger.info(
            "  ✅ treino/validação/teste particionam todos os pares sem sobreposição."
        )

    def test_split_fractions_approximate_request(self, formatter):
        p = 20
        data = generate_split(p, seed=0, train_frac=0.3, val_frac=0.35)
        n_total = p * p

        assert len(data.train_pairs) == pytest.approx(n_total * 0.3, abs=1)
        assert len(data.val_pairs) == pytest.approx(n_total * 0.35, abs=1)
        assert len(data.test_pairs) == pytest.approx(n_total * 0.35, abs=1)
        formatter.logger.info("  ✅ tamanhos dos splits batem com as frações pedidas.")

    def test_same_seed_is_deterministic(self, formatter):
        data_a = generate_split(11, seed=7)
        data_b = generate_split(11, seed=7)

        assert data_a.train_pairs == data_b.train_pairs
        assert data_a.val_pairs == data_b.val_pairs
        assert data_a.test_pairs == data_b.test_pairs
        formatter.logger.info("  ✅ a mesma seed reproduz exatamente o mesmo split.")

    def test_different_seed_changes_split(self, formatter):
        data_a = generate_split(11, seed=1)
        data_b = generate_split(11, seed=2)

        assert data_a.train_pairs != data_b.train_pairs
        formatter.logger.info("  ✅ seeds diferentes produzem splits diferentes.")

    def test_facts_include_positive_and_negatives(self, formatter):
        p = 13
        data = generate_split(p, seed=0, train_frac=0.3, val_frac=0.35, negatives_per_positive=2)

        n_pairs = len(data.train_pairs)
        assert len(data.train_facts) == n_pairs * 3  # 1 positivo + 2 negativos

        for a, b in data.train_pairs:
            correct = (a + b) % p
            matching = [
                (f, t)
                for f, t in data.train_facts
                if f.terms[0].name == str(a) and f.terms[1].name == str(b)
            ]
            positives = [f for f, t in matching if t == 1.0]
            negatives = [f for f, t in matching if t == 0.0]

            assert len(positives) == 1
            assert positives[0].terms[2].name == str(correct)
            assert len(negatives) == 2
            assert all(f.terms[2].name != str(correct) for f in negatives)

        formatter.logger.info(
            "  ✅ cada par gera 1 fato positivo (c correto) e N negativos (c incorreto)."
        )

    def test_invalid_fractions_raise(self, formatter):
        with pytest.raises(ValueError):
            generate_split(11, seed=0, train_frac=0.6, val_frac=0.5)
        formatter.logger.info("  ✅ train_frac + val_frac >= 1 levanta ValueError.")


class TestBuildGroundingEnv:
    def test_creates_one_tensor_per_constant(self, formatter):
        env = build_grounding_env(p=5, embedding_dim=4, seed=0)

        assert set(env.keys()) == {"0", "1", "2", "3", "4"}
        for tensor in env.values():
            assert tensor.shape == (1, 4)
            assert tensor.requires_grad

        formatter.logger.info("  ✅ um embedding treinável por constante, shape correto.")

    def test_deterministic_by_seed(self, formatter):
        env_a = build_grounding_env(p=5, embedding_dim=4, seed=3)
        env_b = build_grounding_env(p=5, embedding_dim=4, seed=3)

        for key in env_a:
            assert env_a[key].data == env_b[key].data
        formatter.logger.info("  ✅ mesma seed reproduz os mesmos embeddings iniciais.")


class TestAxioms:
    def test_commutativity_structure(self, formatter):
        pairs = [(2, 3), (5, 0)]
        p = 7
        axioms = commutativity_axioms(pairs, p)

        assert len(axioms) == 2
        for axiom, (a, b) in zip(axioms, pairs):
            c = (a + b) % p
            assert isinstance(axiom, Implies)
            assert axiom.antecedent.predicate_name == "Add"
            assert [t.name for t in axiom.antecedent.terms] == [str(a), str(b), str(c)]
            assert [t.name for t in axiom.consequent.terms] == [str(b), str(a), str(c)]

        formatter.logger.info("  ✅ Add(a,b,c) -> Add(b,a,c) instanciado corretamente.")

    def test_identity_axioms_cover_whole_domain(self, formatter):
        p = 6
        axioms = identity_axioms(p)

        assert len(axioms) == p
        for a, axiom in enumerate(axioms):
            assert isinstance(axiom, Atom)
            assert [t.name for t in axiom.terms] == [str(a), "0", str(a)]

        formatter.logger.info("  ✅ Add(a,0,a) instanciado para todo a no domínio.")


class _ExactAddPredicate(Module):
    """Predicado de brinquedo (não-treinável) que sabe a soma modular de
    verdade -- usado só para testar a lógica de argmax do avaliador."""

    def __init__(self, p: int):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        a, b, c = x.data[0]
        target = (int(round(a)) + int(round(b))) % self.p
        value = 1.0 if int(round(c)) == target else 0.0
        return Tensor([[value]])


class TestArgmaxAccuracyFn:
    def test_perfect_predicate_scores_full_accuracy(self, formatter):
        p = 7
        grounding_env = {str(i): Tensor([[float(i)]]) for i in range(p)}
        interpreter = Interpreter({"Add": _ExactAddPredicate(p)}, grounding_env)
        accuracy_fn = make_argmax_accuracy_fn(interpreter, p)

        facts = [
            (Atom("Add", [Constant("2"), Constant("3"), Constant(str((2 + 3) % p))]), 1.0),
            (Atom("Add", [Constant("5"), Constant("6"), Constant(str((5 + 6) % p))]), 1.0),
        ]

        assert accuracy_fn(facts) == pytest.approx(1.0)
        formatter.logger.info("  ✅ predicado que sabe a soma correta atinge acurácia 1.0.")

    def test_ignores_negative_facts_in_query_set(self, formatter):
        p = 7
        grounding_env = {str(i): Tensor([[float(i)]]) for i in range(p)}
        interpreter = Interpreter({"Add": _ExactAddPredicate(p)}, grounding_env)
        accuracy_fn = make_argmax_accuracy_fn(interpreter, p)

        correct_c = (2 + 3) % p
        wrong_c = (correct_c + 1) % p
        facts = [
            (Atom("Add", [Constant("2"), Constant("3"), Constant(str(correct_c))]), 1.0),
            (Atom("Add", [Constant("2"), Constant("3"), Constant(str(wrong_c))]), 0.0),
        ]

        # Só o fato positivo conta como consulta (a,b) -> avalia-se o argmax
        # sobre todos os candidatos de c, não os fatos negativos literalmente.
        assert accuracy_fn(facts) == pytest.approx(1.0)
        formatter.logger.info("  ✅ fatos negativos não entram como consultas próprias.")

    def test_empty_facts_returns_none(self, formatter):
        p = 5
        grounding_env = {str(i): Tensor([[float(i)]]) for i in range(p)}
        interpreter = Interpreter({"Add": _ExactAddPredicate(p)}, grounding_env)
        accuracy_fn = make_argmax_accuracy_fn(interpreter, p)

        assert accuracy_fn([]) is None
        formatter.logger.info("  ✅ retorna None quando não há fatos positivos.")
