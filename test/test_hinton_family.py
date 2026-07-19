import sys
import logging
import pytest

try:
    from experiments.hinton_family.domain import (
        BASE_RELATIONS,
        DERIVED_RELATIONS,
        build_family_grounding_env,
        derived_proof_formula,
        generate_family,
    )
    from src.neurosym.explainability.metrics import compute_influences
    from src.neurosym.interpreter.interpreter import Interpreter
    from src.neurosym.module.module import Module, Linear, Sigmoid
except ImportError:
    pytest.fail("❌ Erro ao importar o domínio da família de Hinton.", pytrace=False)


class HintonFamilyTestFormatter:
    def __init__(self, log_level=logging.INFO):
        self.logger = logging.getLogger("HintonFamilyTest")
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
    return HintonFamilyTestFormatter()


class PredicateNet(Module):
    def __init__(self, in_features):
        super().__init__()
        self.add_module("layer", Linear(in_features, 1))
        self.add_module("activation", Sigmoid())

    def forward(self, x):
        return self.activation(self.layer(x))


class TestGenerateFamily:
    def test_full_negative_coverage(self, formatter):
        data = generate_family(seed=0)
        n = len(data.entities)
        assert n == 12
        # cobertura completa: n*(n-1) fatos por relação base
        assert len(data.base_facts) == len(BASE_RELATIONS) * n * (n - 1)
        formatter.logger.info("  ✅ cobertura completa de rótulos por relação base.")

    def test_known_gold_facts(self, formatter):
        data = generate_family(seed=0)
        facts = {(f.predicate_name, f.terms[0].name, f.terms[1].name): t
                 for f, t in data.base_facts}

        assert facts[("father", "christopher", "arthur")] == 1.0
        assert facts[("mother", "victoria", "colin")] == 1.0
        assert facts[("brother", "arthur", "victoria")] == 1.0
        assert facts[("sister", "jennifer", "james")] == 1.0
        assert facts[("father", "arthur", "colin")] == 0.0  # Arthur não tem filhos
        formatter.logger.info("  ✅ fatos-ouro da árvore de Hinton corretos.")

    def test_derived_gold_with_intermediates(self, formatter):
        data = generate_family(seed=0)
        uncle = data.derived_gold["uncle"]
        aunt = data.derived_gold["aunt"]

        # Arthur (irmão de Victoria) é tio de Colin/Charlotte via Victoria
        assert ("arthur", "colin", "victoria") in uncle
        assert ("arthur", "charlotte", "victoria") in uncle
        # Jennifer (irmã de James) é tia via James
        assert ("jennifer", "colin", "james") in aunt
        # sobrinhos: Colin é sobrinho de Arthur via Victoria, e de Jennifer via James
        nephew = data.derived_gold["nephew"]
        assert ("colin", "arthur", "victoria") in nephew
        assert ("colin", "jennifer", "james") in nephew
        formatter.logger.info("  ✅ relações derivadas e intermediários corretos.")


class TestDerivedProofFormula:
    def test_gradient_reaches_intermediate(self, formatter):
        data = generate_family(seed=0)
        grounding_env = build_family_grounding_env(data.entities, 4, seed=0)
        predicate_map = {r: PredicateNet(8) for r in BASE_RELATIONS}
        predicate_map.update(
            {f"{r}_flat": PredicateNet(8) for r in DERIVED_RELATIONS}
        )
        interpreter = Interpreter(predicate_map, grounding_env)

        formula = derived_proof_formula("uncle", "arthur", "colin", data.entities)
        influences = compute_influences(interpreter, formula)

        # o intermediário do caminho (victoria) recebe gradiente na composta
        assert influences["victoria"] > 0.0
        assert influences["arthur"] > 0.0
        assert influences["colin"] > 0.0
        formatter.logger.info(
            "  ✅ gradiente da prova composta alcança o intermediário (victoria)."
        )
