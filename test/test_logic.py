import sys
import logging
import pytest

try:
    from src.neurosym.logic.logic import (
        Variable,
        Constant,
        Formula,
        Atom,
        Not,
        And,
        Or,
        Implies,
        Forall,
        Exists,
    )
except ImportError:
    pytest.fail("❌ Erro ao importar módulos de lógica.", pytrace=False)


class LogicTestFormatter:
    def __init__(self, log_level=logging.INFO):
        self.logger = logging.getLogger("LogicASTTest")
        self.logger.setLevel(log_level)

        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setLevel(log_level)
            formatter = logging.Formatter(
                "%(asctime)s | %(name)s | %(message)s", datefmt="%H:%M:%S"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def print_banner(self, title: str):
        self.logger.info("")
        self.logger.info("=" * 75)
        self.logger.info(f"  📖 {title.upper()} 📖")
        self.logger.info("=" * 75)

    def print_section_header(self, section_name: str):
        self.logger.info("")
        self.logger.info(f"▶️  {section_name}")
        self.logger.info("<" + "-" * 60)

    def print_formula(self, description: str, formula: Formula):
        self.logger.info(f"  🔹 {description}:")
        self.logger.info(f"     {formula}")

    def print_success(self, message: str):
        self.logger.info(f"  ✅ {message}")


@pytest.fixture
def formatter():
    return LogicTestFormatter()


class TestLogic:

    def test_ast_construction(self, formatter):
        formatter.print_banner("Teste do Bloco Simbólico (AST Lógica)")
        formatter.print_section_header("Construção da Árvore de Sintaxe Abstrata (AST)")

        x = Variable("x")
        y = Variable("y")
        z = Variable("z")
        pedro = Constant("pedro")
        joao = Constant("joao")
        maria = Constant("maria")

        formatter.logger.info(
            "  🔹 Termos (Variáveis e Constantes) criados com sucesso."
        )

        p1 = Atom("Pai", [x, y])
        p2 = Atom("Pai", [y, z])
        p3 = Atom("Avo", [x, z])

        formatter.print_formula("Átomo 'Pai(x, y)'", p1)

        assert str(p1) == "Pai(Var(x), Var(y))"

        formula_and = And(p1, p2)
        formatter.print_formula("Fórmula 'E' (And)", formula_and)
        assert str(formula_and) == "(Pai(Var(x), Var(y)) ∧ Pai(Var(y), Var(z)))"

        formula_implies = Implies(formula_and, p3)
        formatter.print_formula("Fórmula 'Implica'", formula_implies)
        assert (
            str(formula_implies)
            == "((Pai(Var(x), Var(y)) ∧ Pai(Var(y), Var(z))) → Avo(Var(x), Var(z)))"
        )

        formula_forall = Forall(x, Forall(y, Forall(z, formula_implies)))
        formatter.print_formula("Fórmula 'Para Todo' (Forall) aninhada", formula_forall)

        assert str(formula_forall).startswith("∀Var(x).(∀Var(y).(∀Var(z).")

        formula_not = Not(Atom("Irmao", [pedro, joao]))
        formatter.print_formula("Fórmula 'Não' (Not)", formula_not)
        assert str(formula_not) == "¬(Irmao(Const(pedro), Const(joao)))"

        formula_or = Or(Atom("Mae", [maria, joao]), Atom("Pai", [pedro, joao]))
        formatter.print_formula("Fórmula 'Ou' (Or)", formula_or)
        assert "∨" in str(formula_or)

        formula_exists = Exists(z, Atom("Filho", [z, pedro]))
        formatter.print_formula("Fórmula 'Existe' (Exists)", formula_exists)
        assert str(formula_exists) == "∃Var(z).(Filho(Var(z), Const(pedro)))"

        formatter.print_success(
            "Todas as estruturas lógicas foram construídas e validadas com sucesso."
        )
