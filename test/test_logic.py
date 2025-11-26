import sys
import logging

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
    print("❌ Erro ao importar módulos de lógica.")
    sys.exit(1)


class LogicTestFormatter:
    def __init__(self, log_level=logging.INFO):
        self.logger = self._setup_logger(log_level)

    def _setup_logger(self, log_level):
        logger = logging.getLogger("LogicASTTest")
        logger.setLevel(log_level)
        if logger.hasHandlers():
            logger.handlers.clear()
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(log_level)
        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(message)s", datefmt="%H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

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


class LogicTestSuite:
    def __init__(self):
        self.formatter = LogicTestFormatter()

    def test_ast_construction(self):
        self.formatter.print_section_header(
            "Construção da Árvore de Sintaxe Abstrata (AST)"
        )

        x = Variable("x")
        y = Variable("y")
        z = Variable("z")

        pedro = Constant("pedro")
        joao = Constant("joao")
        maria = Constant("maria")

        self.formatter.logger.info(
            "  🔹 Termos (Variáveis e Constantes) criados com sucesso."
        )

        p1 = Atom("Pai", [x, y])
        p2 = Atom("Pai", [y, z])
        p3 = Atom("Avo", [x, z])

        self.formatter.print_formula("Átomo 'Pai(x, y)'", p1)

        formula_and = And(p1, p2)
        self.formatter.print_formula("Fórmula 'E' (And)", formula_and)

        formula_implies = Implies(formula_and, p3)
        self.formatter.print_formula("Fórmula 'Implica'", formula_implies)

        formula_forall = Forall(x, Forall(y, Forall(z, formula_implies)))
        self.formatter.print_formula(
            "Fórmula 'Para Todo' (Forall) aninhada", formula_forall
        )

        formula_not = Not(Atom("Irmao", [pedro, joao]))
        self.formatter.print_formula("Fórmula 'Não' (Not)", formula_not)

        formula_or = Or(Atom("Mae", [maria, joao]), Atom("Pai", [pedro, joao]))
        self.formatter.print_formula("Fórmula 'Ou' (Or)", formula_or)

        formula_exists = Exists(z, Atom("Filho", [z, pedro]))
        self.formatter.print_formula("Fórmula 'Existe' (Exists)", formula_exists)

        self.formatter.print_success(
            "Todas as estruturas lógicas foram construídas com sucesso."
        )

    def run_all(self):
        self.formatter.print_banner("Teste do Bloco Simbólico (AST Lógica)")
        self.test_ast_construction()
        self.formatter.logger.info(
            "\n🎉 Todos os testes de lógica foram concluídos com sucesso!"
        )


def main():
    try:
        suite = LogicTestSuite()
        suite.run_all()
    except Exception as e:
        print(f"❌ Erro fatal durante a execução dos testes: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
