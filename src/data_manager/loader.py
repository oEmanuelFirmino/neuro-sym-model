import sys
import re
from pathlib import Path
from typing import List, Dict, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.tensor import Tensor
from src.logic import Formula, Atom, Forall, Variable, Constant, Implies, And, Or, Not


class KnowledgeBaseLoader:
    def __init__(self, data_path: str):
        self.base_path = Path(data_path)

    def load_domain(self, domain_file: str, embedding_dim: int) -> Dict[str, Tensor]:
        grounding_env = {}
        with open(self.base_path / domain_file, "r") as f:
            for line in f:
                const_name = line.strip()
                if const_name:

                    embedding = [
                        [
                            (hash(const_name + str(i)) % 1000 / 1000.0)
                            for i in range(embedding_dim)
                        ]
                    ]
                    grounding_env[const_name] = Tensor(embedding, requires_grad=True)
        return grounding_env

    def load_facts(self, facts_file: str) -> List[Tuple[Formula, float]]:
        facts = []
        with open(self.base_path / facts_file, "r") as f:
            for line in f:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 3:
                    continue
                pred_name = parts[0]
                constants = [Constant(name) for name in parts[1:-1]]
                truth_value = float(parts[-1])
                facts.append((Atom(pred_name, constants), truth_value))
        return facts

    def load_rules(self, rules_file: str) -> List[Formula]:
        rules = []
        with open(self.base_path / rules_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    rules.append(self._parse_rule(line))
        return rules

    def _parse_rule(self, rule_str: str) -> Formula:
        rule_str = (
            rule_str.replace("->", "→")
            .replace("&", "∧")
            .replace("|", "∨")
            .replace("~", "¬")
        )

        quantifier_match = re.match(r"forall\s+([a-zA-Z0-9_]+):\s*\((.*)\)", rule_str)
        if quantifier_match:
            var_name, formula_str = quantifier_match.groups()
            variable = Variable(var_name)

            return Forall(
                variable,
                self._parse_formula_str(formula_str.strip(), {var_name: variable}),
            )
        else:
            raise ValueError(f"Formato de regra inválido ou não suportado: {rule_str}")

    def _parse_formula_str(
        self, formula_str: str, variables: Dict[str, Variable]
    ) -> Formula:

        formula_str = formula_str.strip()

        if formula_str.startswith("(") and formula_str.endswith(")"):
            formula_str = formula_str[1:-1]

        if "→" in formula_str:
            antecedent_str, consequent_str = formula_str.split("→", 1)
            antecedent = self._parse_formula_str(antecedent_str.strip(), variables)
            consequent = self._parse_formula_str(consequent_str.strip(), variables)
            return Implies(antecedent, consequent)

        if "∧" in formula_str:
            left_str, right_str = formula_str.split("∧", 1)
            left = self._parse_formula_str(left_str.strip(), variables)
            right = self._parse_formula_str(right_str.strip(), variables)
            return And(left, right)

        atom_match = re.match(r"([a-zA-Z0-9_]+)\((.*)\)", formula_str)
        if atom_match:
            pred_name, terms_str = atom_match.groups()
            term_names = [t.strip() for t in terms_str.split(",")]
            terms = []
            for name in term_names:
                if name in variables:
                    terms.append(variables[name])
                else:
                    terms.append(Constant(name))
            return Atom(pred_name, terms)

        raise ValueError(f"Não foi possível interpretar a sub-fórmula: {formula_str}")
