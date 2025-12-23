import sys
import re
from pathlib import Path
from typing import List, Dict, Tuple

# As importações agora são relativas ao pacote 'src'
from ..tensor import Tensor
from ..logic import Formula, Atom, Forall, Variable, Constant, Implies, And, Or, Not


class KnowledgeBaseLoader:
    def __init__(self, data_path: str):
        self.base_path = Path(data_path)

    def load_domain(self, domain_file: str, embedding_dim: int) -> Dict[str, Tensor]:
        grounding_env = {}
        with open(self.base_path / domain_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.split("#")[0].strip()
                if line:
                    # NORMALIZAÇÃO: Converte chaves do domínio para lowercase
                    normalized_line = line.lower()
                    embedding = [
                        [
                            (hash(normalized_line + str(i)) % 1000 / 1000.0)
                            for i in range(embedding_dim)
                        ]
                    ]
                    grounding_env[normalized_line] = Tensor(
                        embedding, requires_grad=True
                    )
        return grounding_env

    def load_facts(self, facts_file: str) -> List[Tuple[Formula, float]]:
        facts = []
        with open(self.base_path / facts_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.split("#")[0].strip()
                if not line:
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 2:
                    continue
                pred_name = parts[0]
                # NORMALIZAÇÃO: Garante que constantes nos fatos sejam lowercase
                constants = [Constant(name.lower()) for name in parts[1:-1]]
                truth_value = float(parts[-1])
                facts.append((Atom(pred_name, constants), truth_value))
        return facts

    def load_rules(self, rules_file: str) -> List[Formula]:
        rules = []
        with open(self.base_path / rules_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.split("#")[0].strip()
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

        quantifier_match = re.match(
            r"forall\s+([a-zA-Z0-9_,\s]+?):\s*\((.*)\)", rule_str
        )
        if quantifier_match:
            var_names_str, formula_str = quantifier_match.groups()
            var_names = [v.strip() for v in var_names_str.split(",")]
            variables = {name: Variable(name) for name in var_names}

            parsed_formula = self._parse_formula_str(formula_str.strip(), variables)

            nested_formula = parsed_formula
            for var_name in reversed(var_names):
                nested_formula = Forall(variables[var_name], nested_formula)
            return nested_formula
        else:
            # Tratamento para regras sem quantificador explícito (Fatos ou regras proposicionais)
            try:
                return self._parse_formula_str(rule_str, {})
            except Exception:
                raise ValueError(
                    f"Formato de regra inválido ou não suportado: {rule_str}"
                )

    def _parse_formula_str(
        self, formula_str: str, variables: Dict[str, Variable]
    ) -> Formula:
        formula_str = formula_str.strip()

        if formula_str.startswith("¬"):
            return Not(self._parse_formula_str(formula_str[1:], variables))

        # Lógica de remoção de parênteses externos com verificação de balanceamento
        if formula_str.startswith("(") and formula_str.endswith(")"):
            depth = 0
            balanced_inside = True
            for i, char in enumerate(formula_str):
                if char == "(":
                    depth += 1
                elif char == ")":
                    depth -= 1
                if depth == 0 and i < len(formula_str) - 1:
                    balanced_inside = False
                    break

            if balanced_inside:
                formula_str = formula_str[1:-1].strip()

        if "→" in formula_str:
            antecedent_str, consequent_str = formula_str.split("→", 1)
            return Implies(
                self._parse_formula_str(antecedent_str.strip(), variables),
                self._parse_formula_str(consequent_str.strip(), variables),
            )

        if "∨" in formula_str:
            left_str, right_str = formula_str.split("∨", 1)
            return Or(
                self._parse_formula_str(left_str.strip(), variables),
                self._parse_formula_str(right_str.strip(), variables),
            )

        if "∧" in formula_str:
            left_str, right_str = formula_str.split("∧", 1)
            return And(
                self._parse_formula_str(left_str.strip(), variables),
                self._parse_formula_str(right_str.strip(), variables),
            )

        atom_match = re.match(r"([a-zA-Z0-9_]+)\((.*)\)", formula_str)
        if atom_match:
            pred_name, terms_str = atom_match.groups()
            term_names = [t.strip() for t in terms_str.split(",")]

            terms = []
            for name in term_names:
                # Verifica se é uma variável ligada ao quantificador
                if name in variables:
                    terms.append(variables[name])
                else:
                    # NORMALIZAÇÃO: Se não é variável, é constante -> Lowercase
                    # Isso resolve o problema de 'Socrates' vs 'socrates'
                    terms.append(Constant(name.lower()))

            return Atom(pred_name, terms)

        raise ValueError(f"Não foi possível interpretar a sub-fórmula: {formula_str}")
