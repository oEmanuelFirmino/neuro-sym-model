from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List


class Term(ABC):
    @abstractmethod
    def __repr__(self) -> str:
        pass


class Variable(Term):
    def __init__(self, name: str):
        self.name = name

    def __repr__(self) -> str:
        return f"Var({self.name})"


class Constant(Term):
    def __init__(self, name: str):
        self.name = name

    def __repr__(self) -> str:
        return f"Const({self.name})"


class Formula(ABC):
    @abstractmethod
    def __repr__(self) -> str:
        pass


class Atom(Formula):
    def __init__(self, predicate_name: str, terms: List[Term]):
        self.predicate_name = predicate_name
        self.terms = terms

    def __repr__(self) -> str:
        return f"{self.predicate_name}({', '.join(map(str, self.terms))})"


class Not(Formula):
    def __init__(self, formula: Formula):
        self.formula = formula

    def __repr__(self) -> str:
        return f"¬({self.formula})"


class And(Formula):
    def __init__(self, left: Formula, right: Formula):
        self.left = left
        self.right = right

    def __repr__(self) -> str:
        return f"({self.left} ∧ {self.right})"


class Or(Formula):
    def __init__(self, left: Formula, right: Formula):
        self.left = left
        self.right = right

    def __repr__(self) -> str:
        return f"({self.left} ∨ {self.right})"


class Implies(Formula):
    def __init__(self, antecedent: Formula, consequent: Formula):
        self.antecedent = antecedent
        self.consequent = consequent

    def __repr__(self) -> str:
        return f"({self.antecedent} → {self.consequent})"


class Forall(Formula):
    def __init__(self, variable: Variable, formula: Formula):
        self.variable = variable
        self.formula = formula

    def __repr__(self) -> str:
        return f"∀{self.variable}.({self.formula})"


class Exists(Formula):
    def __init__(self, variable: Variable, formula: Formula):
        self.variable = variable
        self.formula = formula

    def __repr__(self) -> str:
        return f"∃{self.variable}.({self.formula})"
