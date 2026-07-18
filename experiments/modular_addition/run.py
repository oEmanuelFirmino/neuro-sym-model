"""Construtores de experimento (`build_fn`) para o domínio de Adição Modular,
compatíveis com `experiments/run_multiseed.py` -- Fase 3 do plano de
correções do artigo.

Três arquiteturas comparáveis (mesmos embeddings, mesmo MLP de predicado,
mesmo split treino/val/teste, mesmo orçamento de épocas -- só o mecanismo de
inferência muda), refletindo a Tabela 1 do artigo:

  - `build_dlg`: Product T-norm + regularização L1 + axiomas (DLG, proposto).
  - `build_mlp_baseline`: sem regras, sem L_semantic, sem L1 -- só L_data.
  - `build_ltn_baseline`: operadores de Lukasiewicz em vez de Product T-norm,
    sem L1 (aproxima o baseline LTN da Tabela 1; a agregação p-mean da LTN
    original não é exercida aqui porque os axiomas são instanciados por
    exemplo de treino, não via `Forall`, e `Trainer.fit` agrega L_semantic
    como média aritmética simples -- ver nota em `docs/plano-correcoes-artigo.md`).

Semantic Loss e DeepProbLog ficam para uma fase seguinte (mecanismos de WMC /
nAD não mapeiam diretamente para o interpretador atual, só Product/Lukasiewicz
T-norms).
"""

from typing import Optional

from experiments.modular_addition.axioms import commutativity_axioms, identity_axioms
from experiments.modular_addition.dataset import build_grounding_env, generate_split
from experiments.modular_addition.evaluation import make_argmax_accuracy_fn
from experiments.run_multiseed import ExperimentSpec
from src.neurosym.interpreter import Interpreter
from src.neurosym.module.module import Linear, ReLU, Sequential, Sigmoid
from src.neurosym.training.optimizer import AdamW
from src.neurosym.training.trainer import Trainer


def _build_predicate(in_features: int, hidden: int):
    return Sequential(
        Linear(in_features, hidden),
        ReLU(),
        Linear(hidden, hidden),
        ReLU(),
        Linear(hidden, 1),
        Sigmoid(),
    )


def _base_experiment(
    seed: int,
    p: int,
    embedding_dim: int,
    hidden: int,
    epochs: int,
    lr: float,
    train_frac: float,
    val_frac: float,
    negatives_per_positive: int,
    val_eval_every: int,
    operator_config: Optional[dict],
    use_axioms: bool,
    lambda_semantic: float,
    gamma_l1: float,
    weight_decay: float,
):
    data = generate_split(
        p,
        seed=seed,
        train_frac=train_frac,
        val_frac=val_frac,
        negatives_per_positive=negatives_per_positive,
    )
    grounding_env = build_grounding_env(p, embedding_dim, seed)
    predicate_map = {"Add": _build_predicate(3 * embedding_dim, hidden)}
    interpreter = Interpreter(
        predicate_map, grounding_env, operator_config=operator_config
    )

    rules = (
        commutativity_axioms(data.train_pairs, p) + identity_axioms(p)
        if use_axioms
        else []
    )

    all_params = list(grounding_env.values()) + predicate_map["Add"].parameters()
    optimizer = AdamW(all_params, lr=lr, weight_decay=weight_decay)
    accuracy_fn = make_argmax_accuracy_fn(interpreter, p)

    trainer = Trainer(
        interpreter,
        optimizer,
        epochs=epochs,
        lambda_semantic=lambda_semantic,
        gamma_l1=gamma_l1,
        accuracy_fn=accuracy_fn,
        val_eval_every=val_eval_every,
    )

    return ExperimentSpec(
        interpreter=interpreter,
        trainer=trainer,
        rules=rules,
        facts=data.train_facts,
        val_facts=data.val_facts,
        test_facts=data.test_facts,
    )


def make_dlg_build_fn(
    p: int,
    embedding_dim: int = 16,
    hidden: int = 64,
    epochs: int = 200,
    lr: float = 1e-3,
    train_frac: float = 0.30,
    val_frac: float = 0.35,
    negatives_per_positive: int = 1,
    val_eval_every: int = 1,
    lambda_semantic: float = 1.0,
    gamma_l1: float = 1e-4,
    weight_decay: float = 1e-2,
):
    def build_fn(seed: int) -> ExperimentSpec:
        return _base_experiment(
            seed,
            p,
            embedding_dim,
            hidden,
            epochs,
            lr,
            train_frac,
            val_frac,
            negatives_per_positive,
            val_eval_every,
            operator_config=None,  # Product T-norm (default do Interpreter)
            use_axioms=True,
            lambda_semantic=lambda_semantic,
            gamma_l1=gamma_l1,
            weight_decay=weight_decay,
        )

    return build_fn


def make_mlp_baseline_build_fn(
    p: int,
    embedding_dim: int = 16,
    hidden: int = 64,
    epochs: int = 200,
    lr: float = 1e-3,
    train_frac: float = 0.30,
    val_frac: float = 0.35,
    negatives_per_positive: int = 1,
    val_eval_every: int = 1,
    weight_decay: float = 1e-2,
):
    def build_fn(seed: int) -> ExperimentSpec:
        return _base_experiment(
            seed,
            p,
            embedding_dim,
            hidden,
            epochs,
            lr,
            train_frac,
            val_frac,
            negatives_per_positive,
            val_eval_every,
            operator_config=None,
            use_axioms=False,  # sem L_semantic
            lambda_semantic=0.0,
            gamma_l1=0.0,  # sem L1
            weight_decay=weight_decay,
        )

    return build_fn


def make_ltn_baseline_build_fn(
    p: int,
    embedding_dim: int = 16,
    hidden: int = 64,
    epochs: int = 200,
    lr: float = 1e-3,
    train_frac: float = 0.30,
    val_frac: float = 0.35,
    negatives_per_positive: int = 1,
    val_eval_every: int = 1,
    lambda_semantic: float = 1.0,
    weight_decay: float = 1e-2,
):
    lukasiewicz_operators = {
        "and": "lukasiewicz_and",
        "or": "lukasiewicz_or",
        "implies": "lukasiewicz_implies",
    }

    def build_fn(seed: int) -> ExperimentSpec:
        return _base_experiment(
            seed,
            p,
            embedding_dim,
            hidden,
            epochs,
            lr,
            train_frac,
            val_frac,
            negatives_per_positive,
            val_eval_every,
            operator_config=lukasiewicz_operators,
            use_axioms=True,
            lambda_semantic=lambda_semantic,
            gamma_l1=0.0,  # sem esparsidade estrutural
            weight_decay=weight_decay,
        )

    return build_fn
