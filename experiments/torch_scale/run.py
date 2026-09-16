"""Constrói um experimento sobre o motor batched (`torch_engine`) usando o
domínio sintético de escala (`experiments/torch_scale/dataset.py`) -- equivalente,
para o motor novo, do que `experiments/modular_addition/run.py` faz para o motor
original."""

from dataclasses import dataclass
from typing import List

import torch

from experiments.torch_scale.axioms import build_commutativity_axioms
from experiments.torch_scale.dataset import Fact, entity_names, generate_synthetic_domain
from src.neurosym.logic import Formula
from src.neurosym.torch_engine import (
    BatchedGroundingEnv,
    BatchedInterpreter,
    BatchedTrainer,
    DLGModel,
    build_predicate_mlp,
)


@dataclass
class TorchExperimentSpec:
    model: DLGModel
    trainer: BatchedTrainer
    rules: List[Formula]
    facts: List[Fact]
    val_facts: List[Fact]
    test_facts: List[Fact]


def build_torch_experiment(
    seed: int,
    num_entities: int,
    embedding_dim: int,
    hidden: int,
    epochs: int,
    num_relations: int = 1,
    arity: int = 3,
    num_positive_facts: int = 10_000,
    negatives_per_positive: int = 1,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    gamma_l1: float = 0.0,
    lambda_semantic: float = 1.0,
    use_axioms: bool = True,
    num_hidden_layers: int = 2,
    device: str = "cpu",
    val_eval_every: int = 1,
) -> TorchExperimentSpec:
    torch.manual_seed(seed)

    data = generate_synthetic_domain(
        num_entities=num_entities,
        num_relations=num_relations,
        arity=arity,
        num_positive_facts=num_positive_facts,
        negatives_per_positive=negatives_per_positive,
        train_frac=train_frac,
        val_frac=val_frac,
        seed=seed,
    )

    grounding_env = BatchedGroundingEnv(entity_names(num_entities), embedding_dim, seed=seed)
    predicate_map = {
        name: build_predicate_mlp(
            arity * embedding_dim, hidden, num_hidden_layers=num_hidden_layers
        )
        for name in data.relations
    }
    model = DLGModel(grounding_env, predicate_map)
    model.to(device)

    rules = (
        build_commutativity_axioms(data.train_facts, data.relations) if use_axioms else []
    )

    interpreter = BatchedInterpreter(predicate_map, grounding_env)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    trainer = BatchedTrainer(
        model,
        interpreter,
        optimizer,
        epochs=epochs,
        device=device,
        gamma_l1=gamma_l1,
        lambda_semantic=lambda_semantic,
        val_eval_every=val_eval_every,
    )

    return TorchExperimentSpec(
        model=model,
        trainer=trainer,
        rules=rules,
        facts=data.train_facts,
        val_facts=data.val_facts,
        test_facts=data.test_facts,
    )
