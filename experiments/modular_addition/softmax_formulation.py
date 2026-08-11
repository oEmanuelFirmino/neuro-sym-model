"""Formulação por classificação softmax do domínio de Adição Modular.

Motivação (ver docs/plano-correcoes-artigo.md, Fase 3): o sweep de
hiperparâmetros refutou as hipóteses de wd/capacidade/negativos, deixando como
principal suspeita a formulação relacional (avaliar triplas `Add(a,b,c)`
isoladamente com MSE). Esta é a formulação clássica da literatura de grokking
(Power et al. 2022): dado o par (a,b), prever `c` por classificação entre as p
classes com cross-entropy — cada exemplo de treino compara implicitamente
todos os candidatos, em vez de ver 2-4 triplas isoladas.

Os axiomas do artigo são traduzidos para o cenário distribucional:
  - comutatividade: as distribuições previstas para (a,b) e (b,a) devem
    coincidir (penalidade MSE entre os vetores de probabilidade);
  - identidade: a distribuição de (a,0) deve concentrar-se em `a`
    (cross-entropy com alvo `a`).

`SoftmaxTrainer` imita a interface pública do `Trainer` (callbacks, `fit`,
`evaluate_accuracy`, mesmos nomes de chave nos logs) para que
`experiments/run_multiseed.py` e `experiments/reporting.py` funcionem sem
alteração.
"""

import random
from typing import Dict, List, Optional, Tuple

from experiments.run_multiseed import ExperimentSpec
from src.neurosym.module.module import Linear, ReLU, Sequential
from src.neurosym.tensor import Tensor
from src.neurosym.training.callbacks import Callback
from src.neurosym.training.optimizer import AdamW

Pair = Tuple[int, int]
PairFact = Tuple[Pair, int]  # ((a, b), c_alvo)


def build_classifier(embedding_dim: int, hidden: int, p: int) -> Sequential:
    """MLP que mapeia [e_a ; e_b] para p logits (sem sigmoid na saída)."""
    return Sequential(
        Linear(2 * embedding_dim, hidden),
        ReLU(),
        Linear(hidden, hidden),
        ReLU(),
        Linear(hidden, p),
    )


def _stable_shift(logits: Tensor) -> Tensor:
    """Subtrai o máximo (como constante) para estabilidade numérica do softmax.

    O deslocamento por constante não altera o softmax nem seus gradientes.
    """
    max_val = max(Tensor._flatten(logits.data))
    return logits - Tensor(max_val)


def softmax_cross_entropy(logits: Tensor, target: int, p: int) -> Tensor:
    """CE = logsumexp(logits) - logits[target], composta por ops existentes.

    O logit-alvo é extraído por produto com um one-hot (o Tensor não tem
    indexação diferenciável).
    """
    shifted = _stable_shift(logits)
    log_sum = shifted.exp().sum().log()
    onehot = Tensor([[1.0 if i == target else 0.0] for i in range(p)])
    target_logit = shifted.dot(onehot).sum()
    return log_sum - target_logit


def softmax_probs(logits: Tensor) -> Tensor:
    shifted = _stable_shift(logits)
    exp = shifted.exp()
    return exp / exp.sum()


class SoftmaxTrainer:
    """Treinador da formulação por classificação, com a mesma interface
    pública que `training/trainer.py:Trainer` expõe ao runner multi-seed."""

    def __init__(
        self,
        grounding_env: Dict[str, Tensor],
        classifier: Sequential,
        p: int,
        epochs: int,
        lr: float = 1e-3,
        weight_decay: float = 1.0,
        lambda_semantic: float = 1.0,
        gamma_l1: float = 0.0,
        use_axioms: bool = True,
        val_eval_every: int = 1,
        callbacks: Optional[List[Callback]] = None,
    ):
        self.grounding_env = grounding_env
        self.classifier = classifier
        self.p = p
        self.epochs = epochs
        self.lambda_semantic = lambda_semantic
        self.gamma_l1 = gamma_l1
        self.use_axioms = use_axioms
        self.val_eval_every = max(1, val_eval_every)
        self.callbacks = list(callbacks) if callbacks else []
        self._last_val_accuracy: Optional[float] = None

        all_params = list(grounding_env.values()) + classifier.parameters()
        self.optimizer = AdamW(all_params, lr=lr, weight_decay=weight_decay)

        for cb in self.callbacks:
            cb.set_trainer(self)

    # --- interface usada pelo runner/reporting ---

    def callbacks_handler(self, method_name: str, *args):
        for cb in self.callbacks:
            getattr(cb, method_name)(*args)

    def _logits(self, a: int, b: int) -> Tensor:
        e_a = self.grounding_env[str(a)]
        e_b = self.grounding_env[str(b)]
        x = Tensor.concatenate([e_a, e_b], axis=1)
        return self.classifier(x)

    def evaluate_accuracy(self, facts: Optional[List[PairFact]]) -> Optional[float]:
        if not facts:
            return None
        correct = 0
        for (a, b), target in facts:
            flat = Tensor._flatten(self._logits(a, b).data)
            if flat.index(max(flat)) == target:
                correct += 1
        return correct / len(facts)

    def _semantic_loss(self, train_facts: List[PairFact]) -> Tensor:
        terms: List[Tensor] = []
        # comutatividade: distribuição(a,b) == distribuição(b,a)
        for (a, b), _ in train_facts:
            diff = softmax_probs(self._logits(a, b)) - softmax_probs(self._logits(b, a))
            terms.append((diff**2).sum())
        # identidade: (a, 0) -> a
        for a in range(self.p):
            terms.append(softmax_cross_entropy(self._logits(a, 0), a, self.p))
        return sum(terms, Tensor(0.0)) / len(terms)

    def _l1_penalty(self) -> Tensor:
        weights = self.classifier.l1_weight_parameters()
        if not weights:
            return Tensor(0.0)
        return sum((w.abs().sum() for w in weights), Tensor(0.0))

    def fit(
        self,
        rules,  # ignorado: os axiomas são internos à formulação (use_axioms)
        facts: List[PairFact],
        val_facts: Optional[List[PairFact]] = None,
    ):
        logs: Dict[str, float] = {}
        self.callbacks_handler("on_train_begin", logs)

        for epoch in range(self.epochs):
            self.callbacks_handler("on_epoch_begin", epoch, logs)
            self.optimizer.zero_grad()

            data_terms = [
                softmax_cross_entropy(self._logits(a, b), target, self.p)
                for (a, b), target in facts
            ]
            l_data = sum(data_terms, Tensor(0.0)) / len(data_terms)

            l_semantic = (
                self._semantic_loss(facts) if self.use_axioms else Tensor(0.0)
            )
            l1_penalty = self._l1_penalty()

            loss = (
                l_data
                + Tensor(self.lambda_semantic) * l_semantic
                + Tensor(self.gamma_l1) * l1_penalty
            )
            loss.backward()
            self.optimizer.step()

            logs["loss"] = loss.data
            logs["l_data"] = l_data.data
            logs["l_semantic"] = l_semantic.data
            logs["l1_penalty"] = l1_penalty.data

            if val_facts is not None:
                if epoch % self.val_eval_every == 0 or epoch == self.epochs - 1:
                    self._last_val_accuracy = self.evaluate_accuracy(val_facts)
                logs["val_accuracy"] = self._last_val_accuracy

            self.callbacks_handler("on_epoch_end", epoch, logs)

        self.callbacks_handler("on_train_end", logs)


def _pair_facts(pairs: List[Pair], p: int) -> List[PairFact]:
    return [((a, b), (a + b) % p) for a, b in pairs]


def make_softmax_build_fn(
    p: int,
    use_axioms: bool,
    embedding_dim: int = 16,
    hidden: int = 48,
    epochs: int = 3000,
    lr: float = 1e-3,
    weight_decay: float = 1.0,
    lambda_semantic: float = 1.0,
    gamma_l1: float = 0.0,
    train_frac: float = 0.30,
    val_frac: float = 0.35,
    val_eval_every: int = 10,
):
    """`build_fn(seed)` compatível com run_multiseed para a formulação softmax.

    weight_decay=1.0 por padrão: é o valor canônico da literatura de grokking
    para o setup de classificação (Power et al. 2022; Nanda et al. 2023).
    """
    from experiments.modular_addition.dataset import (
        build_grounding_env,
        generate_split,
    )

    def build_fn(seed: int) -> ExperimentSpec:
        data = generate_split(p, seed=seed, train_frac=train_frac, val_frac=val_frac)
        grounding_env = build_grounding_env(p, embedding_dim, seed)
        classifier = build_classifier(embedding_dim, hidden, p)

        trainer = SoftmaxTrainer(
            grounding_env,
            classifier,
            p=p,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            lambda_semantic=lambda_semantic if use_axioms else 0.0,
            gamma_l1=gamma_l1,
            use_axioms=use_axioms,
            val_eval_every=val_eval_every,
        )

        return ExperimentSpec(
            interpreter=None,  # a formulação softmax não usa o Interpreter FOL
            trainer=trainer,
            rules=[],
            facts=_pair_facts(data.train_pairs, p),
            val_facts=_pair_facts(data.val_pairs, p),
            test_facts=_pair_facts(data.test_pairs, p),
        )

    return build_fn
