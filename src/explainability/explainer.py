from typing import List, Tuple
import operator

from src.logic import Formula
from src.interpreter import Interpreter


def explain_inference(
    formula: Formula, interpreter: Interpreter, top_k: int = 5
) -> List[Tuple[str, float]]:

    for const_tensor in interpreter.grounding_env.values():
        const_tensor.requires_grad = True
        const_tensor.zero_grad()

    truth_value_tensor = interpreter.eval_formula(formula, {})

    truth_value_tensor.backward()

    influences = {}
    for name, tensor in interpreter.grounding_env.items():
        if tensor.grad:

            grad_magnitude = sum(abs(g) for g in tensor.grad._flatten(tensor.grad.data))
            influences[name] = grad_magnitude

    sorted_influences = sorted(
        influences.items(), key=operator.itemgetter(1), reverse=True
    )

    return sorted_influences[:top_k]
