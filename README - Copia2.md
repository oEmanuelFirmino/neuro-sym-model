# Neuro-Symbolic Core: A Framework From Scratch

This project is an implementation of a neuro-symbolic artificial intelligence framework, built entirely from scratch in Python. The goal is to provide a reusable and scalable engine that unites neural networks and symbolic logic, inspired by concepts such as Logic Tensor Networks (LTNs).

The system is capable of learning from facts and logical rules, adjusting numerical representations (embeddings) and neural models to satisfy a knowledge base. It has been designed to be flexible, performant, and auditable.

## Core Features

- **Tensor Engine with Dual Backend:** A custom Tensor class with support for automatic differentiation (backpropagation). It operates with a pure Python backend for portability or a NumPy backend for high performance.
- **Dynamic Neural Network Modules:** Build neural network architectures (Linear, ReLU, Sigmoid, etc.) dynamically from configuration files, without needing to change the code.

- **Configurable Logical Representation:** Represent first-order logic formulas (∀, →, ∧, etc.) and choose the fuzzy logic semantics (t-norms) and quantifier aggregators best suited for your problem.

- **Extensible Training Framework:** An abstract Trainer with a Callbacks system allows for cleanly injecting functionalities like ModelCheckpoint (to save the best model) and EarlyStopping.

- **Explainability and Governance:** Includes tools for auditing inferences, capable of identifying which entities in the domain most influenced a model's decision.

## Project Structure

```
neuro-symbolic-core/
  ├── benchmarks/ # Scripts for performance testing
  ├── examples/
  │ └── socrates/ # Example of how to use the library
  │ ├── config.yaml
  │ └── data/
  ├── src/ # The library's source code
  │ ├── explainability/
  │ │ └── explainer.py # (NEW) Inference audit tool
  │ ├── interpreter/
  │ │ ├── fuzzy_operators.py
  │ │ └── interpreter.py
  │ ├── logic/
  │ ├── module/
  │ │ └── factory.py # Neural network model factory
  │ ├── tensor/
  │ │ └── backend/ # Numerical backends (Python, NumPy)
  │ └── training/
  │ ├── callbacks.py # (NEW) Callback system
  │ └── trainer.py # Main training class
  └── ...
```

## How to Use

**1. Train the Model**

Run the training script from an example. The library's `Trainer` will orchestrate the process, and the `ModelCheckpoint` callback will save the best-performing model.

```python
# Run with the default backend (Python)
python examples/socrates/train.py

# To use the high-performance backend, set it at the beginning of your script:
# from src.tensor.backend import set_backend
# set_backend('numpy')

```

**2. Run Inference and Audits**

Use the inference script to ask a question to the trained model. Use the `--explain` flag to enable the governance report.

```python
# Simple query
python src/inference/infer.py --query "Mortal(socrates)"

# Query with an explainability report
python src/inference/infer.py --query "Mortal(socrates)" --explain

```

The explainability report will show which constants from your domain (e.g., `socrates`, `plato`) had the greatest impact (as measured by the gradient) on the result of your query.
