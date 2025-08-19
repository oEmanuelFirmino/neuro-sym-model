# Neuro-Symbolic Model From Scratch

This project is an implementation of a neuro-symbolic artificial intelligence framework, built entirely in Python without the use of deep learning libraries like TensorFlow or PyTorch. The goal is to demonstrate the fundamental principles that unite neural networks and symbolic logic, inspired by concepts such as Logic Tensor Networks (LTNs).

The system is capable of learning from facts and logical rules, adjusting numerical representations (embeddings) and neural models to satisfy a knowledge base.

---

## Features
- **Tensor Engine and Autograd**: A custom Tensor class with support for automatic differentiation (backpropagation).
- **Neural Network Modules**: Implementation of Linear, Sigmoid, and ReLU layers to build models.
- **Logical Representation**: Classes to construct an Abstract Syntax Tree (AST) for first-order logic formulas (∀, →, ∧, etc.).
- **Neuro-Symbolic Interpreter**: A mechanism that translates logical formulas into differentiable computations over tensors.
- **Training and Inference**: Command-line scripts to train the model from data files and to make new logical queries to a trained model.

---

## Project Structure
```
neuro-sym-model/
├── config.yaml             # Main configuration file
├── data/
│   └── socrates/           # Example of a problem
│       ├── domain.csv
│       ├── facts.csv
│       ├── rules.txt
│       └── test_facts.csv
├── src/
│   ├── data/
│   │   └── loader.py       # Data and rules loader
│   ├── inference/
│   │   └── infer.py        # Script to make queries
│   ├── interpreter/
│   │   └── interpreter.py  # The neuro-symbolic interpreter
│   ├── logic/
│   │   └── logic.py        # AST classes for logical formulas
│   ├── module/
│   │   └── module.py       # Base classes for neural networks (Module, Linear, etc.)
│   ├── tensor/
│   │   └── tensor.py       # The main Tensor class and autograd engine
│   └── training/
│       ├── optimizer.py    # SGD optimizer
│       ├── saver.py        # Functions to save and load models
│       └── train.py        # Main training and evaluation script
└── requirements.txt        # Project dependencies
```

---

## Installation and Setup

Clone the repository:
```
git clone <YOUR_REPOSITORY_URL>
cd neuro-sym-model
```

Create and activate a virtual environment:
```
python -m venv .venv
source .venv/bin/activate  # On Linux/macOS
.venv\Scripts\activate    # On Windows
```

Install the dependencies:
```
uv pip install -r requirements.txt
```

---

## How to Use

The system is controlled via the `config.yaml` file and executed from command-line scripts.

### 1. Define a Problem
To define a new problem, create a new folder inside `data/` and add the following files:

- **domain.csv**: Lists all the constants (entities) of your problem, one per line.
- **facts.csv**: Lists the known facts in the format `Predicate,Constant1,Constant2,TruthValue`.
- **rules.txt**: Lists the logical rules in the format `forall x: (Formula(x))`.
- **test_facts.csv**: Lists the facts for validation, in the same format as the training facts.

Then, update the `config.yaml` file to point to your new files and define your predicates and hyperparameters.

---

### 2. Train the Model
Run the training script from the project's root directory. It will read the `config.yaml` file, train the model, and save it to the path specified in `model_save_path`.

```
uv run src/training/train.py
```

Optionally, you can specify a different configuration file:
```
uv run src/training/train.py --config "path/to/another_config.yaml"
```

---

### 3. Make Inferences
Use the inference script to ask a question (query) to the trained model.

```
uv run src/inference/infer.py --query "Mortal(socrates)"
```

The script will load the saved model and the environment defined in `config.yaml` to evaluate your query and print the resulting truth value.
"""
