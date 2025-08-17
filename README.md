# Modelo Neuro-Simbólico a partir do Zero

Este projeto é uma implementação de um framework de inteligência artificial neuro-simbólica, construído inteiramente em Python sem o uso de bibliotecas de deep learning como TensorFlow ou PyTorch. O objetivo é demonstrar os princípios fundamentais que unem redes neurais e lógica simbólica, inspirados em conceitos como Logic Tensor Networks (LTNs).

O sistema é capaz de aprender a partir de factos e regras lógicas, ajustando representações numéricas (embeddings) e modelos neurais para satisfazer uma base de conhecimento.

## Funcionalidades

- **Motor de Tensores e Autograd:** Uma classe `Tensor` customizada com suporte para diferenciação automática (backpropagation).
- **Módulos de Rede Neural:** Implementação de camadas `Linear`, `Sigmoid` e `ReLU` para construir modelos.
- **Representação Lógica:** Classes para construir uma Árvore de Sintaxe Abstrata (AST) de fórmulas de lógica de primeira ordem (∀, →, ∧, etc.).
- **Interpretador Neuro-Simbólico:** Um mecanismo que traduz fórmulas lógicas em computações diferenciáveis sobre tensores.
- **Treino e Inferência:** Scripts de linha de comando para treinar o modelo a partir de ficheiros de dados e para fazer novas consultas lógicas a um modelo treinado.

## Estrutura do Projeto

```
neuro-sym-model/
├── config.yaml             # Ficheiro principal de configuração
├── data/
│   └── socrates/           # Exemplo de um problema
│       ├── domain.csv
│       ├── facts.csv
│       ├── rules.txt
│       └── test_facts.csv
├── src/
│   ├── data/
│   │   └── loader.py       # Carregador de dados e regras
│   ├── inference/
│   │   └── infer.py        # Script para fazer consultas
│   ├── interpreter/
│   │   └── interpreter.py  # O interpretador neuro-simbólico
│   ├── logic/
│   │   └── logic.py        # Classes da AST para fórmulas lógicas
│   ├── module/
│   │   └── module.py       # Classes base para redes neurais (Module, Linear, etc.)
│   ├── tensor/
│   │   └── tensor.py       # A classe principal Tensor e o motor de autograd
│   └── training/
│       ├── optimizer.py    # Otimizador SGD
│       ├── saver.py        # Funções para salvar e carregar modelos
│       └── train.py        # Script principal de treino e avaliação
└── requirements.txt        # Dependências do projeto
```

## Instalação e Configuração

1.  **Clone o repositório:**

    ```bash
    git clone <URL_DO_SEU_REPOSITORIO>
    cd neuro-sym-model
    ```

2.  **Crie e ative um ambiente virtual:**

    ```bash
    python -m venv .venv
    source .venv/bin/activate  # No Linux/macOS
    .venv\Scripts\activate    # No Windows
    ```

3.  **Instale as dependências:**
    ```bash
    uv pip install -r requirements.txt
    ```

## Como Usar

O sistema é controlado através do ficheiro `config.yaml` e executado a partir de scripts de linha de comando.

### 1. Definir um Problema

Para definir um novo problema, crie uma nova pasta dentro de `data/` e adicione os seguintes ficheiros:

- **`domain.csv`**: Lista todas as constantes (entidades) do seu problema, uma por linha.
- **`facts.csv`**: Lista os factos conhecidos no formato `Predicado,Constante1,Constante2,GrauDeVerdade`.
- **`rules.txt`**: Lista as regras lógicas no formato `forall x: (Formula(x))`.
- **`test_facts.csv`**: Lista os factos para validação, no mesmo formato dos factos de treino.

Depois, atualize o `config.yaml` para apontar para os seus novos ficheiros e definir os seus predicados e hiperparâmetros.

### 2. Treinar o Modelo

Execute o script de treino a partir do diretório raiz do projeto. Ele irá ler o `config.yaml`, treinar o modelo e salvá-lo no caminho especificado em `model_save_path`.

```bash
uv run src/training/train.py
```

Opcionalmente, pode especificar outro ficheiro de configuração:

```bash
uv run src/training/train.py --config "caminho/para/outra_config.yaml"
```

### 3. Fazer Inferência

Use o script de inferência para fazer uma pergunta (consulta) ao modelo que acabou de ser treinado.

```bash
uv run src/inference/infer.py --query "Mortal(socrates)"
```

O script irá carregar o modelo salvo e o ambiente definido no `config.yaml` para avaliar a sua consulta e imprimir o grau de verdade resultante.
