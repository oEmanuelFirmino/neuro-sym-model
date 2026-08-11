# Axiom fidelity (held-out) + gamma_l1 sweep — p=13

## Part A — fidelity by architecture

| Architecture | Overall fidelity | Commutativity | Identity | Sparsity |
|---|---|---|---|---|
| dlg | 0.8816 | 0.8558 | 0.9984 | 0.0196 |
| mlp_baseline | 0.7773 | 0.8359 | 0.5110 | 0.0026 |
| ltn_baseline | 0.8792 | 0.8536 | 0.9954 | 0.0017 |

## Part B — gamma_l1 sweep (DLG)

| gamma_l1 | Sparsity (|w| < 1e-3) | Overall fidelity |
|---|---|---|
| 0.0001 | 0.0196 | 0.8816 |
| 0.001 | 0.3444 | 0.8570 |
| 0.01 | 0.8027 | 0.8197 |
| 0.1 | 0.8231 | 0.7299 |
