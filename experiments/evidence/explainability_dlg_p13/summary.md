# Explainability — DLG p=13 (memorized state)

Config: `{'p': 13, 'embedding_dim': 8, 'hidden': 24, 'epochs': 400, 'val_eval_every': 20, 'gamma_l1': 0.0001}`

| Metric | Mean ± std (n) |
|---|---|
| concentration | 1.0000 ± 0.0000 (n=25) |
| deletion_auc_gradient | 0.7118 ± 0.0644 (n=25) |
| deletion_auc_random | 0.7535 ± 0.2125 (n=25) |
| insertion_auc_gradient | 0.6984 ± 0.3345 (n=25) |
| insertion_auc_random | 0.6828 ± 0.2402 (n=25) |

Fidelity expectation: deletion_auc_gradient < deletion_auc_random and insertion_auc_gradient > insertion_auc_random.
