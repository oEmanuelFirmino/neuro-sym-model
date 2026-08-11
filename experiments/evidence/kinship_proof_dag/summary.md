# Kinship — proof-DAG composed inference vs flat predicate

| Metric | Composed (proof DAG) | Flat predicate |
|---|---|---|
| Truth on true ancestor pairs | 0.9419 ± 0.0166 | 0.9924 ± 0.0051 |
| Truth on false ancestor pairs | 0.0168 ± 0.0135 | 0.0031 ± 0.0053 |
| Gradient mass on path intermediates | 0.4466 ± 0.0863 | 0.0000 ± 0.0000 |
| Concentration on closure | 0.9996 ± 0.0001 | 1.0000 ± 0.0000 |
| Δ truth after deleting intermediates | 0.9308 ± 0.0205 | 0.0000 ± 0.0000 |

Composed inference is never trained on ancestor pairs: discrimination and path-dependence emerge from composing the learned parent predicate through the proof DAG (Product T-norms).
