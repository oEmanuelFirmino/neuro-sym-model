# Kinship — proof-DAG composed inference vs flat predicate (multi-seed)

Seeds: [0, 1, 2, 3, 4] (n=5) · 500 épocas · agregação mean ± std ENTRE sementes.

| Metric | Composed (proof DAG) | Flat predicate |
|---|---|---|
| Truth on true ancestor pairs | 0.8378 ± 0.0799 | 0.9923 ± 0.0020 |
| Truth on false ancestor pairs | 0.0166 ± 0.0063 | 0.0038 ± 0.0010 |
| Gradient mass on path intermediates | 0.4544 ± 0.0420 | 0.0000 ± 0.0000 |
| Concentration on closure | 0.9971 ± 0.0028 | 1.0000 ± 0.0000 |
| Δ truth after deleting intermediates | 0.8207 ± 0.0804 | 0.0000 ± 0.0000 |

Composed inference is never trained on ancestor pairs: discrimination and path-dependence emerge from composing the learned parent predicate through the proof DAG (Product T-norms). Dispersion is now across independent seeds (initialization variance), not across queries of a single model.
