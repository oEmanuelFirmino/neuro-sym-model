# Hinton family (1986) — deduced relations via proof DAG (multi-seed)

Seeds: [0, 1, 2, 3, 4] (n=5) · 500 épocas · agregação mean ± std ENTRE sementes.
Base-fact fit accuracy (todos os fatos base treinados): 0.997 ± 0.002

## Deductive consistency (never trained on derived relations)

| Relation | Composed pos | Composed neg | Flat (trained) pos | Flat neg |
|---|---|---|---|---|
| uncle | 0.483 ± 0.429 | 0.002 ± 0.001 | 0.975 ± 0.006 | 0.001 ± 0.001 |
| aunt | 0.556 ± 0.452 | 0.000 ± 0.000 | 0.970 ± 0.006 | 0.001 ± 0.001 |
| nephew | 0.669 ± 0.171 | 0.001 ± 0.001 | 0.972 ± 0.008 | 0.001 ± 0.001 |
| niece | 0.677 ± 0.174 | 0.001 ± 0.001 | 0.974 ± 0.007 | 0.001 ± 0.001 |

## Architectural explainability

| Metric | Composed | Flat |
|---|---|---|
| Gradient mass on path intermediate | 0.336 ± 0.069 | 0.000 ± 0.000 |
| Δ truth after deleting intermediate | 0.571 ± 0.079 | 0.000 ± 0.000 |

Dispersion is across independent seeds (initialization variance), not across queries of a single model.
