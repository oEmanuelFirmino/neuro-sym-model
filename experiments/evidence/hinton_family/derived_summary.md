# Hinton family (1986) — deduced relations via proof DAG

Base-fact held-out accuracy: 0.9304 | training: 1196s

## Deductive consistency (never trained on derived relations)

| Relation | Composed pos/neg | Flat (trained) pos/neg |
|---|---|---|
| uncle | 0.945 / 0.003 | 0.965 / 0.001 |
| aunt | 0.954 / 0.000 | 0.980 / 0.000 |
| nephew | 0.752 / 0.002 | 0.974 / 0.000 |
| niece | 0.778 / 0.000 | 0.971 / 0.000 |

## Architectural explainability

| Metric | Composed | Flat |
|---|---|---|
| Gradient mass on path intermediate | 0.3847 | 0.0000 |
| Δ truth after deleting intermediate | 0.7706 | 0.0000 |
