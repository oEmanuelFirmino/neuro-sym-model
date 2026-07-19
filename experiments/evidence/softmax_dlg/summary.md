# softmax_dlg

## Config

```json
{
  "p": 13,
  "embedding_dim": 16,
  "hidden": 48,
  "epochs": 3000,
  "weight_decay": 1.0,
  "val_eval_every": 10,
  "use_axioms": true
}
```

## Metrics

| Metric | Value |
|---|---|
| Seeds | [0, 1] |
| T_g | — |
| Final val accuracy | 0.3136 ± 0.04237 (n=2) |
| Test accuracy | 0.3333 ± 0.05 (n=2) |
| Final L1 penalty | 859.8 ± 21.34 (n=2) |
