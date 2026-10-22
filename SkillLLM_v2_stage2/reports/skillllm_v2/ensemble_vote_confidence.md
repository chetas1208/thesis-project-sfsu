# Ensemble agreement as a confidence signal (held-out test, 974 docs)

The anchor assigns every span the constant confidence 0.86 while being right 67% of the time (AUROC 0.5, see the
calibration eval). The number of adapters (anchor, QLoRA-loop v1, v2) that predict the same exact span (start, end, label)
is a usable, monotone confidence instead. Reproduce with `PYTHONPATH=src python scripts/vote_precision.py`.

| Adapters agreeing | Spans | Correct | Precision |
|---|---|---|---|
| 1 of 3 | 941 | 273 | 0.2901 |
| 2 of 3 | 644 | 351 | 0.5450 |
| 3 of 3 | 1,428 | 1,145 | 0.8018 |

Operating points (keep spans with at least v votes): v=3 P 0.802 / R 0.506 / F1 0.620; **v=2 P 0.722 / R 0.661 / F1 0.690**;
v=1 (union) P 0.587 / R 0.781 / F1 0.670.

Caveats: the table is estimated on the test split itself (the only split with predictions from all three adapters), so it
describes the signal rather than being a fitted calibrator; re-fit the three numbers on your own held-out data. The adapters
share a base model and training data, so agreement is correlated evidence, and the 3x generation cost applies.
