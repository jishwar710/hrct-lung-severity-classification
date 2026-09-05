# Experimental Results

The feature-fusion classification pipeline was evaluated on four HRCT lung severity classes:

- Normal
- Mild
- Moderate
- Severe

## Reported Performance

| Metric | Value |
|--------|-------|
| Total patients processed | 131 |
| Feature size | 2048 |
| Test samples | 27 |
| Accuracy | 77.78% |
| Macro average precision | 0.82 |
| Macro average recall | 0.81 |
| Macro average F1-score | 0.81 |
| Weighted average F1-score | 0.78 |

## Classification Report

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| Normal | 1.00 | 1.00 | 1.00 | 2 |
| Mild | 0.67 | 0.57 | 0.62 | 7 |
| Moderate | 0.60 | 0.75 | 0.67 | 8 |
| Severe | 1.00 | 0.90 | 0.95 | 10 |

## Confusion Matrix

The confusion matrix obtained from the experiment was:

```text
[[2 0 0 0]
 [0 4 3 0]
 [0 2 6 0]
 [0 0 1 9]]
