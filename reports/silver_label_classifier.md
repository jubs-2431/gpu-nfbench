# Silver-Label Classifier

This experiment predicts the agent-assisted silver primary label from issue title and body excerpt. It is not a gold-standard evaluation.

- Train issues: 655
- Test issues: 156
- Majority baseline accuracy: 0.321
- Naive Bayes silver-label accuracy: 0.487
- Naive Bayes macro F1: 0.317

| label | support | F1 |
| --- | ---: | ---: |
| crash_compile | 7 | 0.000 |
| dtype_casting | 50 | 0.557 |
| nan_inf | 40 | 0.593 |
| overflow_underflow | 25 | 0.531 |
| performance_only | 7 | 0.000 |
| precision_tolerance | 27 | 0.222 |
