# Expanded Gold 120-Row Agreement Audit

Annotate the blind CSV without looking at gold labels, candidate labels, model predictions, or repair logs.

Allowed primary labels: nan_inf, overflow_underflow, precision_tolerance, dtype_casting, crash_compile, performance_only, not_numerical_failure.

For each row, read the issue URL and any public issue context needed. Fill primary failure, true numerical failure status, confidence, evidence quote, and notes.

After two independent passes, compare against the expanded gold label and compute observed agreement, Cohen's kappa, and disagreements by class.
