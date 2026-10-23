# Validation record

Recorded 2026-10-23 on Python 3.12.14, NumPy 2.3.5, pandas 3.0.1.

- 20 unittest tests passed in 1.679 seconds.
- Full fixed benchmark completed successfully.
- All four review-notebook code cells executed successfully.
- CLI inference on ten example rows agrees with the recorded selected-model test scores (relative tolerance 1e-12) and exact alert labels.
- Source, plan, and dataset digests match the recorded run.
- Both generated SVG charts parse as XML.
- Every project Python source compiles.
- One completed benchmark run; no test-based retuning.

The initial scikit-learn prototype could not import because Windows Application Control blocked its binary. It was replaced before training. The published implementation requires only NumPy and pandas.

This is local validation, not a GitHub Actions run or an external deployment test.
