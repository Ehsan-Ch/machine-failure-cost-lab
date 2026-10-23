# Development record

2026-10-23:

1. Selected a new industrial ML portfolio project and obtained the public UCI source archive.
2. Verified the exact CSV and defined a four-way ordered evaluation plan.
3. Implemented the original prototype with scikit-learn; Windows Application Control prevented its binary from loading, so no benchmark results were produced by that prototype.
4. Replaced that dependency with inspectable NumPy logistic regression, Newton tree boosting, ranking metrics, and JSON serialization. Revised the experiment plan before training.
5. Passed 20 tests, including leakage invariance, preprocessing isolation, hand-computed metrics, Newton leaf values, minimum leaf sizes, simple-signal learning, and JSON round trips.
6. Ran one fixed benchmark. Physics-feature boosting was selected by selection-block AP. The policy was frozen before test evaluation.
7. Recorded every candidate, the chosen policy, cost sensitivity, conditional uncertainty, and raw test predictions. No model or policy retuning followed the test results.
8. Prepared source, documentation, examples, a review notebook, and a Git-ready archive.

The repository contains the final implementation. Failed dependency downloads and environment-specific binaries are excluded. The results are locally measured; this is not a Kaggle submission or an external leaderboard result.
