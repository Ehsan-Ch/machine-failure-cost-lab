# Method and experiment contract

## Question

For six available sensor/type inputs, compare rankings of current failure and choose an alert threshold under assumed error costs. A missed positive costs 25 units and a false alarm costs 1 unit in the main experiment. There is no asserted cost for correctly handled observations, no model of intervention effectiveness, and no measured financial return.

## Features

Only Type, air/process temperature, rotational speed, torque, and tool wear enter a model. Type uses two indicators (H and M; L is the reference). Neither UDI, Product ID, Machine failure, nor any of TWF/HDF/PWF/OSF/RNF can enter the feature matrix.

The augmented candidate adds:

- Process temperature minus air temperature, in kelvin.
- Speed times torque times 2*pi/60, in watts.
- Tool wear times torque, in minute-newton-metres.

These combinations are informed by the dataset's documented construction. No published failure cutoff is hardcoded as a classifier.

## Estimators

All implementations are in NumPy and deterministic.

**Prevalence:** the training-set positive fraction for every observation.

**Logistic regression:** standardize each matrix column using training means and population standard deviations, append an intercept, and minimize summed binary log loss plus half the squared non-intercept coefficients. A Newton solver uses backtracking and a tiny Hessian diagonal stabilizer, at most 100 iterations and a maximum parameter-step tolerance of 1e-8. Constant columns have scale 1.

**Newton tree boosting:** initialize log odds at the training prior. At each of 200 rounds, fit a tree to the logistic gradient p-y and Hessian p*(1-p), with Hessians floored at 1e-8. Leaf values are -G/(H+1), clipped to [-5,5]. Greedy splits maximize the regularized second-order gain. Trees have maximum depth 4 and at least 20 training observations per leaf. The learning rate is 0.08.

Each column uses up to 63 unique internal quantile cut points from the training block, yielding at most 64 bins. Candidate split points, means, scales, coefficients, and trees never use selection, policy, or test observations. Empty quantile bins can occur with discrete inputs.

This follows standard regularized tree-boosting mathematics; it is not an XGBoost implementation and does not claim performance equivalence. Reference: [XGBoost's mathematical introduction](https://xgboost.readthedocs.io/en/stable/tutorials/model.html).

## Model and policy selection

Choose the candidate with maximum average precision on UDI 5501–7000. Do not refit on that block. Fit its alert threshold on UDI 7001–8000 by exhaustive evaluation of unique score thresholds, zero, and a threshold just above one (never alert).

Alert means score >= threshold. Score ties are indivisible. Cost ties prefer fewer alerts and then the higher threshold. Also freeze thresholds for miss costs 5, 10, 50, and 100 before touching the test metrics. The code writes the frozen decisions first.

Separate threshold-selection data follow the principle described in [scikit-learn's cost-sensitive learning example](https://scikit-learn.org/stable/auto_examples/model_selection/plot_cost_sensitive_learning.html). This project implements its own discrete threshold search.

## Measurements

Average precision sums precision times each recall increment at distinct score thresholds. ROC-AUC uses trapezoidal integration of the threshold-group ROC curve, assigning half credit to tied positive/negative pairs. Brier score is the mean squared score error.

Confusion counts, recall, precision, alert rate, and assumed error cost describe the frozen policy. Ranking metrics and their tied-score behavior are tested against hand calculations and a pairwise AUC definition.

Uncertainty uses 500 resamples of non-overlapping 100-row test blocks with replacement, seed 123. Each resample keeps the trained model and threshold fixed. Percentiles 2.5 and 97.5 are reported. Samples lacking both classes are skipped and counted. These are conditional descriptive intervals, not full deployment confidence guarantees.

## Reproducibility

The source CSV has an exact-byte SHA-256 contract. Source-code and plan digests normalize text newlines to LF for cross-platform comparison. Results record Python and package versions, split counts, frozen threshold, and raw predictions. Numerical differences across BLAS/platform implementations may change very close threshold decisions.

The original plan was revised before training because Windows Application Control rejected an added scikit-learn binary. No benchmark was run with that package. The final experiment uses only the already available NumPy and pandas runtime. There was one completed benchmark run and no tuning based on its test results.
