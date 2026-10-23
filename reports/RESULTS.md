# Recorded benchmark results

Synthetic contemporaneous failure classification; no advance-warning claim.

| Model | Selection AP | Test AP | Test ROC-AUC | Test Brier |
| --- | ---: | ---: | ---: | ---: |
| prevalence | 0.0207 | 0.0195 | 0.5000 | 0.0198 |
| logistic | 0.5158 | 0.4129 | 0.8828 | 0.0153 |
| boosting_raw | 0.4225 | 0.5903 | 0.9603 | 0.0120 |
| boosting_physics | 0.6110 | 0.6352 | 0.9540 | 0.0112 |

Model selected before test evaluation: **boosting_physics**.
Policy threshold selected on UDI 7001-8000: **0.011717**.

Test: 34 detected failures, 5 missed failures, 118 false alerts; recall 87.2%, precision 22.4%, alert rate 7.6%.

Illustrative cost: **243 units**, compared with 975 for never alerting and 438 at the model's default 0.5 threshold.

Costs are assumptions, not measured maintenance expenses or money saved.

## Cost sensitivity

| Miss cost / false-alert cost | Policy-set threshold | Test cost | Recall | False alerts |
| ---: | ---: | ---: | ---: | ---: |
| 5:1 | 0.354038 | 92 | 61.5% | 17 |
| 10:1 | 0.354038 | 167 | 61.5% | 17 |
| 25:1 | 0.011717 | 243 | 87.2% | 118 |
| 50:1 | 0.011717 | 368 | 87.2% | 118 |
| 100:1 | 0.011717 | 618 | 87.2% | 118 |

Every sensitivity threshold was selected on the policy block, not the test block.

## Conditional 95% uncertainty intervals

- average_precision: 0.4213 to 0.7983
- recall: 0.7670 to 0.9583
- cost_per_1000: 77.4500 to 169.2625

These ordered-block bootstrap intervals condition on this fitted model and policy. They do not include model-selection, training, or deployment-shift uncertainty.

Full measurements and provenance: [results.json](results.json).
