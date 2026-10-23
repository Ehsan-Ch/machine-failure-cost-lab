# Model card

**Name:** Machine Failure Cost Lab / boosting_physics

**Owner:** Ehsan Cheraghi

**Version:** 1.0.0

**Artifact:** reports/model.json, a plain JSON description of preprocessing and trees.

**Intended use:** teaching, code review, reproducible comparison on the bundled synthetic AI4I benchmark, and exploration of assumed false-alert/missed-failure tradeoffs.

**Not established:** safe operational maintenance decisions, prediction before a failure occurs, remaining useful life, performance on real equipment, calibrated probabilities, intervention effectiveness, or monetary savings.

**Training:** 5,500 original ordered observations; 247 failure labels. Two further disjoint blocks select the model and threshold. The model is not refitted after selection.

**Recorded evaluation:** 2,000 held-out rows, 39 failures. AP 0.6352, ROC-AUC 0.9540. The frozen 0.011716755082956706 threshold detects 34 failures, misses 5, and raises 118 false alarms. Recall is 87.2%, precision 22.4%.

**Failure modes:** low precision, distribution shift, incorrect units, synthetic label rules, unobserved machine history, and rare/unpredictable failures. Validation checks basic input form and signs, not every physically plausible range. The tool does not perform out-of-distribution detection or human review.

**Uncertainty:** conditional 95% test-block bootstrap intervals are AP 0.4213–0.7983 and recall 0.7670–0.9583. They exclude fitting/selection and deployment uncertainty.

**Safeguards against misleading evaluation:** identifiers and failure-mode labels are excluded; preprocessing uses training data only; the policy is selected separately; test outcomes are reported with false alarms and limitations.

**Next evaluation:** acquire an independently consented and licensed real-machine dataset with timestamps and equipment identities, define a future event window, agree on measured costs, and reserve a new test period before further modeling.
