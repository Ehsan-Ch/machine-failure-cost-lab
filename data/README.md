# Dataset provenance

**Dataset:** AI4I 2020 Predictive Maintenance Dataset.

**Provider:** UCI Machine Learning Repository.

**DOI:** https://doi.org/10.24432/C5HS5C

**License:** [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).

**Related paper:** S. Matzka (2020), “Explainable Artificial Intelligence for Predictive Maintenance Applications.”

UCI describes a synthetic industrial-maintenance dataset with 10,000 observations and six input features. The downloaded file also includes identifiers, a machine-failure target, and failure-mode indicators. Its labels concern failure at the observation, without a supplied future prediction horizon.

Downloaded 2026-10-23 from:
https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip

The archive's ai4i2020.csv member is redistributed byte-for-byte without edits. Only that member is used. The source ZIP is not committed.

| File | SHA-256 |
| --- | --- |
| Source ZIP | f601f14294bcf190f9d720676b7f0aea46a26cde9ab8ebc7b4f8174d9d26b252 |
| CSV | dc6630cd9b1f0f853922fad78a1b6436570d3f1ec863f1dd5c4340ac56bc8a8e |

The ZIP digest is documented separately in provenance.json. The training loader enforces the CSV digest.

Our inspection finds 339 positive machine-failure labels and 9,661 negatives. The fixed test block contains 39 positives. Do not use the five failure-mode indicators as predictors of the aggregate target.

Derived examples and prediction tables identify this source and retain CC BY 4.0 attribution. No endorsement by UCI or the dataset author is implied.
