# BA 305 Project — Team A7

**Authors:** Gillian Lee, Sean Yi, Linh Bui, Shipra Kalve

San Francisco rents and building permits, using the [TidyTuesday 2022-07-05](https://github.com/rfordatascience/tidytuesday/tree/main/data/2022/2022-07-05) datasets.

## Questions

**Descriptive (rents):** Between 2004–06 and 2014–16, did 2-bedroom asking rents rise faster in San Francisco neighborhoods that started in the cheapest third than in those that started in the priciest third?

**Predictive (permits):** Can the details on a San Francisco building permit application predict whether the project will be abandoned rather than completed?
- Target: `y = 1` if the permit did not finish (expired, cancelled, withdrawn, disapproved, revoked or denied), `y = 0` if completed.
- Audience: the SF Department of Building Inspection (DBI), to prioritize outreach to high-risk permits at filing.

## Repository

| File | What it does |
|---|---|
| `permits_data.py` | Loads and cleans the permits data. Every predictive notebook imports `get_permit_data()` from here, so all models use the same rows, features, grouped train/test split, preprocessing and CV folds. |
| `01_permit_data_overview.ipynb` | Explores the raw permits data and shows the evidence behind each cleaning step in `permits_data.py` (filters, leakage, features, grouping, imputation). |
| `02_logistic.ipynb` | Logistic regression: tuning (`C`, class weight), fit diagnosis, test results, odds ratios. |
| `03_tree_decision.ipynb` | Decision tree and random forest: split criteria, tuning, decision rules, forest tuning, tree stability, feature importance. |
| `04_neural_network.ipynb` | Neural network (MLP): design, input scaling, training curves, tuning in two rounds, seed stability, test results. |
| `05_model_comparison.ipynb` | Compares all four models on the same test set: AUC, recall and precision, risk ranking, feature importance, validation on later permits, the outreach decision policy and error analysis. |
| `06_rents_charts.ipynb` | Descriptive question: rent data cleaning and charts. |
| `07_descriptive_pca.ipynb` | Descriptive question: neighborhood EDA and PCA. |
| `rentdata.ipynb` | Loads the rent dataset. |
| `archive_permits_prep.ipynb` | Original permits prep notebook, kept for reference. Replaced by `permits_data.py` and `01`. |

## Predictive results

Test set: 12,472 permits (20%, grouped by project), of which 36.7% did not finish.

| Model | Test AUC | Accuracy | Recall | Precision | AUC on 2011–13 permits |
|---|---|---|---|---|---|
| Baseline: always "completed" | 0.500 | 0.633 | 0.000 | — | — |
| Logistic regression | 0.633 | 0.585 | 0.605 | 0.452 | 0.601 |
| Decision tree | 0.644 | 0.639 | 0.248 | 0.519 | 0.598 |
| **Random forest** | **0.698** | **0.669** | 0.335 | **0.588** | 0.614 |
| Neural network | 0.659 | 0.648 | 0.303 | 0.538 | 0.614 |

Recall and precision use the default 0.5 cutoff.

- **Recommended model: random forest.** It has the highest AUC, and every gap to the other models is confirmed by bootstrap intervals. 65% of its riskiest 10% of permits fail, against a 37% base rate.
- **Decision:** with a cutoff of 0.34 (a missed failure assumed to cost twice a false alarm), the forest costs about 24% less than contacting every permit, the best policy without a model.
- **Limits:** filing year carries much of the signal, and performance falls on later permits (AUC ≈ 0.61), so the model would need retraining on recent data. Missed failures often have applications that look like completed ones; adding the permit description is the next step.
