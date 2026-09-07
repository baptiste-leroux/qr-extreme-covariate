# qr-extreme-covariate
Code for the numerical experiments (Section 4, "Numerical experiments") of

> Leroux, B., Dombry, C., and Sabourin, A. *Out-of-Distribution generalization
> of quantile regression with heavy tailed inputs: an SVM approach.*

The experiments compare an ERM approach (linear quantile regression) and an
SVM approach (Gaussian-kernel quantile regression) for extrapolating
conditional quantiles into the extreme-covariate regime, against naive
baselines, on daily river-flow measurements from 31 gauging stations of the
upper Danube basin.

## Repository layout

```
scripts/danube_experiments.py   the experiment script (data loading,
                                 preprocessing, cross-validation, and the
                                 figures used in the paper)
requirements.txt
NOTES_REVIEW.md                 correctness review notes and the fixes
                                 applied while cleaning up the original code
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

`liquidSVM` is the one dependency that may need extra care — see the note
in `requirements.txt`, including a one-line fix for a common import failure
on Python 3.11+ (`sysconfig.get_config_var('SO')` → `'EXT_SUFFIX'`).
Everything upstream of the SVM calls (data loading,
standardization, the ERM baselines) only needs `numpy`/`scipy`/`scikit-learn`.

## Running

```bash
python scripts/danube_experiments.py
```

The dataset is downloaded on the fly (see "Data" below), so no manual setup
is required. Running it top to bottom prints the selected `k_train` for
each of the four models (proposed/naive x SVM/ERM) and produces the three
figures from Section 4 as `.pdf` files in the current directory:
`tradeoff_bias_variance_ERM_and_SVM.pdf` (Figure 3),
`svm_pareto_and_normal_scale.pdf` (Figure 4), and
`comparison_with_baseline.pdf` (Figure 5). Figure 2 (the exploratory
asymptotic-dependence scatter plot) is displayed but not saved to disk.

Every source of randomness (the cross-validation splits, the final
train/test permutation) is seeded explicitly via the `RNG_SEED` constant at
the top of the script, so a run is reproducible regardless of how many
times, or in what order, its functions are called — see `NOTES_REVIEW.md`
for the one caveat this doesn't cover (liquidSVM's own internal
determinism, which could not be verified independently).

The functions in the script (`normalize`, `pareto_to_normal`, `fit_svm`,
`cross_validate_proposed`, etc.) are also importable without triggering the
experiment run, e.g. for unit tests or reuse elsewhere:

```python
from scripts.danube_experiments import normalize
```

## Data

The dataset is downloaded on the fly from the `graphicalExtremes` R package's
GitHub repository (`danube.rda`, Asadi et al., 2015) and parsed with the
`rdata` package — no manual download needed.

## Notes

See `NOTES_REVIEW.md` for the correctness review of the original code and
the fixes applied, plus what changed in the notebook-to-script conversion.
