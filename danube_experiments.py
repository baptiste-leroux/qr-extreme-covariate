"""Danube river-flow experiments -- extreme quantile regression (SVM vs ERM).

Reproduces the numerical experiments of Section 4 ("Numerical experiments")
of *Out-of-Distribution generalization of quantile regression with heavy
tailed inputs: an SVM approach* (Leroux, Dombry, Sabourin).
"""

import warnings

import numpy as np
import requests
import rdata
import scipy.stats as st
import matplotlib.pyplot as plt
from sklearn.linear_model import QuantileRegressor
from sklearn.model_selection import RepeatedKFold
from sklearn.utils import Bunch
from liquidSVM import qtSVM

warnings.filterwarnings("ignore", category=UserWarning, module="rdata")

# --- Experiment configuration (Section 4.1) --------------------------------
RNG_SEED = 42

DANUBE_URL = (
    "https://raw.githubusercontent.com/sebastian-engelke/"
    "graphicalExtremes/master/data/danube.rda"
)

# Target station: the paper (1-indexed) focuses on station 18, which is
# column index 17
TARGET_STATION = 17
# Quantile threshold used to fit the GPD tail during marginal standardization.
STANDARDIZATION_QUANTILE = 0.9
# Fraction of the (validation/test) sample kept as "extreme" when estimating
# the asymptotic risk, tau_test in the paper.
TAU_TEST = 0.01
# Fraction of the data used for training in the (inverted) cross-validation
# scheme: 20% train / 80% validation, see Section 4.1 "Selection of
# hyper-parameters".
TRAIN_FRACTION = 0.2

N_FOLDS = 5
N_PERMUTATIONS = 100
N_PERMUTATIONS_NAIVE = 10

# Hyperparameter grid for k_train (number of extreme observations retained
# for training), shared by every cross-validation loop below.
K_LIST = np.hstack([
    np.arange(90, 500, 10),
    np.arange(500, 800, 50),
])


# =============================================================================
# 1. Data loading
# =============================================================================

def load_data():
    """Download the Danube dataset and drop incomplete rows.

    Returns
    -------
    data : ndarray of shape (n, m)
    """
    response = requests.get(DANUBE_URL, timeout=30)
    response.raise_for_status()
    parsed = rdata.parser.parse_data(response.content)
    converted = rdata.conversion.convert(parsed)
    data_raw = converted["danube"]["data_raw"].to_numpy()

    complete_rows = ~np.isnan(data_raw).any(axis=1)
    data = data_raw[complete_rows]

    print(f"{data_raw.shape[0]} daily records, {data_raw.shape[0] - data.shape[0]} dropped "
          f"for missing values -> n = {data.shape[0]} complete observations, "
          f"m = {data.shape[1]} stations")
    return data


# =============================================================================
# 2. Marginal standardization to unit-Pareto margins
# =============================================================================
def normalize(x, q):
    """Standardize a 1-D sample to (approximately) unit-Pareto margins.

    A GPD is fitted to the exceedances above the q-quantile; values below
    that threshold are standardized using the empirical CDF.

    Parameters
    ----------
    x : ndarray of shape (n,)
    q : float
        Quantile level of the standardization threshold.

    Returns
    -------
    ndarray of shape (n,)
        Standardized (unit-Pareto-margin) sample.
    """
    n = np.size(x)
    ranks = np.argsort(np.argsort(x))  # 0-indexed rank of each observation
    u = np.quantile(x, q)
    exceeds = x > u

    shape, _, scale = st.genpareto.fit(x[exceeds], floc=u)
    x_standard = np.empty(n)
    z_exceeds = (x[exceeds] - u) / scale
    x_standard[exceeds] = 1 / ((1 - q) * (1 + shape * z_exceeds) ** (-1 / shape))
    x_standard[~exceeds] = 1 / (1 - ranks[~exceeds] / n)
    return x_standard


def pareto_to_normal(x, v, q):
    """Inverse of `normalize`: map unit-Pareto-margin values back to x's scale.

    Parameters
    ----------
    x : ndarray of shape (n,)
        Original sample that `normalize(x, q)` was applied to.
    v : ndarray of shape (m,)
        Unit-Pareto-margin values to back-transform (e.g. model predictions).
    q : float
        Quantile level used in the forward standardization.

    Returns
    -------
    ndarray of shape (m,)
    """
    n = np.size(x)
    u = np.quantile(x, q)
    exceeds = x > u

    shape, _, scale = st.genpareto.fit(x[exceeds], floc=u)
    x_sorted = np.sort(x)

    threshold = 1 / (1 - q)
    result = np.empty_like(v, dtype=float)
    for i, v_i in enumerate(v):
        if v_i > threshold:
            # GPD branch: exact inverse of the exceedance formula above.
            result[i] = ((v_i * (1 - q)) ** shape - 1) * scale / shape + u
        elif v_i >= 1:
            # Bulk branch: exact inverse of V = 1 / (1 - rank/n).
            cdf = 1 - 1 / v_i
            result[i] = x_sorted[min(int(n * cdf), n - 1)]
        else:
            # v_i < 1 falls outside the range normalize() can ever produce
            # (V >= 1 always). This can only happen for a model prediction
            # clipped to 0 by the ReLU-like step below; fall back to the
            # smallest observed value rather than raising or silently
            # producing a biased result.
            result[i] = x_sorted[0]
    return result


# =============================================================================
# 4. Exploratory plot -- asymptotic dependence (Figure 2)
# =============================================================================
# For the target station, we look at the joint behaviour of Z (the target
# station's value) and ||X||_inf (the largest of the other 30 stations) on
# the 5% most extreme observations, and color points by which station
# attains the max of ||X||_inf.

def infinity_norm(x):
    return np.linalg.norm(x, ord=np.inf, axis=1)


def max_norm_station_labels(x, excluded_station):
    """Original (0-indexed) station attaining the row-wise infinity norm of x,
    where `x` has already had `excluded_station` removed as a column."""
    argmax_in_x = np.argmax(x, axis=1)
    shifted_back = argmax_in_x + (argmax_in_x >= excluded_station)
    return shifted_back


def plot_asymptotic_dependence(data_pareto):
    """Figure 2: log-log scatter of Z and Y=Z/||X|| against ||X|| for the
    target station, on the 5% most extreme observations."""
    Z = data_pareto[:, TARGET_STATION]
    X = np.delete(data_pareto, TARGET_STATION, axis=1)
    Y = Z / infinity_norm(X)

    # Stations 15-18 (1-indexed) = columns 14-17 (0-indexed) are
    # flow-connected and highlighted in red, per the paper's Figure 2 caption.
    station_colors = np.array(["blue"] * 14 + ["red"] * 4 + ["blue"] * 13)
    point_colors = station_colors[max_norm_station_labels(X, TARGET_STATION)]

    norm_X = infinity_norm(X)
    extreme_5pct = norm_X > np.quantile(norm_X, 0.95)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].scatter(norm_X[extreme_5pct], Z[extreme_5pct], c=point_colors[extreme_5pct])
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel(r"$\|X\|_\infty$")
    axes[0].set_ylabel("Z")

    axes[1].scatter(norm_X[extreme_5pct], Y[extreme_5pct], c=point_colors[extreme_5pct])
    axes[1].set_xscale("log")
    axes[1].set_xlabel(r"$\|X\|_\infty$")
    axes[1].set_ylabel(r"$Y = Z / \|X\|_\infty$")

    fig.tight_layout()
    plt.show()


# =============================================================================
# 5. Data preparation and model fitting
# =============================================================================


def pinball_loss(y_pred, y_true, tau=0.5):
    """Empirical pinball (quantile) risk at level tau (tau=0.5: median / MAE-like)."""
    diff = y_true - y_pred
    return np.mean(np.maximum(tau * diff, (tau - 1) * diff))


def extract_target_station(data, station):
    """Split a standardized dataset into covariates X and ratio target y = Z/||X||."""
    z = data[:, station].astype(np.float64)
    x = np.delete(data, station, axis=1).astype(np.float64)
    y = z / infinity_norm(x)
    return x, y


def angular_features_sorted_by_extremeness(x, y):
    """Angular components of x, and y, sorted by decreasing ||x||."""
    norms = infinity_norm(x)
    angular_x = x / norms[:, np.newaxis]
    order = np.argsort(-norms)
    return angular_x[order], y[order]


def fit_svm(x, y, k_train):
    """Algorithm 2: Gaussian-kernel quantile SVM (median), on the k_train most extreme rows."""
    angular_x, y_sorted = angular_features_sorted_by_extremeness(x, y)
    return qtSVM(angular_x[:k_train], y_sorted[:k_train], weights=[0.5])


def fit_erm(x, y, k_train):
    """Algorithm 1: linear quantile regression (median), on the k_train most extreme rows."""
    angular_x, y_sorted = angular_features_sorted_by_extremeness(x, y)
    return QuantileRegressor(alpha=0, quantile=0.5).fit(angular_x[:k_train], y_sorted[:k_train])


def scatter_true_vs_pred(true, pred, ax, title):
    ax.scatter(true, pred, marker="o", c="blue")
    lo, hi = min(true.min(), pred.min()), max(true.max(), pred.max())
    ax.plot([lo, hi], [lo, hi], "r--")
    ax.set_xlabel("True Values")
    ax.set_ylabel("Predictions")
    ax.set_title(title)


def to_target_scale(y_pred, norms):
    """Y-space prediction (ratio) -> Z-space Pareto-scale prediction, clipped at 0."""
    return np.clip(y_pred, 0, None).reshape(-1) * norms


# =============================================================================
# 6. Cross-validation: selecting k_train (proposed approach, Figure 3)
# =============================================================================
# sklearn.model_selection.RepeatedKFold.split yields (train_indices,
# test_indices) with the *standard* 80%/20% split; we deliberately swap
# their roles below to get our 20%/80% protocol.

def cross_validate_proposed(data_pareto, n_train):
    """Select k_train for the proposed (Pareto/angular) SVM and ERM models.

    Returns
    -------
    cv_risk_svm, cv_risk_erm : ndarray of shape (len(K_LIST), n_splits)
    """
    splits = list(
        RepeatedKFold(n_splits=N_FOLDS, n_repeats=N_PERMUTATIONS, random_state=RNG_SEED)
        .split(data_pareto)
    )

    # Defensive check: K_LIST must stay below the smallest training-fold
    # size, otherwise `angular_x[:k_train]` would silently return fewer rows
    # than requested for the largest k values instead of raising an error.
    min_train_fold_size = min(len(fold_20pct) for _, fold_20pct in splits)
    assert K_LIST.max() < min_train_fold_size, (
        f"K_LIST goes up to {K_LIST.max()}, but the smallest training fold "
        f"(20% of the data) only has {min_train_fold_size} rows."
    )

    cv_risk_svm = np.zeros((len(K_LIST), len(splits)))
    cv_risk_erm = np.zeros((len(K_LIST), len(splits)))

    for split_idx, (fold_80pct, fold_20pct) in enumerate(splits):
        # RepeatedKFold.split() yields (train_indices, test_indices) i.e.
        # (80%, 20%) by sklearn's convention. Section 4.1's protocol needs
        # the opposite -- 20% for training, 80% for validation, so that the
        # (large) validation set contains enough extreme points to reliably
        # estimate the asymptotic risk -- so we deliberately use sklearn's
        # "test" fold as our training set, and vice versa.
        train_idx, test_idx = fold_20pct, fold_80pct

        X_train, y_train = extract_target_station(data_pareto[train_idx], TARGET_STATION)
        X_test, y_test = extract_target_station(data_pareto[test_idx], TARGET_STATION)

        k_test = int(len(y_test) * TAU_TEST)
        angular_x_test, y_test_sorted = angular_features_sorted_by_extremeness(X_test, y_test)
        angular_x_extreme_test = angular_x_test[:k_test]
        y_extreme_test = y_test_sorted[:k_test]
        test_bunch = Bunch(data=angular_x_extreme_test, target=y_extreme_test)

        for k_idx, k_train in enumerate(K_LIST):
            model_svm = fit_svm(X_train, y_train, k_train)
            cv_risk_svm[k_idx, split_idx] = model_svm.test(test_bunch)[1][0][0]

            model_erm = fit_erm(X_train, y_train, k_train)
            y_pred_erm = model_erm.predict(angular_x_extreme_test).ravel()
            cv_risk_erm[k_idx, split_idx] = pinball_loss(y_pred_erm, y_extreme_test)

    return cv_risk_svm, cv_risk_erm


def plot_bias_variance_tradeoff(cv_risk_svm, cv_risk_erm, n_train):
    """Figure 3: mean CV risk vs k_train / n_train, for both algorithms."""
    fig, ax = plt.subplots()
    ax.plot(K_LIST / n_train, cv_risk_svm.mean(axis=1), label="SVM")
    ax.plot(K_LIST / n_train, cv_risk_erm.mean(axis=1), label="ERM")
    ax.set_xlabel(r"$k_{\mathrm{train}} / n_{\mathrm{train}}$")
    ax.set_ylabel("Mean approximated asymptotic pinball risk")
    ax.legend()
    plt.savefig("tradeoff_bias_variance_ERM_and_SVM.pdf")
    plt.show()


def select_k_train(cv_risk_svm, cv_risk_erm, n_train, label=""):
    k_train_svm = K_LIST[np.argmin(cv_risk_svm.mean(axis=1))]
    k_train_erm = K_LIST[np.argmin(cv_risk_erm.mean(axis=1))]
    frac_svm = k_train_svm / n_train
    frac_erm = k_train_erm / n_train
    print(f"Selected k_train{label}: SVM={k_train_svm} ({frac_svm:.1%}), "
          f"ERM={k_train_erm} ({frac_erm:.1%})")
    return k_train_svm, k_train_erm


# =============================================================================
# 7. Cross-validation: selecting k_train (naive baseline)
# =============================================================================
# Same protocol, but on the naively-standardized data and using the raw
# covariate X instead of its angular component (see Section 4.2, "Comparison
# with baseline approaches"), and no normalization of the target.
# Fewer permutations here (N_PERMUTATIONS_NAIVE)
# purely to keep runtime reasonable; this does not affect the k selected for
# the proposed approach above.

def cross_validate_naive(data_naive):
    """Select k_train for the naive-baseline SVM and ERM models."""
    splits_naive = list(
        RepeatedKFold(n_splits=N_FOLDS, n_repeats=N_PERMUTATIONS_NAIVE, random_state=RNG_SEED)
        .split(data_naive)
    )

    cv_risk_svm_naive = np.zeros((len(K_LIST), len(splits_naive)))
    cv_risk_erm_naive = np.zeros((len(K_LIST), len(splits_naive)))

    for split_idx, (fold_80pct, fold_20pct) in enumerate(splits_naive):
        # Same deliberate 20%/80% swap as in cross_validate_proposed above.
        train_idx, test_idx = fold_20pct, fold_80pct

        X_train = np.delete(data_naive[train_idx], TARGET_STATION, axis=1)
        Z_train = data_naive[train_idx][:, TARGET_STATION]
        X_test = np.delete(data_naive[test_idx], TARGET_STATION, axis=1)
        Z_test = data_naive[test_idx][:, TARGET_STATION]

        # Sort by decreasing infinity norm so [:k_train]/[:k_test] keeps the
        # most extreme rows, mirroring the proposed pipeline's convention.
        order_train = np.argsort(-infinity_norm(X_train))
        X_train, Z_train = X_train[order_train], Z_train[order_train]
        order_test = np.argsort(-infinity_norm(X_test))
        X_test, Z_test = X_test[order_test], Z_test[order_test]

        k_test = int(len(Z_test) * TAU_TEST)
        X_extreme_test, Z_extreme_test = X_test[:k_test], Z_test[:k_test]
        test_bunch = Bunch(data=X_extreme_test, target=Z_extreme_test)

        for k_idx, k_train in enumerate(K_LIST):
            model_svm = qtSVM(X_train[:k_train], Z_train[:k_train], weights=[0.5])
            cv_risk_svm_naive[k_idx, split_idx] = model_svm.test(test_bunch)[1][0][0]

            model_erm = QuantileRegressor(alpha=0, quantile=0.5).fit(X_train[:k_train], Z_train[:k_train])
            Z_pred_erm = model_erm.predict(X_extreme_test).ravel()
            cv_risk_erm_naive[k_idx, split_idx] = pinball_loss(Z_pred_erm, Z_extreme_test)

    return cv_risk_svm_naive, cv_risk_erm_naive


# =============================================================================
# 8. Held-out prediction: proposed approach (Figure 4)
# =============================================================================
# A single 20%/80% train/test split (independent of the cross-validation
# splits above) is used to fit the final SVM and ERM models with the
# selected k_train, and to produce the diagnostic scatter plots of Figure 4:
# predictions vs. true values, on the Pareto scale (left) and back on the
# original scale via pareto_to_normal (right).

def predict_proposed(data, data_pareto, permutation, n_train, k_test, k_train_svm, k_train_erm):
    data_train_pareto = data_pareto[permutation][:n_train]
    data_test_pareto = data_pareto[permutation][n_train:]
    data_test_original = data[permutation][n_train:]

    X_train, y_train = extract_target_station(data_train_pareto, TARGET_STATION)
    X_test, y_test = extract_target_station(data_test_pareto, TARGET_STATION)
    norm_X_test = infinity_norm(X_test)
    extreme_order = np.argsort(-norm_X_test)

    angular_x_test, y_test_sorted = angular_features_sorted_by_extremeness(X_test, y_test)
    angular_x_extreme_test = angular_x_test[:k_test]
    y_extreme_test = y_test_sorted[:k_test]
    norm_extreme_test = norm_X_test[extreme_order][:k_test]
    Z_extreme_test_original = data_test_original[:, TARGET_STATION][extreme_order][:k_test]

    model_svm = fit_svm(X_train, y_train, k_train_svm)
    model_erm = fit_erm(X_train, y_train, k_train_erm)

    y_pred_svm = model_svm.predict(angular_x_extreme_test).reshape(-1)
    Z_pred_pareto_svm = to_target_scale(y_pred_svm, norm_extreme_test)
    Z_pred_original_svm = pareto_to_normal(
        data[:, TARGET_STATION], Z_pred_pareto_svm, STANDARDIZATION_QUANTILE
    )

    y_pred_erm = model_erm.predict(angular_x_extreme_test).reshape(-1)
    Z_pred_pareto_erm = to_target_scale(y_pred_erm, norm_extreme_test)
    Z_pred_original_erm = pareto_to_normal(
        data[:, TARGET_STATION], Z_pred_pareto_erm, STANDARDIZATION_QUANTILE
    )

    return {
        "y_extreme_test": y_extreme_test,
        "y_pred_svm": y_pred_svm,
        "Z_extreme_test_original": Z_extreme_test_original,
        "Z_pred_original_svm": Z_pred_original_svm,
        "Z_pred_original_erm": Z_pred_original_erm,
    }


def plot_proposed_predictions(results):
    """Figure 4: predictions vs. true values, Pareto scale and original scale."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    scatter_true_vs_pred(results["y_extreme_test"], results["y_pred_svm"], axes[0], "Pareto scale (Y)")
    scatter_true_vs_pred(
        results["Z_extreme_test_original"], results["Z_pred_original_svm"], axes[1], "Original scale (Z)"
    )
    fig.tight_layout(w_pad=4.0, h_pad=3.0)
    plt.savefig("svm_pareto_and_normal_scale.pdf")
    plt.show()


# =============================================================================
# 9. Held-out prediction: naive baseline (Figure 5)
# =============================================================================
# Same held-out split (via the same `permutation`), but using the naive
# standardization and the raw covariate, for both SVM and ERM.

def predict_naive(data, data_naive, permutation, n_train, k_test, k_train_svm_naive, k_train_erm_naive):
    data_train_naive = data_naive[permutation][:n_train]
    data_test_naive = data_naive[permutation][n_train:]

    target_mean = data[:, TARGET_STATION].mean()
    target_std = data[:, TARGET_STATION].std()

    X_train_naive = np.delete(data_train_naive, TARGET_STATION, axis=1)
    Z_train_naive = data_train_naive[:, TARGET_STATION]
    X_test_naive = np.delete(data_test_naive, TARGET_STATION, axis=1)
    Z_test_naive = data_test_naive[:, TARGET_STATION]

    order_train_naive = np.argsort(-infinity_norm(X_train_naive))
    X_train_naive, Z_train_naive = X_train_naive[order_train_naive], Z_train_naive[order_train_naive]
    order_test_naive = np.argsort(-infinity_norm(X_test_naive))
    X_test_naive, Z_test_naive = X_test_naive[order_test_naive], Z_test_naive[order_test_naive]

    X_extreme_test_naive = X_test_naive[:k_test]
    Z_extreme_test_naive_original = Z_test_naive[:k_test] * target_std + target_mean

    model_svm_naive = qtSVM(
        X_train_naive[:k_train_svm_naive], Z_train_naive[:k_train_svm_naive], weights=[0.5]
    )
    model_erm_naive = QuantileRegressor(alpha=0, quantile=0.5).fit(
        X_train_naive[:k_train_erm_naive], Z_train_naive[:k_train_erm_naive]
    )

    Z_pred_svm_naive = model_svm_naive.predict(X_extreme_test_naive) * target_std + target_mean
    Z_pred_erm_naive = model_erm_naive.predict(X_extreme_test_naive) * target_std + target_mean

    return {
        "Z_extreme_test_naive_original": Z_extreme_test_naive_original,
        "Z_pred_svm_naive": Z_pred_svm_naive,
        "Z_pred_erm_naive": Z_pred_erm_naive,
    }


def plot_baseline_comparison(naive_results, proposed_results):
    """Figure 5: naive baseline vs. proposed approach, SVM and ERM, predictions vs. true values."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    scatter_true_vs_pred(
        naive_results["Z_extreme_test_naive_original"], naive_results["Z_pred_svm_naive"],
        axes[0, 0], "Baseline, SVM",
    )
    scatter_true_vs_pred(
        naive_results["Z_extreme_test_naive_original"], naive_results["Z_pred_erm_naive"],
        axes[0, 1], "Baseline, ERM",
    )
    scatter_true_vs_pred(
        proposed_results["Z_extreme_test_original"], proposed_results["Z_pred_original_svm"],
        axes[1, 0], "Proposed, SVM",
    )
    scatter_true_vs_pred(
        proposed_results["Z_extreme_test_original"], proposed_results["Z_pred_original_erm"],
        axes[1, 1], "Proposed, ERM",
    )
    fig.tight_layout(w_pad=4.0, h_pad=3.0)
    plt.savefig("comparison_with_baseline.pdf")
    plt.show()


# =============================================================================
# Main
# =============================================================================

def main():
    data = load_data()
    n, m = data.shape
    n_train = int(TRAIN_FRACTION * n)

    data_pareto = np.column_stack([
        normalize(data[:, j], STANDARDIZATION_QUANTILE) for j in range(m)
    ])
    data_naive = (data - data.mean(axis=0)) / data.std(axis=0)

    plot_asymptotic_dependence(data_pareto)

    cv_risk_svm, cv_risk_erm = cross_validate_proposed(data_pareto, n_train)
    plot_bias_variance_tradeoff(cv_risk_svm, cv_risk_erm, n_train)
    k_train_svm_pareto, k_train_erm_pareto = select_k_train(cv_risk_svm, cv_risk_erm, n_train)

    cv_risk_svm_naive, cv_risk_erm_naive = cross_validate_naive(data_naive)
    k_train_svm_naive, k_train_erm_naive = select_k_train(
        cv_risk_svm_naive, cv_risk_erm_naive, n_train, label=" (naive)"
    )

    permutation = np.random.default_rng(RNG_SEED).permutation(n)
    n_test = n - n_train
    k_test = int(n_test * TAU_TEST)

    proposed_results = predict_proposed(
        data, data_pareto, permutation, n_train, k_test, k_train_svm_pareto, k_train_erm_pareto
    )
    plot_proposed_predictions(proposed_results)

    naive_results = predict_naive(
        data, data_naive, permutation, n_train, k_test, k_train_svm_naive, k_train_erm_naive
    )
    plot_baseline_comparison(naive_results, proposed_results)


if __name__ == "__main__":
    main()
