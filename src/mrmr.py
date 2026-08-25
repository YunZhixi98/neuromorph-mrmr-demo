"""mRMR feature selection for the morphology tutorial.

``my_mRMR`` follows the mutual-information-difference (MID) criterion from
Peng, Long & Ding (2005). The mutual information and discretization used by
that implementation are written here explicitly; scikit-learn is not used by
the custom selector.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


MRMR_METHODS = ("my_mRMR", "mrmr_selection", "feature_engine", "pymrmr")
BINNING_METHODS = ("mean_std_3", "quantile_5")


def _validate_inputs(features: pd.DataFrame, target: pd.Series, k: int) -> None:
    if not isinstance(features, pd.DataFrame) or features.empty:
        raise ValueError("features must be a non-empty pandas DataFrame")
    if len(features) != len(target):
        raise ValueError("features and target must contain the same number of rows")
    if not 1 <= k <= features.shape[1]:
        raise ValueError(f"k must be between 1 and {features.shape[1]}")
    if features.columns.duplicated().any():
        raise ValueError("feature names must be unique")
    if features.isna().any().any() or not np.isfinite(features.to_numpy()).all():
        raise ValueError("features must contain only finite values")
    if pd.Series(target).isna().any():
        raise ValueError("target must not contain missing values")
    if pd.Series(target).nunique() < 2:
        raise ValueError("target must contain at least two classes")


def discretize_features(
    features: pd.DataFrame,
    method: str = "mean_std_3",
    *,
    alpha: float = 1.0,
    n_bins: int = 5,
) -> pd.DataFrame:
    """Discretize every continuous feature independently.

    ``mean_std_3`` reproduces the three-state preprocessing described by the
    original mRMR software documentation. For feature x with population mean
    mu and population standard deviation sigma, values below
    ``mu - alpha*sigma`` map to -1, values above ``mu + alpha*sigma`` map to
    +1, and all other values map to 0.

    ``quantile_5`` divides ranked values into ``n_bins`` approximately
    equal-frequency groups (five by default). Duplicate cut points are
    dropped, so tied or constant features can have fewer occupied bins.
    """

    if method not in BINNING_METHODS:
        raise ValueError(f"Unknown binning method {method!r}; choose {BINNING_METHODS}")
    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")

    discrete = pd.DataFrame(index=features.index)
    for name in features.columns:
        values = features[name].to_numpy(dtype=float)

        if method == "mean_std_3":
            mean = float(np.mean(values))
            # ddof=0 matches the population standard deviation over the
            # observed dataset used to define the three discrete states.
            standard_deviation = float(np.std(values, ddof=0))
            states = np.zeros(len(values), dtype=np.int8)
            if standard_deviation > 0:
                states[values < mean - alpha * standard_deviation] = -1
                states[values > mean + alpha * standard_deviation] = 1
            discrete[name] = states
        else:
            bins = pd.qcut(
                features[name], q=n_bins, labels=False, duplicates="drop"
            )
            discrete[name] = bins.fillna(0).astype(int)

    return discrete


def discrete_mutual_information(x: Sequence, y: Sequence) -> float:
    r"""Compute discrete mutual information from observed counts.

    This is the plug-in estimate used by ``my_mRMR`` and implements Eq. (1)
    of Peng, Long & Ding (2005) directly:

    ``I(X;Y) = sum p(x,y) log(p(x,y) / (p(x)p(y)))``.

    Empty joint states contribute zero and are skipped. The natural logarithm
    reports MI in nats; changing the log base multiplies every MID term by the
    same positive constant and therefore does not change the selection order.
    """

    x_values = np.asarray(x)
    y_values = np.asarray(y)
    if x_values.ndim != 1 or y_values.ndim != 1:
        raise ValueError("x and y must be one-dimensional")
    if len(x_values) != len(y_values) or len(x_values) == 0:
        raise ValueError("x and y must have the same non-zero length")

    _, x_codes = np.unique(x_values, return_inverse=True)
    _, y_codes = np.unique(y_values, return_inverse=True)
    joint_counts = np.zeros(
        (int(x_codes.max()) + 1, int(y_codes.max()) + 1), dtype=float
    )
    np.add.at(joint_counts, (x_codes, y_codes), 1.0)

    joint_probability = joint_counts / len(x_values)
    x_probability = joint_probability.sum(axis=1, keepdims=True)
    y_probability = joint_probability.sum(axis=0, keepdims=True)
    independent_probability = x_probability * y_probability
    occupied = joint_probability > 0
    return float(
        np.sum(
            joint_probability[occupied]
            * np.log(joint_probability[occupied] / independent_probability[occupied])
        )
    )


def feature_relevance(discrete: pd.DataFrame, target: pd.Series) -> pd.Series:
    """Return I(feature; target) for each already-discretized feature."""

    values = {
        name: discrete_mutual_information(discrete[name], target)
        for name in discrete.columns
    }
    return pd.Series(values, name="relevance").sort_values(ascending=False)


def my_mrmr_select(
    features: pd.DataFrame,
    target: pd.Series,
    k: int,
    *,
    binning: str = "mean_std_3",
    alpha: float = 1.0,
    n_bins: int = 5,
) -> tuple[list[str], pd.DataFrame]:
    r"""Select ``k`` features using the original incremental MID criterion.

    For selected set S_(m-1), Eq. (7) of the original paper chooses the
    candidate x_j maximizing

    ``I(x_j;c) - 1/(m-1) * sum_[x_i in S_(m-1)] I(x_j;x_i)``.

    This is maximum relevance minus the candidate's mean redundancy with the
    features already selected. At step one the redundancy term is zero. The
    returned trace exposes every term for inspection.
    """

    _validate_inputs(features, target, k)
    discrete = discretize_features(
        features, method=binning, alpha=alpha, n_bins=n_bins
    )
    relevance = feature_relevance(discrete, target)
    selected: list[str] = []
    trace_rows: list[dict[str, object]] = []

    for step in range(1, k + 1):
        candidates = [name for name in features.columns if name not in selected]
        step_rows: list[dict[str, object]] = []

        for candidate in candidates:
            pairwise_redundancy = [
                discrete_mutual_information(discrete[candidate], discrete[chosen])
                for chosen in selected
            ]
            redundancy = (
                float(np.mean(pairwise_redundancy))
                if pairwise_redundancy
                else 0.0
            )
            step_rows.append(
                {
                    "step": step,
                    "candidate": candidate,
                    "relevance": float(relevance[candidate]),
                    "redundancy": redundancy,
                    "mrmr_score": float(relevance[candidate] - redundancy),
                }
            )

        # Python's max keeps the first candidate on an exact tie, which makes
        # selection deterministic in the input-column order.
        winner = max(step_rows, key=lambda row: row["mrmr_score"])["candidate"]
        selected.append(str(winner))
        for row in step_rows:
            row["selected_next"] = row["candidate"] == winner
        trace_rows.extend(step_rows)

    return selected, pd.DataFrame(trace_rows)


# ---------------------------------------------------------------------------
# Extension: continuous k-nearest-neighbor MI estimation
# ---------------------------------------------------------------------------


def continuous_feature_relevance(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    n_neighbors: int = 3,
    random_state: int = 42,
) -> pd.Series:
    """Estimate I(feature; class) with sklearn's continuous kNN estimator.

    This function is intentionally separate from ``my_mrmr_select``. The
    main teaching path uses count-based MI after three-state
    discretization; this estimator is only used in the sensitivity extension.
    """

    from sklearn.feature_selection import mutual_info_classif

    target_codes, _ = pd.factorize(target, sort=True)
    scores = mutual_info_classif(
        features,
        target_codes,
        discrete_features=False,
        n_neighbors=n_neighbors,
        random_state=random_state,
    )
    return pd.Series(scores, index=features.columns, name="relevance").sort_values(
        ascending=False
    )


def continuous_mrmr_select(
    features: pd.DataFrame,
    target: pd.Series,
    k: int,
    *,
    n_neighbors: int = 3,
    random_state: int = 42,
) -> tuple[list[str], pd.DataFrame]:
    """Greedy MID using continuous kNN estimates for both MI terms.

    Relevance uses ``mutual_info_classif``. Redundancy treats each selected
    continuous feature as the response of ``mutual_info_regression``. This is
    an estimator-sensitivity extension, not part of the main discrete path.
    """

    from sklearn.feature_selection import mutual_info_regression

    _validate_inputs(features, target, k)
    relevance = continuous_feature_relevance(
        features,
        target,
        n_neighbors=n_neighbors,
        random_state=random_state,
    )
    selected: list[str] = []
    trace_rows: list[dict[str, object]] = []
    redundancy_cache: dict[str, dict[str, float]] = {}

    for step in range(1, k + 1):
        candidates = [name for name in features.columns if name not in selected]
        step_rows: list[dict[str, object]] = []

        # Estimate every candidate against a newly selected feature in one
        # vectorized call. Besides being clearer, this matches the way
        # Feature-engine advances sklearn's tiny-noise random state.
        for chosen in selected:
            if chosen not in redundancy_cache:
                scores = mutual_info_regression(
                    features[candidates],
                    features[chosen],
                    discrete_features=False,
                    n_neighbors=n_neighbors,
                    random_state=random_state,
                )
                redundancy_cache[chosen] = dict(zip(candidates, scores))

        for candidate in candidates:
            pairwise_redundancy = [
                float(redundancy_cache[chosen][candidate])
                for chosen in selected
            ]
            redundancy = (
                float(np.mean(pairwise_redundancy))
                if pairwise_redundancy
                else 0.0
            )
            step_rows.append(
                {
                    "step": step,
                    "candidate": candidate,
                    "relevance": float(relevance[candidate]),
                    "redundancy": redundancy,
                    "mrmr_score": float(relevance[candidate] - redundancy),
                }
            )

        winner = max(step_rows, key=lambda row: row["mrmr_score"])["candidate"]
        selected.append(str(winner))
        for row in step_rows:
            row["selected_next"] = row["candidate"] == winner
        trace_rows.extend(step_rows)

    return selected, pd.DataFrame(trace_rows)


def select_features(
    features: pd.DataFrame,
    target: pd.Series,
    k: int,
    method: str = "my_mRMR",
    *,
    binning: str = "mean_std_3",
    alpha: float = 1.0,
    n_bins: int = 5,
) -> list[str]:
    """Select features with ``my_mRMR`` or one of three external packages.

    ``my_mRMR`` and the legacy ``pymrmr`` wrapper receive the same discretized
    table. ``mrmr-selection`` and Feature-engine retain their own package
    definitions and operate on the continuous feature table.
    """

    _validate_inputs(features, target, k)

    if method == "my_mRMR":
        selected, _ = my_mrmr_select(
            features,
            target,
            k,
            binning=binning,
            alpha=alpha,
            n_bins=n_bins,
        )
        return selected

    if method == "mrmr_selection":
        from mrmr import mrmr_classif

        aligned_target = pd.Series(target.to_numpy(), index=features.index)
        return list(
            mrmr_classif(
                X=features,
                y=aligned_target,
                K=k,
                n_jobs=1,
                show_progress=False,
            )
        )

    if method == "feature_engine":
        from feature_engine.selection import MRMR

        selector = MRMR(
            method="MID",
            max_features=k,
            regression=False,
            random_state=42,
            n_jobs=1,
        )
        selector.fit(features, target)
        selected_set = set(selector.get_feature_names_out())
        # Preserve the source-column order for deterministic presentation.
        return [name for name in features.columns if name in selected_set]

    if method == "pymrmr":
        import pymrmr

        discrete = discretize_features(
            features, method=binning, alpha=alpha, n_bins=n_bins
        )
        target_codes, _ = pd.factorize(target, sort=True)
        pymrmr_input = discrete.copy()
        pymrmr_input.insert(0, "target", target_codes.astype(int))
        return list(pymrmr.mRMR(pymrmr_input, "MID", k))

    choices = ", ".join(MRMR_METHODS)
    raise ValueError(f"Unknown mRMR method {method!r}; choose one of: {choices}")
