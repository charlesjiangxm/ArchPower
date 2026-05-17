"""
Shared utilities for the C906 RuleFit power-regression pipeline.

Operates on the c906-db paired waveform / power dataset under
db/c906-db/{presim,pwr}/.  Provides loading, feature filtering, metric
computation, and rule-string parsing helpers used by c906_rulefit.py.
"""

import os
import re
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


PREFIXES = ["MMU", "cache", "csr", "exception", "interrupt"]
TARGET_COL = "/Pc(openC906)"


def _default_base_dir():
    return os.path.join(os.path.dirname(__file__), "..", "..", "db", "c906-db")


def load_c906_pair(prefix, base_dir=None, presim_subdir="presim"):
    """Load one (presim, pwr) pair, aligned by row.

    Parameters
    ----------
    prefix : str            workload prefix (e.g. "MMU", "cache").
    base_dir : str or None  defaults to repo's db/c906-db.
    presim_subdir : str     subdirectory under base_dir for presim files
                            (e.g. "presim" or "presim_large").

    Returns
    -------
    X : pd.DataFrame   (N, M)   signal-state features, float64
    y : pd.Series      (N,)      /Pc(openC906) power target, float64
    time_ps : pd.Series (N,)     per-row time stamp, int64
    """
    base_dir = base_dir or _default_base_dir()
    presim = pd.read_pickle(os.path.join(base_dir, presim_subdir, f"{prefix}_func.pkl"))
    pwr = pd.read_pickle(os.path.join(base_dir, "pwr", f"{prefix}_pwr.pkl"))

    if len(presim) != len(pwr):
        raise ValueError(
            f"{prefix}: row count mismatch presim={len(presim)} pwr={len(pwr)}"
        )
    if TARGET_COL not in pwr.columns:
        raise KeyError(f"{prefix}: pwr missing target column {TARGET_COL!r}")
    if "time_ps" not in pwr.columns:
        raise KeyError(f"{prefix}: pwr missing 'time_ps' column")

    time_ps = pwr["time_ps"].astype("int64").reset_index(drop=True)
    # Some presim layouts include a `time_ps` column themselves (e.g.
    # presim_large/). Drop it if present so X is pure signal data.
    if "time_ps" in presim.columns:
        presim = presim.drop(columns=["time_ps"])
    # Force float64. Presim columns can arrive as object dtype (no harm if
    # already numeric). Some wide-bus signals (e.g., 320-bit data_in) exceed
    # float32 range, so we standardize in float64 inside the feature selector
    # rather than downcasting here.
    X = presim.astype("float64").reset_index(drop=True)
    # Impute NaN with 0 in-place: presim_large encodes "X" (unknown/floating)
    # signal states as NaN. For switching-activity features the natural fill
    # is 0 ("no toggle observed"). Original presim has no NaN so this is a
    # no-op there.
    if X.isna().any().any():
        X = X.fillna(0.0)
    y = pwr[TARGET_COL].astype("float64").reset_index(drop=True)
    return X, y, time_ps


def load_all(base_dir=None):
    """Load and concatenate all 5 prefix pairs.

    Returns
    -------
    X : pd.DataFrame   (~31k, 14782)
    y : pd.Series      (~31k,)
    category : pd.Series  per-row prefix string (for LOCO split)
    time_ps : pd.Series   per-row timestamp
    """
    base_dir = base_dir or _default_base_dir()
    Xs, ys, ts, cats = [], [], [], []
    for p in PREFIXES:
        X, y, t = load_c906_pair(p, base_dir)
        Xs.append(X)
        ys.append(y)
        ts.append(t)
        cats.append(pd.Series([p] * len(X), dtype="category"))

    X = pd.concat(Xs, ignore_index=True)
    y = pd.concat(ys, ignore_index=True)
    time_ps = pd.concat(ts, ignore_index=True)
    category = pd.concat(cats, ignore_index=True).astype("category")
    return X, y, category, time_ps


def compute_metrics(y_true, y_pred):
    """Single-target RMSE / MAE / MAPE(%) / R^2."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    mape = float(
        np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + 1e-12)) * 100
    )
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "mape": mape, "r2": r2}


# ---------------------------------------------------------------------------
# Rule-string parsing for interaction / feature-level analysis
# ---------------------------------------------------------------------------

_COND_RE = re.compile(r"\s*(<=|>=|<|>|==|!=)\s*-?\d+\.?\d*(?:[eE][+-]?\d+)?\s*$")


def parse_rule_features(rule_str, rule_type, feature_names):
    """Return the list of feature names referenced in a RuleFit rule.

    - linear terms -> [rule_str] (the feature itself)
    - rule (conjunction) terms -> features extracted from each conjunct
    """
    if rule_type == "linear":
        return [rule_str] if rule_str in set(feature_names) else []

    conjuncts = re.split(r"\s+&\s+|\s+and\s+", rule_str)
    feat_set = set(feature_names)
    out = []
    for c in conjuncts:
        # strip trailing "<= x" / "> x" etc.
        name = _COND_RE.sub("", c).strip()
        if name in feat_set and name not in out:
            out.append(name)
    return out


# ---------------------------------------------------------------------------
# Optional non-negative refit of RuleFit's internal Lasso
# ---------------------------------------------------------------------------

def refit_nonneg_lasso(rf, X, y, random_state=42, n_alphas=100, cv=3):
    """Re-fit RuleFit's internal LassoCV with ``positive=True``.

    RuleFit fits an L1-penalised linear model on the concatenation of the
    (optionally Friedman-standardised) linear features and the rule indicators.
    The default fit allows any sign. This helper rebuilds the same design
    matrix and refits with ``positive=True`` so every linear and rule
    coefficient is constrained to be >= 0 -- useful when each rule should
    represent an additive switching-event contribution to power and negative
    coefficients are physically suspect.

    Replaces ``rf.lscv``, ``rf.coef_``, ``rf.intercept_`` in place so
    ``predict()``, ``get_rules()``, and the importance scores derived from
    them all reflect the new coefficients.
    """
    from sklearn.linear_model import LassoCV

    X_arr = np.asarray(X, dtype=np.float64)

    X_concat = np.zeros((X_arr.shape[0], 0))
    if "l" in rf.model_type:
        if getattr(rf, "lin_standardise", False):
            X_regn = rf.friedscale.scale(X_arr)
        else:
            X_regn = X_arr.copy()
        X_concat = np.concatenate((X_concat, X_regn), axis=1)
    if "r" in rf.model_type and len(rf.rule_ensemble.rules) > 0:
        X_rules = rf.rule_ensemble.transform(X_arr)
        if X_rules.size > 0 and X_rules.shape[1] > 0:
            X_concat = np.concatenate((X_concat, X_rules), axis=1)

    lscv = LassoCV(
        n_alphas=n_alphas, cv=cv, random_state=random_state, positive=True,
    )
    lscv.fit(X_concat, np.asarray(y, dtype=np.float64))
    rf.lscv = lscv
    rf.coef_ = lscv.coef_
    rf.intercept_ = lscv.intercept_
    return rf
