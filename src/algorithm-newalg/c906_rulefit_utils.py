"""
Shared utilities for the C906 RuleFit power-regression pipeline.

Operates on the c906-db paired waveform / power dataset under
db/c906-db/{<presim-subdir>,pwr}/.  Provides loading, feature filtering,
metric computation, and rule-string parsing helpers used by c906_rulefit.py.
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


def available_presim_subdirs(base_dir=None):
    """Return presim-like child directory names under db/c906-db.

    The returned list is used for helpful error messages only.  A directory is
    considered presim-like when it contains at least one ``*_func.pkl`` file.
    If no such directories are found, fall back to all non-private child
    directories except ``pwr``.
    """
    base_dir = os.path.abspath(base_dir or _default_base_dir())
    try:
        names = sorted(
            name
            for name in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, name))
        )
    except OSError:
        return []

    presim_like = [
        name
        for name in names
        if name != "pwr"
        and not name.startswith("__")
        and any(
            os.path.isfile(os.path.join(base_dir, name, f"{p}_func.pkl"))
            for p in PREFIXES
        )
    ]
    if presim_like:
        return presim_like
    return [name for name in names if name != "pwr" and not name.startswith("__")]


def _format_available_presim_subdirs(base_dir=None):
    names = available_presim_subdirs(base_dir)
    return ", ".join(names) if names else "<none found>"


def validate_presim_subdir(presim_subdir="presim", base_dir=None):
    """Validate and normalize a presim directory argument.

    ``presim_subdir`` may be either:

    * a direct child folder name under ``db/c906-db`` (for example
      ``presim_no_addr_data``), or
    * an absolute path whose parent is ``db/c906-db``.

    The normalized child folder name is returned.  Path traversal and nested
    paths are rejected so the flag cannot accidentally point outside the
    c906-db directory.
    """
    base_dir = os.path.abspath(base_dir or _default_base_dir())
    base_real = os.path.realpath(base_dir)

    if presim_subdir is None or presim_subdir == "":
        raise ValueError(
            "presim_subdir must be a folder name under "
            f"{base_real}; available: {_format_available_presim_subdirs(base_real)}"
        )

    presim_arg = os.fspath(presim_subdir)
    if os.path.isabs(presim_arg):
        presim_real = os.path.realpath(presim_arg)
        name = os.path.basename(presim_real)
        if os.path.dirname(presim_real) != base_real:
            raise ValueError(
                "presim_subdir absolute path must point to a direct child of "
                f"{base_real}; got {presim_arg!r}"
            )
    else:
        normalized = os.path.normpath(presim_arg)
        if (
            normalized in ("", ".", "..")
            or os.path.isabs(normalized)
            or os.path.dirname(normalized)
        ):
            raise ValueError(
                "presim_subdir must be a single folder name under "
                f"{base_real}; got {presim_arg!r}"
            )
        name = presim_arg
        presim_real = os.path.realpath(os.path.join(base_real, name))
        if os.path.dirname(presim_real) != base_real:
            raise ValueError(
                "presim_subdir must stay under "
                f"{base_real}; got {presim_arg!r}"
            )

    if not os.path.isdir(presim_real):
        raise ValueError(
            f"presim_subdir {presim_arg!r} does not exist under {base_real}. "
            f"Available presim folders: {_format_available_presim_subdirs(base_real)}"
        )

    return name


def load_c906_pair(prefix, base_dir=None, presim_subdir="presim"):
    """Load one (presim, pwr) pair, aligned by row.

    Parameters
    ----------
    prefix : str            workload prefix (e.g. "MMU", "cache").
    base_dir : str or None  defaults to repo's db/c906-db.
    presim_subdir : str     direct subdirectory under base_dir for presim
                            files (e.g. "presim", "presim_large", or
                            "presim_no_addr_data").  Absolute paths pointing
                            to a direct child of base_dir are also accepted.

    Returns
    -------
    X : pd.DataFrame   (N, M)   signal-state features, float64
    y : pd.Series      (N,)      /Pc(openC906) power target, float64
    time_ps : pd.Series (N,)     per-row time stamp, int64
    """
    base_dir = os.path.abspath(base_dir or _default_base_dir())
    presim_subdir = validate_presim_subdir(presim_subdir, base_dir)
    presim_path = os.path.join(base_dir, presim_subdir, f"{prefix}_func.pkl")
    pwr_path = os.path.join(base_dir, "pwr", f"{prefix}_pwr.pkl")
    if not os.path.isfile(presim_path):
        raise FileNotFoundError(
            f"{prefix}: missing presim file {presim_path!r} "
            f"for presim_subdir={presim_subdir!r}"
        )
    if not os.path.isfile(pwr_path):
        raise FileNotFoundError(f"{prefix}: missing pwr file {pwr_path!r}")

    presim = pd.read_pickle(presim_path)
    pwr = pd.read_pickle(pwr_path)

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


def load_all(base_dir=None, presim_subdir="presim"):
    """Load and concatenate all 5 prefix pairs.

    Parameters
    ----------
    base_dir : str or None  defaults to repo's db/c906-db.
    presim_subdir : str     direct presim child folder under base_dir.

    Returns
    -------
    X : pd.DataFrame   (~31k, 14782)
    y : pd.Series      (~31k,)
    category : pd.Series  per-row prefix string (for LOCO split)
    time_ps : pd.Series   per-row timestamp
    """
    base_dir = os.path.abspath(base_dir or _default_base_dir())
    presim_subdir = validate_presim_subdir(presim_subdir, base_dir)
    Xs, ys, ts, cats = [], [], [], []
    for p in PREFIXES:
        X, y, t = load_c906_pair(p, base_dir, presim_subdir)
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
