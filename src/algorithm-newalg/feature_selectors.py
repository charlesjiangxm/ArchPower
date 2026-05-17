"""
Pluggable feature selection for the c906 RuleFit pipeline.

Standardizes features in float64 (eliminating the wide-bus float32 overflow
that the previous `filter_features` worked around with errstate guards),
then dispatches to one of 8 selection methods chosen via `--fs_method`:

  pearson      legacy behavior -- rank by |Pearson r| with the target
  variance     drop low-variance cols, keep top-K by raw variance (not y-aware)
  univariate   sklearn SelectKBest with f_regression / mutual_info_regression
  rfe          sklearn RFE with a Ridge estimator
  from_model   sklearn SelectFromModel with LassoCV or RandomForest
  sequential   sklearn SequentialFeatureSelector (forward); impractical at large top_k
  mcp          APOLLO-style MCP-penalised sparse linear regression (Xie et al. MICRO'21)
  deep         DEEP two-step: MCP prune to V_I, then forward + swap best-subset
               (Xie et al. ICCAD'22)

All methods return a list[str] of original (non-standardized) column names,
length <= top_k, ordered best-first where the method defines an ordering.
"""

import functools
import os
import time
import warnings

import numpy as np
import pandas as pd

# Optional numba JIT for MCP coordinate descent. Coord-descent is
# algorithmically sequential (per-coordinate residual update), so we can't
# parallelize the loop -- but JIT'ing removes Python interpreter overhead,
# which is the dominant cost for our problem sizes (p ~ 500-700,
# max_iter ~ 200, ~50 lambdas). Empirically 10-50x speed-up on mcp/deep.
try:
    import numba

    @numba.njit(cache=True, fastmath=True)
    def _mcp_sweep_jit(Xz_T, w, r, indices, lam, gam_lam, rescale, positive):
        n = Xz_T.shape[1]
        max_delta = 0.0
        for k in range(indices.shape[0]):
            j = indices[k]
            wj_old = w[j]
            s = 0.0
            for i in range(n):
                s += Xz_T[j, i] * r[i]
            z = wj_old + s / n
            if positive:
                if z <= lam:
                    wj_new = 0.0
                elif z <= gam_lam:
                    wj_new = (z - lam) * rescale
                else:
                    wj_new = z
            else:
                if z >= 0.0:
                    az = z
                    sign = 1.0
                else:
                    az = -z
                    sign = -1.0
                if az <= lam:
                    wj_new = 0.0
                elif az <= gam_lam:
                    wj_new = sign * (az - lam) * rescale
                else:
                    wj_new = z
            if wj_new != wj_old:
                diff = wj_old - wj_new
                for i in range(n):
                    r[i] += Xz_T[j, i] * diff
                w[j] = wj_new
                d = -diff if diff < 0 else diff
                if d > max_delta:
                    max_delta = d
        return max_delta

    _MCP_BACKEND = "numba"
except ImportError:
    _mcp_sweep_jit = None
    _MCP_BACKEND = "python"


# ---------------------------------------------------------------------------
# Joblib-parallel mutual_info_regression. sklearn's implementation is
# single-threaded; per-feature MI estimation parallelizes well across CPU
# cores via joblib (each worker computes a chunk of columns independently).
# ---------------------------------------------------------------------------

def _parallel_mutual_info_regression(X, y, n_neighbors=3, random_state=None,
                                     n_jobs=-1, chunk_per_worker=1):
    """Parallel wrapper around `sklearn.feature_selection.mutual_info_regression`.

    Splits feature columns into chunks across joblib workers. Returns an
    array of per-feature mutual information estimates, identical in shape
    to the sklearn function's output.
    """
    from sklearn.feature_selection import mutual_info_regression
    n_features = X.shape[1]
    if n_jobs == 1 or n_features <= 1:
        return mutual_info_regression(
            X, y, n_neighbors=n_neighbors, random_state=random_state,
        )

    # Resolve number of workers.
    if n_jobs is None or n_jobs <= 0:
        n_workers = os.cpu_count() or 1
    else:
        n_workers = int(n_jobs)
    n_chunks = max(1, min(n_workers * max(1, int(chunk_per_worker)), n_features))
    chunk_size = max(1, (n_features + n_chunks - 1) // n_chunks)

    from joblib import Parallel, delayed
    chunks = [
        (start, min(start + chunk_size, n_features))
        for start in range(0, n_features, chunk_size)
    ]
    parts = Parallel(n_jobs=n_workers, prefer="processes")(
        delayed(mutual_info_regression)(
            X[:, lo:hi], y, n_neighbors=n_neighbors, random_state=random_state,
        )
        for lo, hi in chunks
    )
    return np.concatenate(parts)


# ---------------------------------------------------------------------------
# MCP coordinate descent (used by methods `mcp` and `deep` step 1)
# ---------------------------------------------------------------------------

def _mcp_prox(z, lam, gamma, positive):
    """MCP proximal operator at one coordinate.

    Assumes the column 2-norm has been absorbed (i.e. the per-feature design
    column has been standardized to ||x_j||^2 / n = 1).
    """
    if positive:
        if z <= lam:
            return 0.0
        if z <= gamma * lam:
            return (z - lam) / (1.0 - 1.0 / gamma)
        return z
    az = abs(z)
    if az <= lam:
        return 0.0
    if az <= gamma * lam:
        return (1.0 if z > 0 else -1.0) * (az - lam) / (1.0 - 1.0 / gamma)
    return z


def _mcp_sweep_python(Xz, yc, w, r, indices, lam, gam_lam, rescale, positive):
    """Pure-Python fallback when numba is unavailable. ``yc`` is unused
    (kept in the signature for parity with the JIT version's caller)."""
    n = Xz.shape[0]
    max_delta = 0.0
    for j in indices:
        wj_old = w[j]
        z = wj_old + Xz[:, j].dot(r) / n
        if positive:
            if z <= lam:
                wj_new = 0.0
            elif z <= gam_lam:
                wj_new = (z - lam) * rescale
            else:
                wj_new = z
        else:
            az = abs(z)
            if az <= lam:
                wj_new = 0.0
            elif az <= gam_lam:
                wj_new = (1.0 if z > 0 else -1.0) * (az - lam) * rescale
            else:
                wj_new = z
        if wj_new != wj_old:
            r += Xz[:, j] * (wj_old - wj_new)
            w[j] = wj_new
            d = abs(wj_new - wj_old)
            if d > max_delta:
                max_delta = d
    return max_delta


def _mcp_coordinate_descent(Xz, yc, lam, gamma, positive, max_iter, tol,
                            w_init=None, active_set_every=5,
                            Xz_T=None):
    """Solve  min_w (1/2n) ||y - Xz w||^2 + sum_j P_MCP(w_j; lam, gamma).

    Xz is assumed already z-scored (column std == 1, ddof=0). yc is centered.
    Alternates between active-set sweeps (only nonzero coordinates) and full
    sweeps so a feature can leave the active set, but new features can still
    join. Returns w in float64.

    When `numba` is installed we JIT the per-sweep inner loop and pass a
    column-major transpose of Xz so per-column dot products are contiguous.
    The transpose may be passed in via `Xz_T` to amortize across lambdas in
    a regularization path.
    """
    n, p = Xz.shape
    if w_init is None:
        w = np.zeros(p, dtype=np.float64)
        r = np.ascontiguousarray(yc, dtype=np.float64)
        r = r.copy()
    else:
        w = np.asarray(w_init, dtype=np.float64).copy()
        r = np.ascontiguousarray(yc - Xz @ w, dtype=np.float64)

    rescale = 1.0 / (1.0 - 1.0 / gamma)
    gam_lam = gamma * lam
    use_numba = _MCP_BACKEND == "numba"
    if use_numba and Xz_T is None:
        Xz_T = np.ascontiguousarray(Xz.T)
    full_indices = np.arange(p, dtype=np.int64) if use_numba else range(p)

    for sweep in range(max_iter):
        if sweep == 0 or sweep % active_set_every == 0:
            indices = full_indices
        else:
            active = np.flatnonzero(w)
            if active.size == 0:
                indices = full_indices
            else:
                indices = active.astype(np.int64) if use_numba else active.tolist()

        if use_numba:
            max_delta = _mcp_sweep_jit(
                Xz_T, w, r, indices, lam, gam_lam, rescale, bool(positive),
            )
        else:
            max_delta = _mcp_sweep_python(
                Xz, yc, w, r, indices, lam, gam_lam, rescale, positive,
            )

        if max_delta < tol:
            break

    return w


def _mcp_select_with_target(Xz, yc, target_nnz, gamma, positive,
                            max_iter, tol, lambda_path_len, verbose=False):
    """Run an MCP regularization path with warm starts, early-stopping as
    soon as the active set hits target_nnz. Returns (w, lam, info).
    """
    n, p = Xz.shape
    lam_max = float(np.max(np.abs(Xz.T @ yc)) / n)
    lam_min = lam_max * 1e-3
    lams = np.geomspace(lam_max, lam_min, num=lambda_path_len)

    # Pre-compute the transposed design matrix once for the whole path so
    # numba's per-coordinate dot products stay contiguous.
    Xz_T = np.ascontiguousarray(Xz.T) if _MCP_BACKEND == "numba" else None

    w = np.zeros(p, dtype=np.float64)
    chosen_lam = lams[-1]
    n_tried = 0
    for lam in lams:
        n_tried += 1
        w = _mcp_coordinate_descent(
            Xz, yc, lam=lam, gamma=gamma, positive=positive,
            max_iter=max_iter, tol=tol, w_init=w, Xz_T=Xz_T,
        )
        nnz = int(np.count_nonzero(w))
        if verbose:
            print(f"    mcp path[{n_tried}/{lambda_path_len}]: "
                  f"lam={lam:.4g}  nnz={nnz}")
        if nnz >= target_nnz:
            chosen_lam = float(lam)
            break
        chosen_lam = float(lam)

    info = {"lam": chosen_lam, "lam_max": lam_max,
            "n_lams_tried": n_tried, "nnz": int(np.count_nonzero(w))}
    return w, info


# ---------------------------------------------------------------------------
# DEEP step B: swap refinement on the V_I candidate set
# ---------------------------------------------------------------------------

def _deep_swap_refine(G, b, ynorm2, S_local, n_candidates, max_iter,
                      verbose=False):
    """Greedy swap refinement of best subset of fixed size.

    G  : Gram matrix Xz[:, V_I]^T Xz[:, V_I]      shape (|V_I|, |V_I|)
    b  : Xz[:, V_I]^T y                            shape (|V_I|,)
    ynorm2: y^T y (scalar)
    S_local: starting subset, indices into [0, |V_I|)
    n_candidates: |V_I|

    Per outer pass, for each j_in in S we tentatively remove it (rank-1
    downdate of (G_S)^{-1}), pick the candidate j_best with the largest
    marginal partial-correlation score, and accept the swap if the exact
    new RSS is smaller. Stops when one outer pass makes no swaps.
    """
    S_local = list(S_local)
    K = len(S_local)
    if K == 0:
        return S_local

    def _ols_rss(S):
        G_S = G[np.ix_(S, S)]
        b_S = b[S]
        try:
            w = np.linalg.solve(G_S, b_S)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(G_S, b_S, rcond=None)[0]
        return float(ynorm2 - b_S @ w), w

    current_rss, _ = _ols_rss(S_local)

    for outer in range(max_iter):
        in_S = np.zeros(n_candidates, dtype=bool)
        in_S[S_local] = True

        # Inverse of G_S; recomputed each outer pass.
        G_S = G[np.ix_(S_local, S_local)]
        try:
            G_S_inv = np.linalg.inv(G_S)
        except np.linalg.LinAlgError:
            G_S_inv = np.linalg.pinv(G_S)
        w = G_S_inv @ b[S_local]

        improved = False
        n_swaps_this_pass = 0
        for i_local in range(K):
            j_in = S_local[i_local]

            v = G_S_inv[:, i_local]
            v_ii = v[i_local]
            if v_ii <= 1e-12:
                continue
            # OLS coefficients with feature i removed: rank-1 downdate.
            w_minus_full = w - w[i_local] * v / v_ii
            w_minus = np.delete(w_minus_full, i_local)
            S_minus = S_local[:i_local] + S_local[i_local + 1:]

            cands = np.flatnonzero(~in_S)
            if cands.size == 0:
                continue
            # Marginal partial correlation of each candidate with the
            # S-minus residual (proportional, ignoring the partial-variance
            # denominator -- fast scoring).
            G_cands_S_minus = G[np.ix_(cands, S_minus)]
            partial_corr = b[cands] - G_cands_S_minus @ w_minus
            j_best_local = int(np.argmax(partial_corr ** 2))
            j_best = int(cands[j_best_local])

            # Exact RSS for the candidate swap.
            S_try = S_minus + [j_best]
            new_rss, w_try = _ols_rss(S_try)

            if new_rss < current_rss - 1e-12 * max(abs(current_rss), 1.0):
                # Accept the swap; refresh state.
                S_local = S_try
                in_S[j_in] = False
                in_S[j_best] = True
                current_rss = new_rss
                G_S = G[np.ix_(S_local, S_local)]
                try:
                    G_S_inv = np.linalg.inv(G_S)
                except np.linalg.LinAlgError:
                    G_S_inv = np.linalg.pinv(G_S)
                w = w_try
                improved = True
                n_swaps_this_pass += 1

        if verbose:
            print(f"    deep swap pass {outer + 1}: "
                  f"{n_swaps_this_pass} swaps, rss={current_rss:.6g}")
        if not improved:
            break

    return S_local


# ---------------------------------------------------------------------------
# The FeatureSelector class
# ---------------------------------------------------------------------------

class FeatureSelector:
    """Pluggable feature selector for the c906 RuleFit pipeline.

    Drops zero-variance columns, z-scores in float64 (avoiding the wide-bus
    float32 overflow), then runs the configured selection method.

    Parameters
    ----------
    method : one of FeatureSelector.METHODS
    top_k : int      target number of features
    score_func : "f_regression" | "mutual_info_regression"   (univariate)
    variance_threshold : float                                (variance)
    rfe_step : int or float                                   (rfe)
    from_model_estimator : "lasso" | "rf"                     (from_model)
    from_model_max_iter : int                                 (from_model lasso)
    sfs_tol : float                                           (sequential)
    mcp_gamma, mcp_positive, mcp_max_iter, mcp_tol,
      mcp_lambda_path_len                                     (mcp / deep)
    deep_v_i_multiplier : int                                 (deep)
    deep_max_swap_iters : int                                 (deep)
    n_jobs : int                                              (-1 = all cores)
        Parallelism for sklearn estimators (LassoCV, RandomForestRegressor,
        SequentialFeatureSelector) and for the joblib-parallel
        `mutual_info_regression` chunking. The MCP coordinate descent is
        algorithmically sequential, so n_jobs does NOT affect mcp/deep --
        those benefit instead from numba JIT (auto-enabled if numba is
        importable).
    random_state, verbose
    """

    METHODS = ("pearson", "variance", "univariate", "rfe",
               "from_model", "sequential", "mcp", "deep")

    def __init__(
        self,
        method,
        top_k=1000,
        *,
        score_func="f_regression",
        variance_threshold=0.0,
        rfe_step=0.1,
        from_model_estimator="lasso",
        from_model_max_iter=5000,
        sfs_tol=1e-4,
        mcp_gamma=5.0,
        mcp_positive=False,
        mcp_max_iter=200,
        mcp_tol=1e-5,
        mcp_lambda_path_len=50,
        deep_v_i_multiplier=3,
        deep_max_swap_iters=20,
        n_jobs=-1,
        random_state=42,
        verbose=True,
    ):
        if method not in self.METHODS:
            raise ValueError(
                f"method={method!r} not in {self.METHODS}"
            )
        self.method = method
        self.top_k = int(top_k)
        self.score_func = score_func
        self.variance_threshold = float(variance_threshold)
        self.rfe_step = rfe_step
        self.from_model_estimator = from_model_estimator
        self.from_model_max_iter = int(from_model_max_iter)
        self.sfs_tol = float(sfs_tol)
        self.mcp_gamma = float(mcp_gamma)
        self.mcp_positive = bool(mcp_positive)
        self.mcp_max_iter = int(mcp_max_iter)
        self.mcp_tol = float(mcp_tol)
        self.mcp_lambda_path_len = int(mcp_lambda_path_len)
        self.deep_v_i_multiplier = int(deep_v_i_multiplier)
        self.deep_max_swap_iters = int(deep_max_swap_iters)
        self.n_jobs = int(n_jobs)
        self.random_state = int(random_state)
        self.verbose = bool(verbose)

        self.nonconst_cols_ = None
        self.feature_means_ = None
        self.feature_stds_ = None
        self.selected_cols_ = None
        self.scores_ = None

    @classmethod
    def from_args(cls, args):
        """Build from a c906_rulefit.py argparse Namespace."""
        return cls(
            method=args.fs_method,
            top_k=args.top_k,
            score_func=getattr(args, "fs_score_func", "f_regression"),
            variance_threshold=getattr(args, "fs_variance_threshold", 0.0),
            rfe_step=getattr(args, "fs_rfe_step", 0.1),
            from_model_estimator=getattr(args, "fs_from_model_estimator", "lasso"),
            from_model_max_iter=getattr(args, "fs_from_model_max_iter", 5000),
            sfs_tol=getattr(args, "fs_sfs_tol", 1e-4),
            mcp_gamma=getattr(args, "fs_mcp_gamma", 5.0),
            mcp_positive=getattr(args, "fs_mcp_positive", False),
            mcp_max_iter=getattr(args, "fs_mcp_max_iter", 200),
            mcp_tol=getattr(args, "fs_mcp_tol", 1e-5),
            mcp_lambda_path_len=getattr(args, "fs_mcp_lambda_path_len", 50),
            deep_v_i_multiplier=getattr(args, "fs_deep_v_i_multiplier", 3),
            deep_max_swap_iters=getattr(args, "fs_deep_max_swap_iters", 20),
            n_jobs=getattr(args, "fs_n_jobs", -1),
            random_state=getattr(args, "seed", 42),
        )

    # ----- main entry point -------------------------------------------------

    def fit_select(self, X_train, y_train):
        """Run the configured method. Returns selected column names."""
        t0 = time.time()
        if not isinstance(X_train, pd.DataFrame):
            raise TypeError("X_train must be a pandas DataFrame")
        y_arr = np.asarray(y_train, dtype=np.float64)

        # 1. Drop zero-variance columns on the raw data. Use pandas std,
        # which skips NaN -- an all-NaN column gets std=NaN, which fails > 0
        # and is dropped here. Partially-NaN columns survive and are imputed
        # in step 1b.
        stds_raw = X_train.std(axis=0, ddof=0).to_numpy()
        nonconst_mask = (stds_raw > 0) & np.isfinite(stds_raw)
        if not nonconst_mask.any():
            raise RuntimeError(
                "All training feature columns are constant; nothing to select."
            )
        self.nonconst_cols_ = X_train.columns[nonconst_mask]
        X_nc = X_train[self.nonconst_cols_].to_numpy(dtype=np.float64, copy=True)

        # 1a. Impute NaN with 0. Some presim layouts (presim_large/) contain
        # NaN entries representing "X" (unknown / floating) signal states.
        # For switching-activity features, "no toggle observed" = 0 is the
        # natural fill.
        n_nan = int(np.isnan(X_nc).sum())
        if n_nan:
            X_nc = np.nan_to_num(X_nc, nan=0.0, copy=False)
            if self.verbose:
                print(f"  [FeatureSelector] imputed {n_nan} NaN cells to 0 "
                      f"({100.0 * n_nan / X_nc.size:.2f}% of matrix).")

        # 1b. Drop columns whose raw values exceed float32 dynamic range:
        # downstream RuleFit's internal GradientBoostingRegressor casts to
        # float32 and would turn those values into +inf. The legacy
        # filter_features hit this implicitly via NaN correlations on
        # overflowing float32 cov computations; with float64 standardization
        # we have to filter explicitly.
        float32_max = float(np.finfo(np.float32).max)
        abs_max = np.max(np.abs(X_nc), axis=0)
        fits_f32 = abs_max < float32_max
        n_huge = int((~fits_f32).sum())
        if n_huge:
            self.nonconst_cols_ = self.nonconst_cols_[fits_f32]
            X_nc = X_nc[:, fits_f32]
            if self.verbose:
                print(f"  [FeatureSelector] dropped {n_huge} wide-bus columns "
                      f"exceeding float32 dynamic range (would overflow inside "
                      f"RuleFit's GBRT).")
        if X_nc.shape[1] == 0:
            raise RuntimeError(
                "No feature columns survived zero-variance + float32-range filter."
            )
        p_nc = X_nc.shape[1]

        # 2. Standardize in float64 (fixes the wide-bus overflow root cause).
        self.feature_means_ = X_nc.mean(axis=0)
        self.feature_stds_ = X_nc.std(axis=0, ddof=0)
        # Belt-and-braces: nothing should be zero here after step 1, but the
        # division would NaN if it were.
        self.feature_stds_[self.feature_stds_ == 0] = 1.0
        Xz = (X_nc - self.feature_means_) / self.feature_stds_

        y_mean = float(y_arr.mean())
        y_std = float(y_arr.std(ddof=0)) + 1e-12
        yc = y_arr - y_mean
        yz = yc / y_std

        if self.verbose:
            print(f"  [FeatureSelector method={self.method}] "
                  f"n={X_nc.shape[0]}, p={p_nc} after zero-var drop "
                  f"({(~nonconst_mask).sum()} columns dropped)  "
                  f"[n_jobs={self.n_jobs}, mcp_backend={_MCP_BACKEND}]")

        # 3. Dispatch.
        dispatch = {
            "pearson":    self._select_pearson,
            "variance":   self._select_variance,
            "univariate": self._select_univariate,
            "rfe":        self._select_rfe,
            "from_model": self._select_from_model,
            "sequential": self._select_sequential,
            "mcp":        self._select_mcp,
            "deep":       self._select_deep,
        }
        local_idx = dispatch[self.method](Xz, X_nc, yc, yz, y_arr)

        # local_idx is a list of column indices into self.nonconst_cols_,
        # already ranked best-first, capped to top_k by the method.
        local_idx = list(local_idx)
        if len(local_idx) > self.top_k:
            local_idx = local_idx[:self.top_k]
        self.selected_cols_ = [str(self.nonconst_cols_[i]) for i in local_idx]
        if self.verbose:
            print(f"  [FeatureSelector] selected {len(self.selected_cols_)} "
                  f"features in {time.time() - t0:.1f}s")
        return self.selected_cols_

    # ----- per-method implementations --------------------------------------

    def _select_pearson(self, Xz, X_nc, yc, yz, y_arr):
        # Xz has zero mean and unit std, so corr = (1/n) sum_i Xz_ij * yz_i.
        n = Xz.shape[0]
        corr = (Xz.T @ yz) / n
        abs_corr = np.abs(corr)
        self.scores_ = pd.Series(abs_corr, index=self.nonconst_cols_)
        k = min(self.top_k, abs_corr.shape[0])
        top_idx = np.argpartition(-abs_corr, k - 1)[:k]
        top_idx = top_idx[np.argsort(-abs_corr[top_idx])]
        return top_idx.tolist()

    def _select_variance(self, Xz, X_nc, yc, yz, y_arr):
        # On the RAW (non-standardized) features. y not used.
        vars_raw = X_nc.var(axis=0, ddof=0)
        survivor_mask = vars_raw > self.variance_threshold
        survivor_idx = np.flatnonzero(survivor_mask)
        if survivor_idx.size == 0:
            raise RuntimeError(
                f"No features survived variance_threshold="
                f"{self.variance_threshold}"
            )
        order = np.argsort(-vars_raw[survivor_idx])
        ranked = survivor_idx[order]
        k = min(self.top_k, ranked.size)
        self.scores_ = pd.Series(vars_raw, index=self.nonconst_cols_)
        return ranked[:k].tolist()

    def _select_univariate(self, Xz, X_nc, yc, yz, y_arr):
        from sklearn.feature_selection import (
            SelectKBest, f_regression, mutual_info_regression,
        )
        if self.score_func == "f_regression":
            score_func = f_regression
        elif self.score_func == "mutual_info_regression":
            # sklearn's mutual_info_regression is single-threaded; split the
            # feature columns into chunks and run them in parallel via joblib.
            score_func = functools.partial(
                _parallel_mutual_info_regression,
                n_neighbors=3,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            )
        else:
            raise ValueError(f"Unknown score_func={self.score_func!r}")
        k = min(self.top_k, Xz.shape[1])
        kbest = SelectKBest(score_func=score_func, k=k).fit(Xz, y_arr)
        scores = np.nan_to_num(kbest.scores_, nan=0.0, posinf=0.0, neginf=0.0)
        self.scores_ = pd.Series(scores, index=self.nonconst_cols_)
        # SelectKBest doesn't expose a ranking among the kept; sort by score.
        support = np.flatnonzero(kbest.get_support())
        order = np.argsort(-scores[support])
        return support[order].tolist()

    def _select_rfe(self, Xz, X_nc, yc, yz, y_arr):
        from sklearn.feature_selection import RFE
        from sklearn.linear_model import Ridge
        k = min(self.top_k, Xz.shape[1])
        estimator = Ridge(alpha=1.0)
        rfe = RFE(
            estimator=estimator,
            n_features_to_select=k,
            step=self.rfe_step,
        ).fit(Xz, y_arr)
        support = np.flatnonzero(rfe.support_)
        # Rank kept features by |coef| of the final estimator.
        coefs = np.abs(rfe.estimator_.coef_)
        order = np.argsort(-coefs)
        self.scores_ = pd.Series(
            np.where(rfe.support_, 1.0 / np.maximum(rfe.ranking_, 1), 0.0),
            index=self.nonconst_cols_,
        )
        return support[order].tolist()

    def _select_from_model(self, Xz, X_nc, yc, yz, y_arr):
        from sklearn.feature_selection import SelectFromModel
        k = min(self.top_k, Xz.shape[1])
        if self.from_model_estimator == "lasso":
            from sklearn.linear_model import LassoCV
            est = LassoCV(
                cv=3, n_alphas=20,
                positive=self.mcp_positive,
                max_iter=self.from_model_max_iter,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                est.fit(Xz, y_arr)
            scores = np.abs(est.coef_)
        elif self.from_model_estimator == "rf":
            from sklearn.ensemble import RandomForestRegressor
            est = RandomForestRegressor(
                n_estimators=200,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
            )
            est.fit(Xz, y_arr)
            scores = est.feature_importances_
        else:
            raise ValueError(
                f"Unknown from_model_estimator={self.from_model_estimator!r}"
            )
        self.scores_ = pd.Series(scores, index=self.nonconst_cols_)
        # Keep up to k features ranked by score; drop strict zeros for lasso.
        nz = scores > 0
        candidate_idx = np.flatnonzero(nz)
        if candidate_idx.size < k:
            # fall back to scoring all features
            candidate_idx = np.arange(scores.shape[0])
        order = np.argsort(-scores[candidate_idx])
        return candidate_idx[order][:k].tolist()

    def _select_sequential(self, Xz, X_nc, yc, yz, y_arr):
        from sklearn.feature_selection import SequentialFeatureSelector
        from sklearn.linear_model import Ridge
        k = min(self.top_k, Xz.shape[1])
        if self.verbose:
            print(
                "  [FeatureSelector] WARNING: sequential is O(top_k * p * CV_fit) "
                "-- expect hours at top_k=1000 and p>>K. Recommend small top_k."
            )
        sfs = SequentialFeatureSelector(
            estimator=Ridge(alpha=1.0),
            n_features_to_select=k,
            direction="forward",
            scoring="r2",
            cv=3,
            n_jobs=self.n_jobs,
            tol=self.sfs_tol,
        ).fit(Xz, y_arr)
        # SFS does not expose addition order. Best we can do: support_ +
        # final |Ridge coef| ranking.
        support = np.flatnonzero(sfs.get_support())
        from sklearn.linear_model import Ridge as _Ridge
        ridge = _Ridge(alpha=1.0).fit(Xz[:, support], y_arr)
        coefs = np.abs(ridge.coef_)
        scores = np.zeros(Xz.shape[1])
        scores[support] = coefs
        self.scores_ = pd.Series(scores, index=self.nonconst_cols_)
        order = np.argsort(-coefs)
        return support[order].tolist()

    def _select_mcp(self, Xz, X_nc, yc, yz, y_arr):
        target = min(self.top_k, Xz.shape[1])
        w, info = _mcp_select_with_target(
            Xz, yc,
            target_nnz=target,
            gamma=self.mcp_gamma,
            positive=self.mcp_positive,
            max_iter=self.mcp_max_iter,
            tol=self.mcp_tol,
            lambda_path_len=self.mcp_lambda_path_len,
            verbose=self.verbose,
        )
        self.scores_ = pd.Series(np.abs(w), index=self.nonconst_cols_)
        nz = np.flatnonzero(w)
        if nz.size == 0:
            raise RuntimeError(
                "MCP returned all-zero weights; try a smaller "
                "--fs_mcp_lambda_path_len or check standardization."
            )
        order = np.argsort(-np.abs(w[nz]))
        ranked = nz[order]
        if self.verbose:
            print(f"  [FeatureSelector] mcp: chose lam={info['lam']:.4g} "
                  f"(of lam_max={info['lam_max']:.4g}), "
                  f"nnz={info['nnz']}, tried {info['n_lams_tried']} "
                  f"of {self.mcp_lambda_path_len} lambdas")
        return ranked[:target].tolist()

    def _select_deep(self, Xz, X_nc, yc, yz, y_arr):
        K = min(self.top_k, Xz.shape[1])
        # ---- Step 1: MCP-pruned V_I -----------------------------------
        target = max(
            self.deep_v_i_multiplier * K,
            min(30 * K, Xz.shape[1]),
        )
        target = min(target, Xz.shape[1])
        if self.verbose:
            print(f"  [FeatureSelector] deep step 1: MCP pruning toward "
                  f"|V_I|~{target}")
        w_mcp, info = _mcp_select_with_target(
            Xz, yc,
            target_nnz=target,
            gamma=self.mcp_gamma,
            positive=self.mcp_positive,
            max_iter=self.mcp_max_iter,
            tol=self.mcp_tol,
            lambda_path_len=self.mcp_lambda_path_len,
            verbose=False,
        )
        v_i_idx = np.flatnonzero(w_mcp)
        if v_i_idx.size == 0:
            raise RuntimeError(
                "DEEP step 1 (MCP) returned an empty V_I; check standardization."
            )
        if v_i_idx.size < K:
            warnings.warn(
                f"DEEP step 1 produced |V_I|={v_i_idx.size} < top_k={K}; "
                f"skipping swap refinement and returning the MCP set."
            )
            order = np.argsort(-np.abs(w_mcp[v_i_idx]))
            return v_i_idx[order][:K].tolist()

        # Order V_I by descending |w_mcp| for stable seed selection.
        v_i_idx = v_i_idx[np.argsort(-np.abs(w_mcp[v_i_idx]))]
        if self.verbose:
            print(f"  [FeatureSelector] deep step 1: |V_I|={v_i_idx.size} "
                  f"(lam={info['lam']:.4g})")

        # ---- Step 2 prep: Gram on V_I ----------------------------------
        Xz_VI = Xz[:, v_i_idx]
        # Use 1/n scaling so b and G are on the same scale as the columns
        # being unit-variance.
        n = Xz_VI.shape[0]
        G = (Xz_VI.T @ Xz_VI) / n
        b = (Xz_VI.T @ yc) / n
        ynorm2 = float(yc @ yc) / n

        # ---- Step 2a: greedy forward via sklearn OMP -------------------
        from sklearn.linear_model import OrthogonalMatchingPursuit
        if self.verbose:
            print(f"  [FeatureSelector] deep step 2a: OMP forward to K={K}")
        omp = OrthogonalMatchingPursuit(
            n_nonzero_coefs=K, fit_intercept=False,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            omp.fit(Xz_VI, yc)
        S_local = np.flatnonzero(omp.coef_).tolist()
        if len(S_local) < K and self.verbose:
            print(f"    OMP returned {len(S_local)} non-zero coefs (< K={K})")

        # ---- Step 2b: swap refinement ----------------------------------
        if self.deep_max_swap_iters > 0 and len(S_local) > 1:
            if self.verbose:
                print(f"  [FeatureSelector] deep step 2b: swap refinement "
                      f"(<= {self.deep_max_swap_iters} passes)")
            S_local = _deep_swap_refine(
                G, b, ynorm2, S_local,
                n_candidates=v_i_idx.size,
                max_iter=self.deep_max_swap_iters,
                verbose=self.verbose,
            )

        # Final OLS to rank by |coef|.
        G_S = G[np.ix_(S_local, S_local)]
        b_S = b[S_local]
        try:
            w_final = np.linalg.solve(G_S, b_S)
        except np.linalg.LinAlgError:
            w_final = np.linalg.lstsq(G_S, b_S, rcond=None)[0]
        order = np.argsort(-np.abs(w_final))
        S_local_sorted = [S_local[i] for i in order]

        # Map local V_I indices back to indices into nonconst_cols_.
        scores_vec = np.zeros(Xz.shape[1])
        for i_local, j_local in enumerate(S_local):
            scores_vec[v_i_idx[j_local]] = abs(w_final[i_local])
        self.scores_ = pd.Series(scores_vec, index=self.nonconst_cols_)
        return [int(v_i_idx[j]) for j in S_local_sorted]
