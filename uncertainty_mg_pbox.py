import numpy as np
import matplotlib.pyplot as plt

# Optional (used only for gradient-based p-box via Normal CDF)
try:
    from scipy.stats import norm
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


# -----------------------------
# Utilities
# -----------------------------
def _as_2d_samples(y_scenarios):
    """
    Normalize input to a list of 1D numpy arrays.
    Accepts:
      - list/tuple of arrays
      - dict of {scenario_name: array}
      - 2D numpy array shaped (K, N) (each row is a scenario)
    """
    if isinstance(y_scenarios, dict):
        ys = [np.asarray(v, dtype=float).ravel() for v in y_scenarios.values()]
        labels = list(y_scenarios.keys())
        return ys, labels
    if isinstance(y_scenarios, (list, tuple)):
        ys = [np.asarray(v, dtype=float).ravel() for v in y_scenarios]
        return ys, None
    arr = np.asarray(y_scenarios, dtype=float)
    if arr.ndim == 2:
        ys = [arr[k, :].ravel() for k in range(arr.shape[0])]
        return ys, None
    raise ValueError("y_scenarios must be a dict, list/tuple of arrays, or a 2D array (K, N).")


def _ecdf_on_grid(samples_1d, y_grid):
    """
    Empirical CDF evaluated on y_grid for one scenario.
    """
    s = np.sort(np.asarray(samples_1d, dtype=float).ravel())
    # F(y) = P(S <= y). Use searchsorted for fast ECDF.
    return np.searchsorted(s, y_grid, side="right") / s.size


def _default_grid_from_samples(y_scenarios, num=600, pad=0.05):
    """
    Build a y-grid spanning all scenario samples with a small padding.
    """
    ys, _ = _as_2d_samples(y_scenarios)
    allv = np.concatenate([v[np.isfinite(v)] for v in ys if v.size > 0])
    if allv.size == 0:
        raise ValueError("No finite samples provided.")
    lo, hi = float(np.min(allv)), float(np.max(allv))
    if lo == hi:
        lo -= 1.0
        hi += 1.0
    span = hi - lo
    lo -= pad * span
    hi += pad * span
    return np.linspace(lo, hi, num=num)


# -----------------------------
# 1) Monte Carlo p-box plotting
# -----------------------------
def compute_pbox_from_mc(y_scenarios, y_grid=None, grid_num=600):
    """
    Construct a p-box from Monte Carlo outputs across epistemic scenarios.

    Parameters
    ----------
    y_scenarios : dict or list of arrays or 2D array (K, N)
        K scenarios, each provides N Monte Carlo samples of the output.
    y_grid : 1D array, optional
        Points where CDFs are evaluated. If None, constructed automatically.
    grid_num : int
        Number of grid points if y_grid is None.

    Returns
    -------
    y_grid : (M,) array
    F_low  : (M,) array   pointwise min CDF
    F_high : (M,) array   pointwise max CDF
    """
    ys, _ = _as_2d_samples(y_scenarios)
    if y_grid is None:
        y_grid = _default_grid_from_samples(ys, num=grid_num)
    y_grid = np.asarray(y_grid, dtype=float).ravel()

    F_all = []
    for v in ys:
        if v.size == 0:
            continue
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        F_all.append(_ecdf_on_grid(v, y_grid))

    if len(F_all) == 0:
        raise ValueError("No valid (finite) samples across scenarios.")

    F_all = np.stack(F_all, axis=0)  # (K_eff, M)
    F_low = np.min(F_all, axis=0)
    F_high = np.max(F_all, axis=0)
    return y_grid, F_low, F_high


def plot_pbox_mc(y_scenarios, y_grid=None, grid_num=600, ax=None,
                 show_band=True, show_bounds=True, alpha=0.25, linewidth=2.0,
                 label_prefix="MC p-box"):
    """
    Plot p-box computed from Monte Carlo scenario outputs.
    """
    y_grid, F_low, F_high = compute_pbox_from_mc(y_scenarios, y_grid=y_grid, grid_num=grid_num)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))

    if show_band:
        ax.fill_between(y_grid, F_low, F_high, alpha=alpha, label=f"{label_prefix} band", color="r")
    if show_bounds:
        ax.plot(y_grid, F_low, linewidth=linewidth, linestyle='--', color="r", label=f"{label_prefix} lower CDF")
        ax.plot(y_grid, F_high, linewidth=linewidth, color="r",label=f"{label_prefix} upper CDF")

    ax.set_xlabel("y")
    ax.set_ylabel("CDF")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend()
    return ax


# -----------------------------------------
# 2) Gradient-based (delta-method) p-box
# -----------------------------------------
def compute_pbox_from_gradient_delta(
    mu_scenarios,
    sigma_scenarios,
    f_g,
    y_grid=None,
    grid_num=600,
    y_grid_pad=0.10,
    min_std=1e-12,
):
    """
    Gradient-based p-box using 1st-order delta method with a Normal approximation.

    Supports either:
      - 1D parameter: theta ~ N(mu, sigma^2)
          Var(Y) ≈ (f'(mu))^2 * sigma^2
      - dD parameters:
          * Diagonal (no covariance): theta ~ N(mu, diag(sigma^2))
              Var(Y) ≈ sum_i (df/dtheta_i)^2 * sigma_i^2
          * Full covariance: theta ~ N(mu, Sigma)
              Var(Y) ≈ g^T Sigma g

    The p-box is the envelope over epistemic scenarios.

    Parameters
    ----------
    mu_scenarios : array-like
        Shape (K,) for 1D, or (K, d) for dD.
    sigma_scenarios : array-like
        For 1D: shape (K,) stddev.
        For dD diagonal: shape (K, d) stddev per dimension.
        For dD covariance: shape (K, d, d) covariance matrices.
    f_g : callable
        f_g(theta) -> (y_scalar, grad)
        For dD, grad must be length-d.

    Returns
    -------
    y_grid : (M,) array
    F_low  : (M,) array
    F_high : (M,) array
    """
    if not _HAS_SCIPY:
        raise ImportError("scipy is required for gradient-based p-box (Normal CDF). Install with: pip install scipy")

    mu = np.asarray(mu_scenarios, dtype=float)
    sig = np.asarray(sigma_scenarios, dtype=float)

    # --------------
    # 1D parameter
    # --------------
    if mu.ndim == 1:
        mu = mu.ravel()
        sig = sig.ravel()
        if mu.shape != sig.shape:
            raise ValueError("mu_scenarios and sigma_scenarios must have the same shape for 1D.")

        m_list = []
        g_list = []
        for mui in mu:
            mk, gk = f_g(mui)
            m_list.append(float(mk))
            g_list.append(float(gk))
        m = np.asarray(m_list, dtype=float)
        g = np.asarray(g_list, dtype=float)

        s = np.maximum(np.abs(g) * sig, min_std)

    # -------------------
    # Multi-D parameters
    # -------------------
    elif mu.ndim == 2:
        K, d = mu.shape

        # Evaluate f and grad at each scenario mean
        m_list = []
        g_list = []
        for k in range(K):
            mk, gk = f_g(mu[k])
            m_list.append(float(mk))
            gk = np.asarray(gk, dtype=float).ravel()
            if gk.size != d:
                raise ValueError(f"f_g must return a length-{d} gradient for multi-D. Got {gk.size}.")
            g_list.append(gk)
        m = np.asarray(m_list, dtype=float)               # (K,)
        g = np.stack(g_list, axis=0)                      # (K, d)

        if sig.ndim == 2:
            # Diagonal stddev per-dim
            if sig.shape != (K, d):
                raise ValueError("For diagonal multi-D, sigma_scenarios must have shape (K, d).")
            var = np.sum((g * sig) ** 2, axis=1)           # (K,)
        elif sig.ndim == 3:
            # Full covariance per scenario
            if sig.shape != (K, d, d):
                raise ValueError("For covariance multi-D, sigma_scenarios must have shape (K, d, d).")
            # var_k = g_k^T Sigma_k g_k
            var = np.einsum('ki,kij,kj->k', g, sig, g)
        else:
            raise ValueError("For multi-D, sigma_scenarios must be (K, d) (diag stddev) or (K, d, d) (covariance).")

        # Convert to stddev of Y
        s = np.sqrt(np.maximum(var, min_std * min_std))

    else:
        raise ValueError("mu_scenarios must be shape (K,) or (K, d).")

    # -------------------
    # Auto grid from m ± 4*s
    # -------------------
    if y_grid is None:
        lo = float(np.min(m - 4.0 * s))
        hi = float(np.max(m + 4.0 * s))
        if lo == hi:
            lo -= 1.0
            hi += 1.0
        span = hi - lo
        lo -= y_grid_pad * span
        hi += y_grid_pad * span
        y_grid = np.linspace(lo, hi, num=grid_num)
    y_grid = np.asarray(y_grid, dtype=float).ravel()

    # Compute scenario CDFs and envelope them
    F_all = []
    for mk, sk in zip(m, s):
        F_all.append(norm.cdf((y_grid - mk) / sk))
    F_all = np.stack(F_all, axis=0)  # (K, M)

    F_low = np.min(F_all, axis=0)
    F_high = np.max(F_all, axis=0)
    return y_grid, F_low, F_high


def plot_pbox_gradient_delta(
    mu_scenarios,
    sigma_scenarios,
    f_g,
    y_grid=None,
    grid_num=600,
    ax=None,
    show_band=True,
    show_bounds=True,
    alpha=0.25,
    linewidth=2.0,
    label_prefix="Grad-Δ p-box",
):
    """
    Plot gradient-based (delta-method) p-box (Normal approximation).
    """
    y_grid, F_low, F_high = compute_pbox_from_gradient_delta(
        mu_scenarios, sigma_scenarios, f_g, y_grid=y_grid, grid_num=grid_num
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))

    if show_band:
        ax.fill_between(y_grid, F_low, F_high, alpha=alpha, label=f"{label_prefix} band",  color="b")
    if show_bounds:
        ax.plot(y_grid, F_low, linewidth=linewidth, linestyle="--", color="b", label=f"{label_prefix} lower CDF")
        ax.plot(y_grid, F_high, linewidth=linewidth,  color="b", label=f"{label_prefix} upper CDF")

    ax.set_xlabel("y")
    ax.set_ylabel("CDF")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend()
    return ax

from tests import uncertainty_f

# -----------------------------
# Example usage (optional)
# -----------------------------
if __name__ == "__main__":
    # Multi-group (3 energy groups) p-box sensitivity analysis.
    # Uncertain parameters: inner-region sig_t for each group (3 params).
    # Output y: middle channel value (group index 1) returned by tests.uncertainty_f.

    f_g = uncertainty_f

    rng = np.random.default_rng(0)

    # Number of epistemic scenarios
    K = 120

    # Parameter dimension (3 energy groups)
    D = 3

    # Mean/stdev ranges for sig_t parameters (must stay positive)
    MU_LOW, MU_HIGH = 0.6, 1.2
    SIG_LOW, SIG_HIGH = 0.01, 0.05

    # Scenario means
    mu_s = rng.uniform(MU_LOW, MU_HIGH, size=(K, D))

    # Monte Carlo samples per scenario
    N = 100

    def _run_case(case_name, sigma_spec, sample_theta_fn, out_png):
        # Monte Carlo outputs per scenario (reference p-box)
        y_scenarios = []
        for k in range(K):
            theta = sample_theta_fn(k)
            ys = []
            for theta_i in theta:
                ys.append(f_g(theta_i, False)[0])
            y_scenarios.append(np.array(ys, dtype=float))

        # Plot both p-boxes on same axes
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        plot_pbox_mc(y_scenarios, ax=ax, alpha=0.20, label_prefix="MC")
        plot_pbox_gradient_delta(mu_s, sigma_spec, f_g, ax=ax, alpha=0.20, label_prefix="Grad-Δ")
        ax.set_title(case_name)
        plt.savefig(out_png)

    # -----------------------------
    # Case 1: 3 params, no covariance (diagonal)
    # -----------------------------
    sig_s_diag = rng.uniform(SIG_LOW, SIG_HIGH, size=(K, D))

    def _sample_diag(k):
        return rng.normal(mu_s[k], sig_s_diag[k], size=(N, D))

    _run_case(
        case_name="Multi-group p-box (3 params, no covariance) — y = middle channel",
        sigma_spec=sig_s_diag,
        sample_theta_fn=_sample_diag,
        out_png="uncertainty_mg_diag.png",
    )

    # -----------------------------
    # Case 2: 3 params, with covariance
    # -----------------------------
    sig_s_cov = rng.uniform(SIG_LOW, SIG_HIGH, size=(K, D))
    rho_s = rng.uniform(0.0, 0.8, size=K)  # equicorrelation per scenario (SPD for rho in (-0.5, 1))

    cov_s = np.zeros((K, D, D), dtype=float)
    for k in range(K):
        rho = float(rho_s[k])
        Corr = np.full((D, D), rho, dtype=float)
        np.fill_diagonal(Corr, 1.0)
        cov_s[k] = np.outer(sig_s_cov[k], sig_s_cov[k]) * Corr

    def _sample_cov(k):
        return rng.multivariate_normal(mu_s[k], cov_s[k], size=N)

    _run_case(
        case_name="Multi-group p-box (3 params, covariance) — y = middle channel",
        sigma_spec=cov_s,
        sample_theta_fn=_sample_cov,
        out_png="uncertainty_mg_cov.png",
    )
