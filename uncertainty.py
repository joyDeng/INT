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
    Gradient-based p-box using 1st-order delta method with a Normal approximation:
        Y = f(θ), θ ~ N(μ, σ^2)
        => approx Y ~ N( m = f(μ), v = (f'(μ))^2 σ^2 )

    The p-box is the envelope over scenarios (μ_k, σ_k).

    Parameters
    ----------
    mu_scenarios : array-like, shape (K,)
    sigma_scenarios : array-like, shape (K,)
    f : callable
        f(θ) -> y (scalar)
    fp : callable
        f'(θ) -> dy/dθ (scalar)
    y_grid : 1D array, optional
        If None, auto-chosen from the range of (m ± 4*s) across scenarios.
    grid_num : int
    y_grid_pad : float
        Relative padding when auto-choosing y_grid.
    min_std : float
        Floor on stddev to avoid division-by-zero.

    Returns
    -------
    y_grid : (M,) array
    F_low  : (M,) array
    F_high : (M,) array
    """
    if not _HAS_SCIPY:
        raise ImportError("scipy is required for gradient-based p-box (Normal CDF). Install with: pip install scipy")

    mu = np.asarray(mu_scenarios, dtype=float).ravel()
    sig = np.asarray(sigma_scenarios, dtype=float).ravel()
    if mu.shape != sig.shape:
        raise ValueError("mu_scenarios and sigma_scenarios must have the same shape.")

    # Compute approximate output Normal parameters per scenario

    m, g = zip(*(f_g(mui) for mui in mu))
    m = np.array(m, dtype=float).reshape(-1)
    g = np.array(g, dtype=float).reshape(-1)
    print(g)

    s = np.maximum(np.abs(g) * sig, min_std)

    # print(m.shape, sig.shape)
    # print(s.shape)
    # exit(0)

    # Auto grid from a wide band around means
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
    # Define a 1D nonlinear function and its derivative
    # def f(theta):
    #     # return theta + 0.3 * theta**2
    #     return np.exp( -0.01 * theta)

    # def fp(theta):
    #     # return 1.0 + 0.6 * theta
    #     return -0.01 * f(theta)

    # def f_g(theta):
    #     return f(theta), fp(theta)
    f_g = uncertainty_f


    # Epistemic scenarios: uncertain (mu, sigma) for θ ~ N(mu, sigma^2)
    rng = np.random.default_rng(0)
    K = 120
    mu_s = rng.uniform(-0.15, 0.15, size=K)
    sig_s = rng.uniform(0.01, 0.05, size=K)

    # Monte Carlo outputs per scenario (reference p-box)
    N = 100
    y_scenarios = []
    for mu, sig in zip(mu_s, sig_s):
        theta = rng.normal(mu, sig, size=N)
        ys = []
        print("mu, sig: ", mu, sig)
        for theta_i in theta:
            print("theta: ", theta_i)
            ys.append(f_g(theta_i, False)[0])
        y_scenarios.append(np.array(ys))

    # save_uncertainty_mc 
    # exit(0)

    # Plot both p-boxes on same axes
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    plot_pbox_mc(y_scenarios, ax=ax, alpha=0.20, label_prefix="MC")
    plot_pbox_gradient_delta(mu_s, sig_s, f_g, ax=ax, alpha=0.20, label_prefix="Grad-Δ")
    ax.set_title("Monte Carlo p-box vs Gradient (Delta-method) p-box")
    # plt.show()
    plt.savefig("uncertainty_geo.png")
