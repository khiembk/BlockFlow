"""Vocab-size-aware time reparameterization for categorical flow matching.

For the linear interpolant ``z_t = (1-t) * noise + t * one_hot(target)`` with
``noise ~ N(0, I_V)``, the probability that naive argmax(z_t) already recovers
the true token ("alpha", an exact order-statistics quantity) is a highly
nonlinear function of ``(t, V)``. For large vocab sizes, the informative
transition region -- where alpha moves from ~0 (task is hopeless) to ~1 (task
is already trivial) -- is compressed into a narrow band near t=1. Sampling t
uniformly on [0,1] for training (and spacing Euler steps uniformly in t at
generation time) wastes most of the budget on t values that carry little
training signal either way.

This reparameterizes: sample tau ~ Uniform(0,1) and map tau -> t through a
precomputed, vocab-size-specific lookup table built from the *inverse* of the
exact alpha(t, V) curve, so sampling/step-spacing is uniform in task
difficulty (alpha), not raw interpolation time.

Ported from the sibling FLM (Flow Language Model) codebase's
``utils.build_luts`` / ``utils.compute_alpha_exact`` (arXiv:2602.16813),
re-derived here to avoid a cross-repo dependency -- same Gauss-Hermite
quadrature over the exact order-statistics probability, same cubic-spline
inversion.
"""

import numpy as np
import torch
from numpy.polynomial.hermite import hermgauss
from scipy.interpolate import CubicSpline
from scipy.special import log_ndtr


def compute_alpha_exact(
    t: np.ndarray, vocab_size: int, n_gh: int = 100, sigma_floor: float = 1e-12
) -> np.ndarray:
    """P(argmax(z_t) == true token) for z_t = (1-t)*N(0,I_V) + t*one_hot(true),
    exact via Gauss-Hermite quadrature over the true coordinate's noise draw
    (the other V-1 coordinates are independent N(0, sigma^2), so conditional
    on the true coordinate's value the probability all V-1 others fall below
    it is a product of normal CDFs)."""
    t = np.asarray(t, dtype=np.float64)
    sigma = np.maximum(1.0 - t, sigma_floor)
    m = t / sigma  # standardized mean of the true coordinate
    x, w = hermgauss(n_gh)
    w = w / np.sqrt(np.pi)
    z_nodes = np.sqrt(2.0) * x
    log_cdf = log_ndtr(z_nodes[None, :] + m[:, None])
    log_prod = (vocab_size - 1) * log_cdf
    q = np.sum(w[None, :] * np.exp(log_prod), axis=-1)
    alpha = vocab_size / (vocab_size - 1.0) * (q - 1.0 / vocab_size)
    alpha += (t - 1) * 1e-10  # strict-monotonicity trick (matches FLM's LUT build)
    return np.clip(alpha, 0.0, 1.0)


class TimeReparam:
    """Caches the alpha(task-difficulty) -> t(raw interpolation time) spline
    for one vocab size. Built once (CPU, numpy/scipy); `tau_to_t` calls are
    cheap small-tensor CPU round-trips, negligible next to a transformer
    forward/backward pass."""

    def __init__(self, vocab_size: int, n_points: int = 10000):
        self.vocab_size = vocab_size
        t_grid = np.linspace(0.0, 1.0, n_points)
        alpha_grid = compute_alpha_exact(t_grid, vocab_size)
        order = np.argsort(alpha_grid)
        alpha_sorted, t_sorted = alpha_grid[order], t_grid[order]
        unique_alpha, unique_idx = np.unique(alpha_sorted, return_index=True)
        self._alpha_to_t = CubicSpline(unique_alpha, t_sorted[unique_idx])

    def tau_to_t(self, tau: torch.Tensor) -> torch.Tensor:
        """Maps tau (uniform-in-difficulty) to the raw interpolation time t
        that achieves that difficulty level for this vocab size."""
        dtype, device = tau.dtype, tau.device
        t = self._alpha_to_t(tau.detach().cpu().double().numpy())
        t = np.clip(t, 0.0, 1.0)
        return torch.from_numpy(t).to(device=device, dtype=dtype)
