#!/usr/bin/env python3
"""

Sequential experimental design simulator for a participant-level MEP recruitment curve model,
with posterior inference via Stan (CmdStanPy).

Implements 3 strategies:
  1) Random
  2) Myopic Expected Information Gain (nested Monte Carlo)
  3) Policy-based (user-provided black box function)

Outputs:
  - Timestamped .npz results with raw histories, true parameters, metrics, config
  - Timestamped metadata .json
  - Plots:
      A) Mean posterior entropy vs t
      B) RMSE of threshold a vs t
    C) Posterior predictive frames (per strategy, per run)
    D) Posterior of a frames (per strategy, per run)

Requires:
  numpy, scipy, matplotlib, tqdm, cmdstanpy, sklearn (optional for kNN entropy)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

import torch

from cmdstanpy import CmdStanModel

from scipy.special import expit, gammaln
from scipy.stats import gaussian_kde, truncnorm
from tqdm import tqdm

# matplotlib: only import when needed (faster startup on headless)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import logging
logger = logging.getLogger("cmdstanpy")
logger.addHandler(logging.NullHandler())
logger.propagate = False
logger.setLevel(logging.CRITICAL)


# =========================
# Config
# =========================

@dataclass
class PriorConfig:
    a_mean: float = 50.0
    a_scale: float = 10.0
    a_low: float = 0.0
    a_high: float = 100.0

    b_scale: float = 1.0          # HalfNormal
    L_scale: float = 0.1          # HalfNormal
    H_scale: float = 5.0          # HalfNormal
    ell_scale: float = 5.0        # HalfNormal
    c1_scale: float = 5.0         # HalfNormal
    c2_scale: float = 0.5         # HalfNormal


@dataclass
class MCMCConfig:
    chains: int = 4
    num_samples: int = 800
    warmup_steps: int = 800
    adapt_delta: float = 0.9
    max_treedepth: int = 12
    seed: int = 12345
    parallel_chains: int = 4


@dataclass
class EIGConfig:
    K_grid: int = 41              # candidate x grid size
    N_outer: int = 64             # outer samples
    M_inner: int = 128            # inner samples (posterior draws for marginal)
    x_grid_low: float = 0.0
    x_grid_high: float = 100.0
    # For speed, optionally subsample posterior draws when very large:
    max_posterior_draws_for_eig: int = 5000


@dataclass
class EntropyConfig:
    method: str = "gaussian"      # "gaussian" or "knn"
    knn_k: int = 5                # used if method="knn"
    # small jitter for covariance stability:
    cov_jitter: float = 1e-8


@dataclass
class SimConfig:
    R: int = 10                   # independent runs
    T: int = 20                   # time steps per run
    num_initial_zero: int = 0     # initial number of samples at intensity x=0 before strategies start
    num_initial_uniform: int = 0  # initial number of samples with x ~ Uniform(0,100) before strategies start
    tracked_run_idx: int = 0      # which run to produce per-t frames
    out_dir: str = "output"
    # Posterior draw thinning for metrics/plots to reduce compute
    max_draws_for_metrics: int = 5000
    max_draws_for_frames: int = 2000
    # Numerical floors
    mu_floor: float = 1e-12
    y_floor: float = 1e-300


# =========================
# Model math (Python)
# =========================

PARAM_NAMES = ["a", "b", "L", "H", "ell", "c1", "c2"]


def stable_mu(
    x: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    L: np.ndarray,
    H: np.ndarray,
    ell: np.ndarray,
    mu_floor: float = 1e-12,
) -> np.ndarray:
    """
    μ = L + max(0, -ell + (H+ell) * sigmoid(b(x-a) - log(H/ell)))

    Vectorized over x and parameters with broadcasting.
    """
    # Ensure strictly positive for log(H/ell)
    H_pos = np.maximum(H, 1e-300)
    ell_pos = np.maximum(ell, 1e-300)
    z = b * (x - a) - np.log(H_pos / ell_pos)
    s = expit(z)
    core = -ell + (H + ell) * s
    mu = L + np.maximum(0.0, core)
    return np.maximum(mu, mu_floor)


def gamma_alpha_beta(mu: np.ndarray, c1: np.ndarray, c2: np.ndarray, mu_floor: float = 1e-12) -> Tuple[np.ndarray, np.ndarray]:
    """
    y ~ Gamma(shape=α, rate=β)
      α = μ * β
      β = 1/c1 + 1/(c2 * μ)

    Returns (alpha, beta), vectorized with broadcasting.
    """
    mu = np.maximum(mu, mu_floor)
    c1_pos = np.maximum(c1, 1e-300)
    c2_pos = np.maximum(c2, 1e-300)
    beta = 1.0 / c1_pos + 1.0 / (c2_pos * mu)
    alpha = mu * beta
    alpha = np.maximum(alpha, 1e-300)
    beta = np.maximum(beta, 1e-300)
    return alpha, beta


def log_gamma_pdf(y: np.ndarray, alpha: np.ndarray, beta: np.ndarray, y_floor: float = 1e-300) -> np.ndarray:
    """
    Log density of Gamma(shape=alpha, rate=beta) at y.
    """
    y = np.maximum(y, y_floor)
    # log f = alpha*log(beta) - gammaln(alpha) + (alpha-1)*log(y) - beta*y
    return alpha * np.log(beta) - gammaln(alpha) + (alpha - 1.0) * np.log(y) - beta * y


def simulate_y(rng: np.random.Generator, mu: float, c1: float, c2: float, mu_floor: float) -> float:
    alpha, beta = gamma_alpha_beta(np.array(mu), np.array(c1), np.array(c2), mu_floor=mu_floor)
    shape = float(alpha)
    scale = float(1.0 / beta)
    return float(rng.gamma(shape=shape, scale=scale))


# =========================
# Priors (Python)
# =========================

def sample_halfnormal(rng: np.random.Generator, scale: float, size: Optional[int] = None) -> np.ndarray:
    return np.abs(rng.normal(loc=0.0, scale=scale, size=size))


def sample_truncnorm_a(rng: np.random.Generator, mean: float, sd: float, low: float, high: float, size: Optional[int] = None) -> np.ndarray:
    a, b = (low - mean) / sd, (high - mean) / sd
    return truncnorm.rvs(a=a, b=b, loc=mean, scale=sd, size=size, random_state=rng)


def sample_prior_theta(rng: np.random.Generator, prior: PriorConfig) -> Dict[str, float]:
    a = float(sample_truncnorm_a(rng, prior.a_mean, prior.a_scale, prior.a_low, prior.a_high))
    b = float(sample_halfnormal(rng, prior.b_scale))
    L = float(sample_halfnormal(rng, prior.L_scale))
    H = float(sample_halfnormal(rng, prior.H_scale))
    ell = float(sample_halfnormal(rng, prior.ell_scale))
    c1 = float(sample_halfnormal(rng, prior.c1_scale))
    c2 = float(sample_halfnormal(rng, prior.c2_scale))
    return {"a": a, "b": b, "L": L, "H": H, "ell": ell, "c1": c1, "c2": c2}


def draw_prior_samples(rng: np.random.Generator, prior: PriorConfig, S: int) -> np.ndarray:
    """
    Returns array of shape (S, 7) in PARAM_NAMES order.
    """
    a = sample_truncnorm_a(rng, prior.a_mean, prior.a_scale, prior.a_low, prior.a_high, size=S)
    b = sample_halfnormal(rng, prior.b_scale, size=S)
    L = sample_halfnormal(rng, prior.L_scale, size=S)
    H = sample_halfnormal(rng, prior.H_scale, size=S)
    ell = sample_halfnormal(rng, prior.ell_scale, size=S)
    c1 = sample_halfnormal(rng, prior.c1_scale, size=S)
    c2 = sample_halfnormal(rng, prior.c2_scale, size=S)
    return np.stack([a, b, L, H, ell, c1, c2], axis=1)


# =========================
# Stan model
# =========================

def stan_model_code(mu_floor: float = 1e-12) -> str:
    """
    Stan code as a string; supports arbitrary N observations.
    """
    # mu_floor embedded as literal for Stan
    return f"""
functions {{
  real stable_mu(real x, real a, real b, real L, real H, real ell) {{
    // μ = L + max(0, -ell + (H + ell) * inv_logit(b(x-a) - log(H/ell)))
    real Hpos = fmax(H, 1e-300);
    real ellpos = fmax(ell, 1e-300);
    real z = b * (x - a) - log(Hpos / ellpos);
    real s = inv_logit(z);
    real core = -ell + (H + ell) * s;
    real mu = L + fmax(0.0, core);
    return fmax(mu, {mu_floor});
  }}

  real gamma_logpdf_rate(real y, real alpha, real beta) {{
    // y ~ Gamma(shape=alpha, rate=beta)
    // log f = alpha*log(beta) - lgamma(alpha) + (alpha-1)*log(y) - beta*y
    real ypos = fmax(y, 1e-300);
    return alpha * log(beta) - lgamma(alpha) + (alpha - 1) * log(ypos) - beta * ypos;
  }}
}}

data {{
  int<lower=0> N;
  array[N] real<lower=0, upper=100> x;
  array[N] real<lower=0> y;

  // Prior hyperparameters passed from Python
  real a_mean;
  real<lower=0> a_scale;
  real<lower=0> a_low;
  real<lower=0> a_high;

  real<lower=0> b_scale;
  real<lower=0> L_scale;
  real<lower=0> H_scale;
  real<lower=0> ell_scale;
  real<lower=0> c1_scale;
  real<lower=0> c2_scale;
}}

parameters {{
  real<lower=0, upper=100> a;
  real<lower=0> b;
  real<lower=0> L;
  real<lower=0> H;
  real<lower=0> ell;
  real<lower=0> c1;
  real<lower=0> c2;
}}

model {{
  // TruncatedNormal for a via normal prior + bounds on parameter
  a ~ normal(a_mean, a_scale);

  // HalfNormal implemented as normal(0, scale) with lower=0 parameter
  b ~ normal(0, b_scale);
  L ~ normal(0, L_scale);
  H ~ normal(0, H_scale);
  ell ~ normal(0, ell_scale);
  c1 ~ normal(0, c1_scale);
  c2 ~ normal(0, c2_scale);

  // Likelihood
  for (n in 1:N) {{
    real mu = stable_mu(x[n], a, b, L, H, ell);
    real beta = 1 / fmax(c1, 1e-300) + 1 / (fmax(c2, 1e-300) * mu);
    real alpha = mu * beta;
    target += gamma_logpdf_rate(y[n], alpha, beta);
  }}
}}
"""


def compile_stan_model(cache_dir: Path, mu_floor: float) -> CmdStanModel:
    """
    Writes Stan model to cache_dir and compiles once.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    stan_path = cache_dir / "mep_recruitment_gamma.stan"
    code = stan_model_code(mu_floor=mu_floor)
    stan_path.write_text(code, encoding="utf-8")
    return CmdStanModel(stan_file=str(stan_path), cpp_options={'STAN_THREADS': True})


# =========================
# Posterior inference
# =========================

def stan_data_from_history(
    x_hist: np.ndarray,
    y_hist: np.ndarray,
    prior: PriorConfig,
) -> Dict:
    x_hist = np.asarray(x_hist, dtype=float)
    y_hist = np.asarray(y_hist, dtype=float)
    assert x_hist.shape == y_hist.shape
    N = int(x_hist.size)
    return {
        "N": N,
        "x": x_hist.tolist() if N > 0 else [],
        "y": y_hist.tolist() if N > 0 else [],
        "a_mean": prior.a_mean,
        "a_scale": prior.a_scale,
        "a_low": prior.a_low,
        "a_high": prior.a_high,
        "b_scale": prior.b_scale,
        "L_scale": prior.L_scale,
        "H_scale": prior.H_scale,
        "ell_scale": prior.ell_scale,
        "c1_scale": prior.c1_scale,
        "c2_scale": prior.c2_scale,
    }


def fit_posterior_stan(
    model: CmdStanModel,
    x_hist: np.ndarray,
    y_hist: np.ndarray,
    prior: PriorConfig,
    mcmc: MCMCConfig,
) -> np.ndarray:
    """
    Returns posterior draws as array shape (S, 7) in PARAM_NAMES order.
    """
    data = stan_data_from_history(x_hist, y_hist, prior)
    fit = model.sample(
        data=data,
        chains=mcmc.chains,
        iter_sampling=mcmc.num_samples,
        iter_warmup=mcmc.warmup_steps,
        seed=mcmc.seed,
        adapt_delta=mcmc.adapt_delta,
        max_treedepth=mcmc.max_treedepth,
        parallel_chains=mcmc.parallel_chains,
        show_progress=False,
    )
    draws = fit.draws_pd(vars=PARAM_NAMES)  # pandas DataFrame
    arr = draws.to_numpy(dtype=float)
    return arr


def maybe_subsample_rows(rng: np.random.Generator, arr: np.ndarray, max_rows: int) -> np.ndarray:
    if arr.shape[0] <= max_rows:
        return arr
    idx = rng.choice(arr.shape[0], size=max_rows, replace=False)
    return arr[idx]


# =========================
# Entropy estimation
# =========================

def entropy_gaussian(draws: np.ndarray, jitter: float = 1e-8) -> float:
    """
    Differential entropy of multivariate Gaussian with sample covariance.
    H = 0.5 * log( (2πe)^d det Σ )
    """
    X = np.asarray(draws, dtype=float)
    d = X.shape[1]
    cov = np.cov(X, rowvar=False)
    cov = cov + jitter * np.eye(d)
    sign, logdet = np.linalg.slogdet(cov)
    if not np.isfinite(logdet) or sign <= 0:
        # fallback: add more jitter
        cov = cov + (100.0 * jitter) * np.eye(d)
        sign, logdet = np.linalg.slogdet(cov)
    return float(0.5 * (d * np.log(2.0 * np.pi * np.e) + logdet))


def entropy_knn(draws: np.ndarray, k: int = 5) -> float:
    """
    Kozachenko–Leonenko kNN differential entropy estimator.
    Requires sklearn. Uses Chebyshev / infinity norm distances (standard variant).
    """
    try:
        from sklearn.neighbors import NearestNeighbors
    except Exception as e:
        raise RuntimeError("sklearn is required for kNN entropy. Install scikit-learn or use method='gaussian'.") from e

    X = np.asarray(draws, dtype=float)
    n, d = X.shape
    if n <= k + 1:
        return float("nan")

    # Use infinity norm for computational stability
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="chebyshev").fit(X)
    distances, _ = nbrs.kneighbors(X)
    eps = distances[:, -1]  # distance to k-th neighbor (excluding self)
    eps = np.maximum(eps, 1e-300)

    # KL estimator constant terms
    # H ≈ ψ(n) - ψ(k) + log(V_d) + d * mean(log(eps))   (up to metric constants)
    # For Chebyshev (L∞), unit ball volume V_d = 2^d.
    from scipy.special import digamma
    V_d = 2.0 ** d
    return float(digamma(n) - digamma(k) + np.log(V_d) + d * np.mean(np.log(eps)))


def estimate_entropy(draws: np.ndarray, cfg: EntropyConfig) -> float:
    if cfg.method.lower() == "gaussian":
        return entropy_gaussian(draws, jitter=cfg.cov_jitter)
    if cfg.method.lower() == "knn":
        return entropy_knn(draws, k=cfg.knn_k)
    raise ValueError(f"Unknown entropy method: {cfg.method}")


# =========================
# Myopic EIG (nested MC)
# =========================

def estimate_myopic_eig_for_x(
    rng: np.random.Generator,
    x: float,
    post_draws: np.ndarray,   # (S,7)
    eig_cfg: EIGConfig,
    sim_cfg: SimConfig,
) -> float:
    """
    Î(x) = (1/N) Σ [ log p(y_n | θ_n, x) - log p(y_n | x, h) ]
    where p(y_n | x, h) approximated by log-mean-exp over M posterior draws.
    """
    draws = post_draws
    if draws.shape[0] > eig_cfg.max_posterior_draws_for_eig:
        draws = maybe_subsample_rows(rng, draws, eig_cfg.max_posterior_draws_for_eig)

    S = draws.shape[0]
    N = eig_cfg.N_outer
    M = min(eig_cfg.M_inner, S)

    # Pre-sample indices for efficiency
    idx_outer = rng.integers(low=0, high=S, size=N)
    idx_inner = rng.integers(low=0, high=S, size=(N, M))

    # Unpack draws columns
    aS, bS, LS, HS, ellS, c1S, c2S = [draws[:, i] for i in range(7)]

    # For each outer sample, simulate y_n from theta_n at x
    x_arr = np.array(x, dtype=float)

    theta_n = draws[idx_outer, :]  # (N,7)
    mu_n = stable_mu(
        x_arr,
        theta_n[:, 0], theta_n[:, 1], theta_n[:, 2], theta_n[:, 3], theta_n[:, 4],
        mu_floor=sim_cfg.mu_floor
    )
    alpha_n, beta_n = gamma_alpha_beta(mu_n, theta_n[:, 5], theta_n[:, 6], mu_floor=sim_cfg.mu_floor)

    # Sample y_n ~ Gamma(shape=alpha_n, rate=beta_n)
    y_n = rng.gamma(shape=alpha_n, scale=1.0 / beta_n)  # (N,)

    # Compute log p(y_n | theta_n, x)
    logp_cond = log_gamma_pdf(y_n, alpha_n, beta_n, y_floor=sim_cfg.y_floor)  # (N,)

    # Compute log p(y_n | x, h) via log-mean-exp over M inner draws per n
    # Build inner parameters arrays (N,M)
    a_m = aS[idx_inner]
    b_m = bS[idx_inner]
    L_m = LS[idx_inner]
    H_m = HS[idx_inner]
    ell_m = ellS[idx_inner]
    c1_m = c1S[idx_inner]
    c2_m = c2S[idx_inner]

    # mu_m shape (N,M) by broadcasting x
    mu_m = stable_mu(
        x_arr,
        a_m, b_m, L_m, H_m, ell_m,
        mu_floor=sim_cfg.mu_floor
    )
    alpha_m, beta_m = gamma_alpha_beta(mu_m, c1_m, c2_m, mu_floor=sim_cfg.mu_floor)

    # log p(y_n | theta_m, x): need y_n broadcast to (N,M)
    y_nm = y_n[:, None]
    logp_m = log_gamma_pdf(y_nm, alpha_m, beta_m, y_floor=sim_cfg.y_floor)  # (N,M)

    # log-mean-exp across M
    maxv = np.max(logp_m, axis=1, keepdims=True)
    lme = (maxv + np.log(np.mean(np.exp(logp_m - maxv), axis=1, keepdims=True))).reshape(-1)  # (N,)
    logp_marg = lme

    return float(np.mean(logp_cond - logp_marg))


def choose_x_myopic_eig(
    rng: np.random.Generator,
    post_draws: np.ndarray,
    eig_cfg: EIGConfig,
    sim_cfg: SimConfig,
    pbar: Optional[tqdm] = None,
) -> float:
    """
    Grid search over K candidates in [0,100].
    """
    x_grid = np.linspace(eig_cfg.x_grid_low, eig_cfg.x_grid_high, eig_cfg.K_grid)
    eig_vals = np.empty_like(x_grid)

    it = range(len(x_grid))
    if pbar is None:
        for i in it:
            eig_vals[i] = estimate_myopic_eig_for_x(rng, float(x_grid[i]), post_draws, eig_cfg, sim_cfg)
    else:
        for i in it:
            eig_vals[i] = estimate_myopic_eig_for_x(rng, float(x_grid[i]), post_draws, eig_cfg, sim_cfg)
            pbar.set_postfix({"eig_best": float(np.max(eig_vals[: i + 1]))})

    return float(x_grid[int(np.argmax(eig_vals))])


# =========================
# Policy hook
# =========================

def default_policy(history: List[Tuple[float, float]]) -> float:
    """
    Placeholder policy. Users should replace with their own policy(history)->x_next.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the raw PyTorch model (no implicit .eval())
    mlrun_path = "mlruns/250577988941689306/d748591dcf384679ba786cd940e22a36/artifacts/model"
    pth = "/home/dantwili/repos/hbmep_dad/" + mlrun_path + "/data/model.pth"
    mep_model = torch.load(pth, map_location=device).to(device)

    design_net = mep_model.design_net

    # Build tensors in the same shapes used during training
    xi_designs = [torch.tensor([[[float(x)]]], device=device) for x, _ in history]
    y_outcomes = [torch.tensor([[float(y)]], device=device) for _, y in history]

    with torch.no_grad():
        if history:
            xi_next = design_net(*zip(xi_designs, y_outcomes))
        else:
            xi_next = design_net()

    return float(xi_next.squeeze().cpu().item())


# =========================
# Posterior predictive summaries
# =========================

def posterior_predictive_mean_sd(
    x_grid: np.ndarray,
    draws: np.ndarray,   # (S,7)
    sim_cfg: SimConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns predictive mean and SD of y at each x in x_grid via:
      E[y] = E_theta[mu(x,theta)]
      Var[y] = E_theta[Var(y|theta)] + Var_theta[mu]
    where Var(y|theta) = mu / beta.
    """
    xg = np.asarray(x_grid, dtype=float)[None, :]  # (1,G)
    d = draws
    a, b, L, H, ell, c1, c2 = [d[:, i][:, None] for i in range(7)]  # (S,1)

    mu = stable_mu(xg, a, b, L, H, ell, mu_floor=sim_cfg.mu_floor)  # (S,G)
    alpha, beta = gamma_alpha_beta(mu, c1, c2, mu_floor=sim_cfg.mu_floor)  # (S,G)
    var_cond = mu / beta  # (S,G)

    mean_mu = np.mean(mu, axis=0)                         # (G,)
    var_mu = np.var(mu, axis=0, ddof=1) if mu.shape[0] > 1 else np.zeros(mu.shape[1])
    mean_var_cond = np.mean(var_cond, axis=0)             # (G,)

    var_total = mean_var_cond + var_mu
    sd_total = np.sqrt(np.maximum(var_total, 0.0))
    return mean_mu, sd_total


# =========================
# Plotting helpers
# =========================

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def title_block(sim: SimConfig, prior: PriorConfig, mcmc: MCMCConfig, eig: EIGConfig, entropy: EntropyConfig) -> str:
    return (
        f"R={sim.R}, T={sim.T} | "
        f"Priors: a~TN({prior.a_mean},{prior.a_scale})[{prior.a_low},{prior.a_high}], "
        f"b~HN({prior.b_scale}), L~HN({prior.L_scale}), H~HN({prior.H_scale}), "
        f"ell~HN({prior.ell_scale}), c1~HN({prior.c1_scale}), c2~HN({prior.c2_scale}) | "
        f"MCMC: chains={mcmc.chains}, samp={mcmc.num_samples}, warmup={mcmc.warmup_steps}, "
        f"adapt_delta={mcmc.adapt_delta}, treedepth={mcmc.max_treedepth} | "
        f"EIG: K={eig.K_grid}, N={eig.N_outer}, M={eig.M_inner} | "
        f"Entropy={entropy.method}"
    )


def plot_mean_entropy(
    out_path: Path,
    entropies: Dict[str, np.ndarray],  # strategy -> (R,T+1)
    sim: SimConfig,
    prior: PriorConfig,
    mcmc: MCMCConfig,
    eig: EIGConfig,
    entropy_cfg: EntropyConfig,
) -> None:
    plt.figure()
    t = np.arange(sim.T + 1)

    for name, arr in entropies.items():
        mean = np.nanmean(arr, axis=0)
        se = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(arr.shape[0])

        line = plt.plot(t, mean, linewidth=2.0, label=name)[0]

        # Solid dots
        plt.scatter(t, mean, s=25, color=line.get_color(), zorder=3)

        # Small vertical SE bars with caps
        plt.errorbar(
            t,
            mean,
            yerr=se,
            fmt="none",
            ecolor=line.get_color(),
            elinewidth=1.2,
            capsize=3,
            alpha=0.9,
        )

    plt.xlabel("t")
    plt.ylabel("Posterior entropy H(theta | h_t)")
    plt.title("Mean Posterior Entropy vs t")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def plot_rmse_a(
    out_path: Path,
    se_a: Dict[str, np.ndarray],  # strategy -> (R,T+1) squared error
    sim: SimConfig,
    prior: PriorConfig,
    mcmc: MCMCConfig,
    eig: EIGConfig,
    entropy_cfg: EntropyConfig,
) -> None:
    plt.figure()
    t = np.arange(sim.T + 1)

    for name, arr in se_a.items():
        rmse = np.sqrt(np.nanmean(arr, axis=0))
        rmse_per_run = np.sqrt(arr)
        se = np.nanstd(rmse_per_run, axis=0, ddof=1) / np.sqrt(arr.shape[0])

        line = plt.plot(t, rmse, linewidth=2.0, label=name)[0]

        # Solid dots
        plt.scatter(t, rmse, s=25, color=line.get_color(), zorder=3)

        # Small vertical SE bars with caps
        plt.errorbar(
            t,
            rmse,
            yerr=se,
            fmt="none",
            ecolor=line.get_color(),
            elinewidth=1.2,
            capsize=3,
            alpha=0.9,
        )

    plt.ylim(bottom=0)
    plt.xlabel("t")
    plt.ylabel("RMSE of threshold a")
    plt.title("RMSE(a) vs t")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def plot_posterior_predictive_frames(
    frames_dir: Path,
    strategy: str,
    tracked: Dict,
    sim: SimConfig,
    t_values: Optional[List[int]] = None,
) -> None:
    """
    tracked contains:
      - true_theta: (7,)
      - x_hist: (num_initial_total+T,)
      - y_hist: (num_initial_total+T,)
      - draws_list: list length (num_initial_total+T+1), starting from prior
    """
    ensure_dir(frames_dir)
    theta_true = tracked["true_theta"]
    x_hist = tracked["x_hist"]
    y_hist = tracked["y_hist"]
    draws_list = tracked["draws_list"]

    x_grid = np.linspace(0.0, 100.0, 400)
    # True mu curve
    mu_true = stable_mu(
        x_grid,
        theta_true[0], theta_true[1], theta_true[2], theta_true[3], theta_true[4],
        mu_floor=sim.mu_floor
    )

    # Fixed y-axis scaling
    y_min = 0
    y_max = float(np.max(mu_true) * 1.5 + 1e-6)

    if t_values is None:
        t_values = list(range(sim.T + 1))
    if len(t_values) != len(draws_list):
        raise ValueError("t_values must have the same length as draws_list")

    for frame_idx, t in enumerate(t_values):
        draws = draws_list[frame_idx]
        draws = draws if draws.shape[0] <= sim.max_draws_for_frames else draws[: sim.max_draws_for_frames]

        mean_pred, sd_pred = posterior_predictive_mean_sd(x_grid, draws, sim)
        lo = mean_pred - sd_pred
        hi = mean_pred + sd_pred

        plt.figure()
        plt.plot(x_grid, mu_true, label="True μ(x)")
        # Vertical line at true threshold a
        plt.axvline(theta_true[0], linestyle="--", linewidth=1.0)

        mean_line = plt.plot(x_grid, mean_pred, color="orange", label="Posterior predictive mean")[0]
        mean_color = mean_line.get_color()
        # Match the ±1 SD shading color to the posterior mean curve color
        plt.fill_between(x_grid, lo, hi, color=mean_color, alpha=0.2, label="±1 SD")

        n_samples = frame_idx
        if n_samples > 0:
            # Plot the *actual observed responses* (x_i, y_i) as green dots.
            plt.scatter(
                x_hist[:n_samples],
                y_hist[:n_samples],
                s=25,
                color="green",
                label="Sample",
            )
            # Newest sample as a larger green star
            plt.scatter(
                [x_hist[n_samples - 1]],
                [y_hist[n_samples - 1]],
                marker="*",
                s=180,
                color="green",
                label="Newest sample",
            )

        plt.xlim(0, 100)
        plt.ylim(y_min, y_max)
        plt.xlabel("Intensity x")
        plt.ylabel("MEP size")
        plt.title(f"{strategy} | Posterior Predictive of y | Frame t={t}")
        plt.legend(loc="best")
        plt.tight_layout()

        t_str = f"{t:03d}" if t >= 0 else f"-{abs(t):03d}"
        out_path = frames_dir / f"frame_{frame_idx}_{t_str}.png"
        plt.savefig(out_path, dpi=200)
        plt.close()



def plot_posterior_a_frames(
    frames_dir: Path,
    strategy: str,
    tracked: Dict,
    sim: SimConfig,
    t_values: Optional[List[int]] = None,
) -> None:
    """
    Plot true μ(x) curve on primary axis and KDE of posterior a on secondary axis.

    - Sample points are plotted at the *actual observed responses* (x_t, y_t) in green.
    - The KDE y-axis is auto-scaled per frame so the posterior density is always clearly visible.
    """
    ensure_dir(frames_dir)
    theta_true = tracked["true_theta"]
    x_hist = tracked["x_hist"]
    y_hist = tracked.get("y_hist", None)
    draws_list = tracked["draws_list"]

    x_grid = np.linspace(0.0, 100.0, 400)
    mu_true = stable_mu(
        x_grid,
        theta_true[0], theta_true[1], theta_true[2], theta_true[3], theta_true[4],
        mu_floor=sim.mu_floor
    )

    # Fixed scaling for the μ(x) curve axis
    y1_min = 0.0
    y1_max = float(np.max(mu_true) * 1.1 + 1e-6)

    a_eval = np.linspace(0.0, 100.0, 500)

    if t_values is None:
        t_values = list(range(sim.T + 1))
    if len(t_values) != len(draws_list):
        raise ValueError("t_values must have the same length as draws_list")

    for frame_idx, t in enumerate(t_values):
        plt.figure()
        ax1 = plt.gca()

        ax1.plot(x_grid, mu_true, label="True μ(x)")

        n_samples = frame_idx
        if n_samples > 0 and y_hist is not None:
            ax1.scatter(
                x_hist[:n_samples],
                y_hist[:n_samples],
                s=25,
                color="green",
                label="Sample",
            )
            ax1.scatter(
                [x_hist[n_samples - 1]],
                [y_hist[n_samples - 1]],
                marker="*",
                s=180,
                color="green",
                label="Newest sample",
            )


        ax1.set_xlim(0, 100)
        ax1.set_ylim(y1_min, y1_max)
        ax1.set_xlabel("Intensity x")
        ax1.set_ylabel("True μ(x)")

        draws = draws_list[frame_idx]
        a_samp = draws[:, 0]
        if a_samp.size < 2 or np.allclose(np.std(a_samp), 0.0):
            dens = np.zeros_like(a_eval)
        else:
            dens = gaussian_kde(a_samp)(a_eval)

        ax2 = ax1.twinx()
        # Make posterior density distinct from the μ(x) curve
        ax2.fill_between(a_eval, 0.0, dens, color="orange", alpha=0.15)
        ax2.plot(a_eval, dens, color="orange", alpha=0.8)
        ax2.axvline(theta_true[0], linestyle="--", linewidth=1.0)

        dens_max = float(np.max(dens)) if dens.size else 0.0
        ax2.set_ylim(0.0, max(dens_max * 1.10, 1e-12))
        ax2.set_ylabel("Posterior density of a")
        # Two fixed decimals on the right y-axis
        ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

        ax1.set_title(f"{strategy} | Posterior of a | Frame t={t}")
        ax1.legend(loc="upper left")
        plt.tight_layout()

        t_str = f"{t:03d}" if t >= 0 else f"-{abs(t):03d}"
        out_path = frames_dir / f"frame_{frame_idx}_{t_str}.png"
        plt.savefig(out_path, dpi=200)
        plt.close()


# =========================
# Simulation core
# =========================


def run_strategy(
    strategy_name: str,
    rng: np.random.Generator,
    stan_model: CmdStanModel,
    sim: SimConfig,
    prior: PriorConfig,
    mcmc: MCMCConfig,
    eig: EIGConfig,
    entropy_cfg: EntropyConfig,
    policy_fn: Optional[Callable[[List[Tuple[float, float]]], float]] = None,
    true_theta_fixed: Optional[np.ndarray] = None,
    initial_x: Optional[np.ndarray] = None,  # (R, num_initial_total)
    initial_y: Optional[np.ndarray] = None,  # (R, num_initial_total)
) -> Dict:
    """
    Returns dict with:
      histories_x: (R,T) float
      histories_y: (R,T) float
      true_theta:  (R,7) float
      entropy:     (R,T+1) float
      a_mean:      (R,T+1) float
      se_a:        (R,T+1) float
            tracked_runs: list length R with per-run draws/history for frames
    """
    R, T = sim.R, sim.T
    m0 = int(sim.num_initial_zero + sim.num_initial_uniform)
    total_len = m0 + T
    histories_x = np.full((R, total_len), np.nan, dtype=float)
    histories_y = np.full((R, total_len), np.nan, dtype=float)
    true_theta = np.zeros((R, 7), dtype=float)

    entropy = np.full((R, T + 1), np.nan, dtype=float)
    a_mean = np.full((R, T + 1), np.nan, dtype=float)
    se_a = np.full((R, T + 1), np.nan, dtype=float)

    tracked_runs: List[Dict] = [
        {
            "true_theta": None,
            "x_hist": None,
            "y_hist": None,
            "draws_list": [],
        }
        for _ in range(R)
    ]

    runs_pbar = tqdm(range(R), desc=f"{strategy_name}: runs", leave=True)

    for r in runs_pbar:
        if true_theta_fixed is None:
            theta = sample_prior_theta(rng, prior)
            theta_vec = np.array([theta[k] for k in PARAM_NAMES], dtype=float)
        else:
            theta_vec = true_theta_fixed[r].copy()
            theta = {k: float(theta_vec[i]) for i, k in enumerate(PARAM_NAMES)}
        true_theta[r, :] = theta_vec

        # Prior draws for frames and (if no data) EIG
        prior_draws = draw_prior_samples(rng, prior, S=min(sim.max_draws_for_metrics, 3000))
        # Strategy-time t=0 metrics will be computed after applying initial samples.

        tracked_run = {
            "true_theta": theta_vec.copy(),
            "x_hist": np.full(total_len, np.nan, dtype=float),
            "y_hist": np.full(total_len, np.nan, dtype=float),
            "draws_list": [],
        }
        tracked_runs[r] = tracked_run

        x_hist: List[float] = []
        y_hist: List[float] = []

        # -------------------------------------------------
        # Apply initial samples (uniform x in [0,100])
        # -------------------------------------------------
        if m0 > 0:
            if initial_x is None or initial_y is None:
                raise ValueError("initial_x/initial_y must be provided when num_initial_total > 0")
            x0 = np.asarray(initial_x[r], dtype=float).tolist()
            y0 = np.asarray(initial_y[r], dtype=float).tolist()
            x_hist.extend(x0)
            y_hist.extend(y0)
            histories_x[r, :m0] = np.array(x0, dtype=float)
            histories_y[r, :m0] = np.array(y0, dtype=float)
            tracked_run["x_hist"][:m0] = np.array(x0, dtype=float)
            tracked_run["y_hist"][:m0] = np.array(y0, dtype=float)

        # -------------------------------------------------
        # Build tracked frames for negative t (tracked run only)
        # Frames t = -m0,...,-1,0,...,T
        # draws_list[0] corresponds to t=-m0 (prior)
        # draws_list[k] corresponds to t=-m0+k
        # -------------------------------------------------
        tracked_run["draws_list"].append(
            prior_draws[: min(prior_draws.shape[0], sim.max_draws_for_frames)].copy()
        )
        for k in range(1, m0 + 1):
            post_k = fit_posterior_stan(
                stan_model,
                np.array(x_hist[:k], dtype=float),
                np.array(y_hist[:k], dtype=float),
                prior,
                mcmc,
            )
            post_k = maybe_subsample_rows(rng, post_k, sim.max_draws_for_frames)
            tracked_run["draws_list"].append(post_k.copy())

        # -------------------------------------------------
        # Strategy-time t=0 metrics: posterior after initial samples
        # -------------------------------------------------
        if len(x_hist) == 0:
            post_draws_t0 = prior_draws
        else:
            post_draws_t0 = fit_posterior_stan(
                stan_model,
                np.array(x_hist, dtype=float),
                np.array(y_hist, dtype=float),
                prior,
                mcmc,
            )
        post_draws_t0 = maybe_subsample_rows(rng, post_draws_t0, sim.max_draws_for_metrics)
        entropy[r, 0] = estimate_entropy(post_draws_t0, entropy_cfg)
        a_mean[r, 0] = float(np.mean(post_draws_t0[:, 0]))
        se_a[r, 0] = (a_mean[r, 0] - theta_vec[0]) ** 2

        for t in tqdm(range(1, T + 1), desc=f"{strategy_name}: t", leave=False):
            # Choose x_t
            if strategy_name.lower() == "random":
                x_t = float(rng.uniform(0.0, 100.0))

            elif strategy_name.lower() == "myopic_eig":
                # Fit posterior given history up to time t-1.
                # If we have initial samples, then at t=1 we must condition on them (not the prior).
                if t == 1:
                    if len(x_hist) == 0:
                        post_draws = prior_draws
                    else:
                        # Reuse t=0 posterior (after initial samples) for efficiency.
                        post_draws = post_draws_t0
                else:
                    post_draws = fit_posterior_stan(
                        stan_model,
                        np.array(x_hist, dtype=float),
                        np.array(y_hist, dtype=float),
                        prior,
                        mcmc,
                    )
                post_draws = maybe_subsample_rows(rng, post_draws, sim.max_draws_for_metrics)
                x_t = choose_x_myopic_eig(rng, post_draws, eig, sim)

            elif strategy_name.lower() == "policy":
                if policy_fn is None:
                    policy_fn = default_policy
                x_t = float(policy_fn(list(zip(x_hist, y_hist))))
                x_t = float(np.clip(x_t, 0.0, 100.0))

            else:
                raise ValueError(f"Unknown strategy: {strategy_name}")

            # Simulate y_t from true model
            mu_t = stable_mu(
                np.array([x_t]),
                theta["a"], theta["b"], theta["L"], theta["H"], theta["ell"],
                mu_floor=sim.mu_floor
            )[0]
            y_t = simulate_y(rng, mu_t, theta["c1"], theta["c2"], mu_floor=sim.mu_floor)

            # Append history
            x_hist.append(x_t)
            y_hist.append(y_t)
            idx = m0 + (t - 1)
            histories_x[r, idx] = x_t
            histories_y[r, idx] = y_t

            # Fit posterior p(theta | h_t)
            post_draws = fit_posterior_stan(
                stan_model,
                np.array(x_hist, dtype=float),
                np.array(y_hist, dtype=float),
                prior,
                mcmc,
            )
            post_draws = maybe_subsample_rows(rng, post_draws, sim.max_draws_for_metrics)

            # Metrics
            entropy[r, t] = estimate_entropy(post_draws, entropy_cfg)
            a_mean[r, t] = float(np.mean(post_draws[:, 0]))
            se_a[r, t] = (a_mean[r, t] - theta_vec[0]) ** 2

            # Tracked frames data
            tracked_run["x_hist"][idx] = x_t
            tracked_run["y_hist"][idx] = y_t
            tracked_run["draws_list"].append(
                post_draws[: min(post_draws.shape[0], sim.max_draws_for_frames)].copy()
            )

    return {
        "histories_x": histories_x,
        "histories_y": histories_y,
        "true_theta": true_theta,
        "entropy": entropy,
        "a_mean": a_mean,
        "se_a": se_a,
        "tracked_runs": np.array(tracked_runs, dtype=object),
        "tracked": tracked_runs[int(np.clip(sim.tracked_run_idx, 0, R - 1))],
    }


# =========================
# Saving
# =========================

def timestamp_str() -> str:
    # Readable and filesystem-safe
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def save_results(
    run_dir: Path,
    results_by_strategy: Dict[str, Dict],
    sim: SimConfig,
    prior: PriorConfig,
    mcmc: MCMCConfig,
    eig: EIGConfig,
    entropy_cfg: EntropyConfig,
) -> Tuple[Path, Path]:
    """
    Save into:
      ./output/{timestamp}/
        data/results.npz
        data/metadata.json
        figures/...
    """
    ensure_dir(run_dir)
    data_dir = run_dir / "data"
    ensure_dir(data_dir)

    npz_path = data_dir / "results.npz"
    json_path = data_dir / "metadata.json"

    save_dict = {
        "timestamp": str(run_dir.name),
        "config_sim": np.array([json.dumps(asdict(sim))], dtype=object),
        "config_prior": np.array([json.dumps(asdict(prior))], dtype=object),
        "config_mcmc": np.array([json.dumps(asdict(mcmc))], dtype=object),
        "config_eig": np.array([json.dumps(asdict(eig))], dtype=object),
        "config_entropy": np.array([json.dumps(asdict(entropy_cfg))], dtype=object),
    }

    for strat, res in results_by_strategy.items():
        save_dict[f"{strat}_histories_x"] = res["histories_x"]
        save_dict[f"{strat}_histories_y"] = res["histories_y"]
        save_dict[f"{strat}_true_theta"] = res["true_theta"]
        save_dict[f"{strat}_entropy"] = res["entropy"]
        save_dict[f"{strat}_a_mean"] = res["a_mean"]
        save_dict[f"{strat}_se_a"] = res["se_a"]
        save_dict[f"{strat}_tracked"] = np.array([res["tracked"]], dtype=object)
        save_dict[f"{strat}_tracked_runs"] = res["tracked_runs"]

    if npz_path.exists() or json_path.exists():
        raise RuntimeError(f"Refusing to overwrite existing results: {npz_path} or {json_path}")

    np.savez_compressed(npz_path, **save_dict)

    metadata = {
        "timestamp": str(run_dir.name),
        "R": sim.R,
        "T": sim.T,
        "tracked_run_idx": sim.tracked_run_idx,
        "prior": asdict(prior),
        "mcmc": asdict(mcmc),
        "eig": asdict(eig),
        "entropy": asdict(entropy_cfg),
        "strategies": list(results_by_strategy.keys()),
    }
    json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return npz_path, json_path


# =========================
# Main
# =========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sequential MEP design simulator with Stan backend (CmdStanPy).")
    parser.add_argument("--R", type=int, default=10)
    parser.add_argument("--T", type=int, default=20)
    parser.add_argument("--out_dir", type=str, default="output")
    parser.add_argument("--tracked_run_idx", type=int, default=0)
    parser.add_argument("--num_initial_zero", type=int, default=8)
    parser.add_argument("--num_initial_uniform", type=int, default=3)

    # MCMC
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--num_samples", type=int, default=800)
    parser.add_argument("--warmup_steps", type=int, default=800)
    parser.add_argument("--adapt_delta", type=float, default=0.9)
    parser.add_argument("--max_treedepth", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--parallel_chains", type=int, default=4)

    # EIG
    parser.add_argument("--K_grid", type=int, default=101)
    parser.add_argument("--N_outer", type=int, default=64)
    parser.add_argument("--M_inner", type=int, default=128)

    # Entropy
    parser.add_argument("--entropy_method", type=str, default="gaussian", choices=["gaussian", "knn"])
    parser.add_argument("--knn_k", type=int, default=5)

    args = parser.parse_args()

    sim = SimConfig(
        R=args.R,
        T=args.T,
        num_initial_uniform=args.num_initial_uniform,
        num_initial_zero=args.num_initial_zero,
        out_dir=args.out_dir,
        tracked_run_idx=args.tracked_run_idx,
    )
    prior = PriorConfig()
    mcmc = MCMCConfig(
        chains=args.chains,
        num_samples=args.num_samples,
        warmup_steps=args.warmup_steps,
        adapt_delta=args.adapt_delta,
        max_treedepth=args.max_treedepth,
        seed=args.seed,
        parallel_chains=args.parallel_chains,
    )
    eig = EIGConfig(K_grid=args.K_grid, N_outer=args.N_outer, M_inner=args.M_inner)
    entropy_cfg = EntropyConfig(method=args.entropy_method, knn_k=args.knn_k)

    ts = timestamp_str()
    run_dir = Path(sim.out_dir) / ts
    ensure_dir(run_dir)

    rng = np.random.default_rng(mcmc.seed)

    # Compile Stan model (cached)
    stan_cache = run_dir / "stan_cache"
    stan_model = compile_stan_model(stan_cache, mu_floor=sim.mu_floor)

    # Shared true parameters across strategies
    rng_true = np.random.default_rng(mcmc.seed + 999)
    true_theta_fixed = np.vstack([
        np.array([sample_prior_theta(rng_true, prior)[k] for k in PARAM_NAMES], dtype=float)
        for _ in range(sim.R)
    ])

    # Shared initial samples across strategies (per run)
    m_zero = int(args.num_initial_zero)
    m_unif = int(args.num_initial_uniform)
    m0 = m_zero + m_unif
    if m0 > 0:
        rng_init = np.random.default_rng(mcmc.seed + 2026)
        initial_x = np.zeros((sim.R, m0), dtype=float)
        if m_unif > 0:
            initial_x[:, m_zero:] = rng_init.uniform(0.0, 100.0, size=(sim.R, m_unif)).astype(float)

        initial_y = np.zeros((sim.R, m0), dtype=float)
        for r in range(sim.R):
            theta_vec = true_theta_fixed[r]
            a, b, L, H, ell, c1, c2 = theta_vec
            for j in range(m0):
                xj = float(initial_x[r, j])  # first m_zero are exactly 0.0
                muj = float(stable_mu(np.array([xj]), a, b, L, H, ell, mu_floor=sim.mu_floor)[0])
                initial_y[r, j] = simulate_y(rng_init, muj, float(c1), float(c2), mu_floor=sim.mu_floor)
    else:
        initial_x = None
        initial_y = None

    # Strategies
    strategies = {
        "Random": dict(policy_fn=None),
        "Myopic_EIG": dict(policy_fn=None),
        "Policy": dict(policy_fn=default_policy),  # user can replace in code
    }

    results_by_strategy: Dict[str, Dict] = {}

    for strat_name, extra in strategies.items():
        res = run_strategy(
            strategy_name=strat_name,
            rng=rng,
            stan_model=stan_model,
            sim=sim,
            prior=prior,
            mcmc=mcmc,
            eig=eig,
            entropy_cfg=entropy_cfg,
            policy_fn=extra.get("policy_fn", None),
            true_theta_fixed=true_theta_fixed,
            initial_x=initial_x,
            initial_y=initial_y,
        )
        results_by_strategy[strat_name] = res

    # Save raw results
    npz_path, json_path = save_results(run_dir, results_by_strategy, sim, prior, mcmc, eig, entropy_cfg)

    # Required plots A, B
    plots_dir = run_dir / "figures" / "plots"
    ensure_dir(plots_dir)

    entropies = {k: v["entropy"] for k, v in results_by_strategy.items()}
    se_a = {k: v["se_a"] for k, v in results_by_strategy.items()}

    plot_mean_entropy(plots_dir / "A_mean_entropy_vs_t.png", entropies, sim, prior, mcmc, eig, entropy_cfg)
    plot_rmse_a(plots_dir / "B_rmse_a_vs_t.png", se_a, sim, prior, mcmc, eig, entropy_cfg)

    # Required frames C and D (per strategy, per run)
    frames_root = run_dir / "figures" / "frames"
    ensure_dir(frames_root)

    for strat_name, res in results_by_strategy.items():
        # Frame t values: t=-m0,...,-1,0,...,T
        m0 = int(sim.num_initial_zero + sim.num_initial_uniform)
        t_values = list(range(-m0, sim.T + 1))
        tracked_runs = res.get("tracked_runs", None)
        if tracked_runs is None:
            tracked_runs = np.array([res["tracked"]], dtype=object)

        for run_idx, tracked in enumerate(tracked_runs):
            if tracked is None or tracked.get("true_theta", None) is None:
                continue

            run_dir_frames = frames_root / strat_name / f"run_{run_idx:03d}"

            # C) predictive frames
            pred_dir = run_dir_frames / "posterior_predictive"
            plot_posterior_predictive_frames(pred_dir, strat_name, tracked, sim, t_values=t_values)

            # D) posterior-a frames
            a_dir = run_dir_frames / "posterior_a"
            plot_posterior_a_frames(a_dir, strat_name, tracked, sim, t_values=t_values)

    # Print summary paths
    print(str(npz_path))
    print(str(json_path))
    print(str(plots_dir))
    print(str(frames_root))