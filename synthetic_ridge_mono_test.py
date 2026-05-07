import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd

Array = np.ndarray

# Parameters
n_1 = 800
p = 400
alpha = 1.0
sigma = 1.0
K = 1
xi_num = 1
num_seeds = 30
start_seed = 2026

lambda_reg = 0.2
gamma_l = p / n_1

gamma_u_vals = np.logspace(np.log10(0.1), np.log10(6.0), 40)
n_2_vals = (p / gamma_u_vals).astype(int)

sigma_beta_type = 'identity'

def ar1_covariance(p, rho=0.25):
    Sigma = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            Sigma[i, j] = rho ** abs(i - j)
    return Sigma

def spiked_covariance(p, num_spikes=1, spike_strength=5.0):
    Sigma = np.eye(p)
    for _ in range(num_spikes):
        v = np.random.randn(p)
        v /= np.linalg.norm(v)
        Sigma += spike_strength * np.outer(v, v)
    return Sigma

def random_wishart(p, m):
    X = np.random.randn(m, p)
    return X.T @ X / m

# Data covariance
Sigma_1 = np.eye(p)
# Sigma_1 = ar1_covariance(p)
Sigma_2 = Sigma_1
# Sigma_2 = 5 * Sigma_1 + 10*np.eye(p)

# For "top_aligned" or "bottom_aligned": use spike_frac and align_frac below
spike_frac = 0.1              # Sparse level: fraction of dimensions for top/bottom (e.g., 0.1 -> 10% of p)
align_frac = 0.9              # Alignment factor: fraction of total variance in aligned subspace (e.g., 0.9 -> 90% in top/bottom)

def generate_sigma_beta(V, p, sigma_beta_type, k_top=50, k_bottom=50, m_amp=20, amp_factor=10.0):
    if sigma_beta_type == "top50":
        D = np.ones(p) * 0.05
        D[-k_top:] = 1.0
        Sigma_beta = V @ np.diag(D) @ V.T
        trace = np.trace(Sigma_beta)
        Sigma_beta = (p / trace) * Sigma_beta
    elif sigma_beta_type == "bottom50":
        D = np.ones(p) * 0.05
        D[:k_bottom] = 1.0
        Sigma_beta = V @ np.diag(D) @ V.T
        trace = np.trace(Sigma_beta)
        Sigma_beta = (p / trace) * Sigma_beta
    elif sigma_beta_type == "mixed20":
        D = np.ones(p)
        D[:m_amp] *= amp_factor
        Sigma_beta = np.diag(D)
        trace = np.trace(Sigma_beta)
        Sigma_beta = (p / trace) * Sigma_beta
    elif sigma_beta_type == "identity":
        Sigma_beta = np.eye(p)
    elif sigma_beta_type == "top_aligned":
        k = int(spike_frac * p)
        D = np.zeros(p)
        D[-k:] = align_frac * p / k      # Top k: equal share of aligned energy
        D[:-k] = (1 - align_frac) * p / (p - k)  # Bottom: equal share of unaligned
        Sigma_beta = V @ np.diag(D) @ V.T
    elif sigma_beta_type == "bottom_aligned":
        k = int(spike_frac * p)
        D = np.zeros(p)
        D[:k] = align_frac * p / k        # Bottom k: equal share of aligned energy
        D[k:] = (1 - align_frac) * p / (p - k)  # Top: equal share of unaligned
        Sigma_beta = V @ np.diag(D) @ V.T
    else:
        raise ValueError(f"Invalid sigma_beta_type: {sigma_beta_type}")
    return Sigma_beta

def excess_risk(beta_hat, beta_star, Sigma, sigma):
    diff = beta_hat - beta_star
    return diff.T @ Sigma @ diff + sigma**2


# Theory functions

def solve_kappa_bisect(
    eigs: Array,
    gamma: float,
    lam: float,
    tol: float = 1e-12,
    maxit: int = 300,
) -> float:
    s = np.asarray(eigs, dtype=float)

    def f(k: float) -> float:
        return k - lam - gamma * k * np.mean(s / (s + k))

    lo = 0.0
    hi = max(lam + gamma * float(s.max()) + 1.0, 1.0)
    while f(hi) <= 0.0:
        hi *= 2.0
        if hi > 1e12:
            raise RuntimeError("Failed to bracket the kappa fixed point.")

    for _ in range(maxit):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0.0:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol * max(1.0, hi):
            break
    return 0.5 * (lo + hi)


def theory_fresh_pm(
    Sigma: Array,
    beta: Array,
    sigma2: float,
    gamma_l: float,
    gamma_u: float,
    lam: float,
) -> float:
    """Compute theoretical R_pm^* for a single (gamma_u, lambda) pair."""
    eigs, U = np.linalg.eigh(Sigma)
    beta_e = U.T @ beta

    kappa   = solve_kappa_bisect(eigs, gamma_l, lam)
    kappa_u = solve_kappa_bisect(eigs, gamma_u, lam)

    g   = 1.0 / (eigs + kappa)
    g_u = 1.0 / (eigs + kappa_u)

    t2 = gamma_l * np.mean((eigs ** 2) * (g ** 2))
    b  = 1.0 / (1.0 - t2)
    u2 = t2 * b

    t2u_unlab = gamma_u * np.mean((eigs ** 2) * (g_u ** 2))
    b_u = 1.0 / (1.0 - t2u_unlab)

    q20 = np.sum((beta_e ** 2) * eigs * (g ** 2))
    q11 = np.sum((beta_e ** 2) * eigs * g * g_u)
    q21 = np.sum((beta_e ** 2) * eigs * (g ** 2) * g_u)
    q12 = np.sum((beta_e ** 2) * eigs * g * (g_u ** 2))
    q22 = np.sum((beta_e ** 2) * eigs * (g ** 2) * (g_u ** 2))
    q02 = np.sum((beta_e ** 2) * eigs * (g_u ** 2))

    t21 = gamma_l * np.mean((eigs ** 2) * (g ** 2) * g_u)
    t22 = gamma_l * np.mean((eigs ** 2) * (g ** 2) * (g_u ** 2))

    bar_R     = (kappa ** 2) * b * q20 + sigma2 * u2
    bar_Delta = kappa_u * (
        -kappa * q11
        + (kappa ** 2) * q21
        + b * t21 * ((kappa ** 2) * q20 + sigma2)
    )
    bar_D = (kappa_u ** 2) * b_u * (
        q02
        - 2.0 * kappa * q12
        + (kappa ** 2) * q22
        + b * t22 * ((kappa ** 2) * q20 + sigma2)
    )
    bar_C    = bar_R - bar_Delta
    bar_Rpd  = bar_R - 2.0 * bar_Delta + bar_D

    R_te    = sigma2 + bar_R
    C_fresh = sigma2 + bar_C
    R_pd    = sigma2 + bar_Rpd
    R_pm    = R_te - (R_te - C_fresh) ** 2 / (R_te + R_pd - 2 * C_fresh)

    return R_pm


# Fix beta_star across seeds
np.random.seed(start_seed - 1)
eigvals, V = np.linalg.eigh(Sigma_1)
Sigma_beta = generate_sigma_beta(V, p, sigma_beta_type)
cov_beta   = (alpha**2 / p) * Sigma_beta
beta_star  = np.random.multivariate_normal(np.zeros(p), cov_beta)

# Theory curve (over gamma_u)
print("Computing theory curve...")
theory_rpm = []
for gamma_u in gamma_u_vals:
    val = theory_fresh_pm(Sigma_1, beta_star, sigma**2, gamma_l, gamma_u, lambda_reg)
    theory_rpm.append(val)
theory_rpm = np.array(theory_rpm)

# Empirical runs (over gamma_u, averaged over seeds)
print("Running empirical simulation...")
I_p = np.eye(p)

# Precompute teacher quantities (same for all gamma_u)
seed_teacher_betas = []
for current_seed in range(start_seed, start_seed + num_seeds):
    np.random.seed(current_seed)
    X_train = np.random.multivariate_normal(np.zeros(p), Sigma_1, size=n_1)
    noise   = np.random.randn(n_1) * sigma
    y_train = X_train @ beta_star + noise

    S_train  = X_train.T @ X_train / n_1
    Omega    = S_train + lambda_reg * I_p
    beta_hat = np.linalg.solve(Omega, X_train.T @ y_train) / n_1
    seed_teacher_betas.append((X_train, y_train, beta_hat))

emp_rpm      = np.zeros(len(gamma_u_vals))
emp_rpm_sems = np.zeros(len(gamma_u_vals))   # SEM across seeds

for idx, (gamma_u, n_2) in enumerate(zip(gamma_u_vals, n_2_vals)):
    n_2 = max(n_2, 1)  # safety
    rpm_seeds = []

    for seed_idx, current_seed in enumerate(range(start_seed, start_seed + num_seeds)):
        np.random.seed(current_seed + 1000 * (idx + 1))  # different fresh data per gamma_u

        X_train, y_train, beta_hat = seed_teacher_betas[seed_idx]

        # Fresh unlabeled data for this gamma_u
        X_fresh = np.random.multivariate_normal(np.zeros(p), Sigma_2, size=n_2)

        S_fresh     = X_fresh.T @ X_fresh / n_2
        Omega_fresh = S_fresh + lambda_reg * I_p

        # PD student
        M_fresh = np.linalg.solve(Omega_fresh, S_fresh)
        beta_pd = M_fresh @ beta_hat

        # Optimal PM mixing
        error       = beta_hat - beta_star
        tilde_error = beta_pd  - beta_star
        A = error       @ Sigma_1 @ error
        B = tilde_error @ Sigma_1 @ tilde_error
        C = error       @ Sigma_1 @ tilde_error

        xi      = (A - C) / (A + B - 2 * C)
        beta_pm = (1 - xi) * beta_hat + xi * beta_pd
        rpm_seeds.append(excess_risk(beta_pm, beta_star, Sigma_1, sigma))

    emp_rpm[idx]      = np.mean(rpm_seeds)
    emp_rpm_sems[idx] = np.std(rpm_seeds) / np.sqrt(num_seeds)   # SEM
    if (idx + 1) % 10 == 0:
        print(f"  gamma_u progress: {idx+1}/{len(gamma_u_vals)}")