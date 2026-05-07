import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd

Array = np.ndarray

# Parameters
n_1 = 400
n_2 = 400
p = 200
alpha = 1.0
sigma = 1.0
K = 1
xi_num = 1
num_seeds = 1
start_seed = 2026

lambda_regs = np.logspace(np.log10(1e-3), np.log10(2e2), 100)
sigma_beta_type = 'top_aligned'

gamma_l = p / n_1
gamma_u = p / n_2

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
# Sigma_1 = np.eye(p)
Sigma_1 = ar1_covariance(p)
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

def theory_fresh_full_general(
    Sigma: Array,
    beta: Array,
    sigma2: float,
    gamma_l: float,
    gamma_u: float,
    lambdas: Array,
) -> pd.DataFrame:
    """Compute the full fresh-X oracle deterministic equivalents in the anisotropic setting.

    Returns the asymptotic predictions for
    ``R``, ``C^fresh``, ``D^fresh``, ``R_pd^fresh``, ``xi^*``, and ``R_pm^*``.
    Also returns bar_R, bar_Delta, bar_D as theory analogues of empirical A, C, B.
    """
    eigs, U = np.linalg.eigh(Sigma)
    beta_e = U.T @ beta

    rows = []
    for lam in np.asarray(lambdas, dtype=float):
        kappa = solve_kappa_bisect(eigs, gamma_l, float(lam))
        kappa_u = solve_kappa_bisect(eigs, gamma_u, float(lam))

        g = 1.0 / (eigs + kappa)
        g_u = 1.0 / (eigs + kappa_u)

        t2 = gamma_l * np.mean((eigs ** 2) * (g ** 2))
        b = 1.0 / (1.0 - t2)
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

        bar_R = (kappa ** 2) * b * q20 + sigma2 * u2
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
        bar_C = bar_R - bar_Delta
        bar_Rpd = bar_R - 2.0 * bar_Delta + bar_D

        xi = bar_Delta / bar_D
        R_te = sigma2 + bar_R
        C_fresh = sigma2 + bar_C
        R_pd = sigma2 + bar_Rpd
        R_pm = R_te - (R_te - C_fresh)**2/(R_te + R_pd - 2*C_fresh)

        rows.append({
            'lambda': lam,
            'kappa_l': kappa,
            'kappa_u': kappa_u,
            'b_l': b,
            'b_u': b_u,
            'theory_R': R_te,
            'theory_C': C_fresh,
            'theory_D': bar_D,
            'theory_Rpd': R_pd,
            'theory_xi_star': xi,
            'theory_Rpm_star': R_pm,
            'theory_Delta': bar_Delta,
            'theory_A': bar_R,
            'theory_B': bar_Rpd,
            'theory_C_cross': bar_C,
        })
    return pd.DataFrame(rows)


# Fix beta_star across seeds
np.random.seed(start_seed - 1)
eigvals, V = np.linalg.eigh(Sigma_1)
Sigma_beta = generate_sigma_beta(V, p, sigma_beta_type)
cov_beta = (alpha**2 / p) * Sigma_beta
beta_star = np.random.multivariate_normal(np.zeros(p), cov_beta)

# Setup
collected_risk_teacher = []
collected_risk_pd = []
collected_risk_pm = []
collected_xi_pm = []
collected_A = []
collected_B = []
collected_C = []

for current_seed in range(start_seed, start_seed + num_seeds):
    print(f"Processing Seed: {current_seed}")
    np.random.seed(current_seed)

    # Teacher training data
    X_train = np.random.multivariate_normal(np.zeros(p), Sigma_1, size=n_1)
    noise = np.random.randn(n_1) * sigma
    y_train = X_train @ beta_star + noise

    # Fresh unlabeled
    X_fresh = np.random.multivariate_normal(np.zeros(p), Sigma_2, size=n_2)
    y_fresh = X_fresh @ beta_star + np.random.randn(n_2) * sigma

    I_p = np.eye(p)
    XtX = X_train.T @ X_train
    S_train = XtX / n_1
    score_train = X_train.T @ y_train / n_1
    S_fresh = X_fresh.T @ X_fresh / n_2

    # Temporary lists
    seed_r_teacher, seed_r_pd, seed_r_pm = [], [], []
    seed_xi_pm = []
    seed_A, seed_B, seed_C = [], [], []

    for lambda_reg in lambda_regs:
        # Teacher
        Omega = S_train + lambda_reg * I_p
        beta_hat = np.linalg.solve(Omega, X_train.T @ y_train) / n_1

        Omega_fresh = S_fresh + lambda_reg * I_p

        risk_teacher = excess_risk(beta_hat, beta_star, Sigma_1, sigma)
        seed_r_teacher.append(risk_teacher)

        # PD student
        M_fresh = np.linalg.solve(Omega_fresh, S_fresh)
        beta_pd = M_fresh @ beta_hat
        seed_r_pd.append(excess_risk(beta_pd, beta_star, Sigma_1, sigma))

        # Optimal PM student
        error = beta_hat - beta_star
        tilde_error = beta_pd - beta_star
        A = error @ Sigma_1 @ error
        B = tilde_error @ Sigma_1 @ tilde_error
        C = error @ Sigma_1 @ tilde_error

        seed_A.append(A)
        seed_B.append(B)
        seed_C.append(C)

        xi = (A - C) / (A + B - 2 * C)
        seed_xi_pm.append(xi)

        beta_pm = (1 - xi) * beta_hat + xi * beta_pd
        seed_r_pm.append(excess_risk(beta_pm, beta_star, Sigma_1, sigma))

    collected_risk_teacher.append(seed_r_teacher)
    collected_risk_pd.append(seed_r_pd)
    collected_risk_pm.append(seed_r_pm)
    collected_xi_pm.append(seed_xi_pm)
    collected_A.append(seed_A)
    collected_B.append(seed_B)
    collected_C.append(seed_C)


# Averaging results
avg_risk_teacher = np.mean(collected_risk_teacher, axis=0)
avg_risk_pd = np.mean(collected_risk_pd, axis=0)
avg_risk_pm = np.mean(collected_risk_pm, axis=0)
avg_xi_pm = np.mean(collected_xi_pm, axis=0)
avg_A = np.mean(collected_A, axis=0)
avg_B = np.mean(collected_B, axis=0)
avg_C = np.mean(collected_C, axis=0)

# Theory curves
theory_df = theory_fresh_full_general(
    Sigma=Sigma_1,
    beta=beta_star,
    sigma2=sigma**2,
    gamma_l=gamma_l,
    gamma_u=gamma_u,
    lambdas=lambda_regs,
)

theory_risk_teacher = theory_df['theory_R'].values
theory_risk_pd      = theory_df['theory_Rpd'].values
theory_risk_pm      = theory_df['theory_Rpm_star'].values
theory_xi_pm        = theory_df['theory_xi_star'].values
theory_A            = theory_df['theory_A'].values
theory_B            = theory_df['theory_B'].values
theory_C            = theory_df['theory_C_cross'].values

plt.style.use('default')

fig, ax = plt.subplots(1, 1, figsize=(10, 8))

colors_main = ['tab:blue', '#A0CBE8', 'tab:green', 'tab:olive']

lw = 3
ftsize = 18

# Risk curves
test_teacher_line, = ax.semilogx(lambda_regs, theory_risk_teacher,
                                  label=r'$\mathcal{R}$',
                                  color=colors_main[0], linewidth=lw)
test_pd_line, = ax.semilogx(lambda_regs, theory_risk_pd,
                              label=r'$\mathcal{R}_{\mathrm{pd}}$',
                              color=colors_main[1], linewidth=lw)
test_pm_line, = ax.semilogx(lambda_regs, theory_risk_pm,
                              label=r'$\mathcal{R}_{\mathrm{pm}}^{\star}$',
                              color=colors_main[3], linewidth=lw)

# Axis labels
ax.set_xlabel(r'Ridge penalty $\lambda$', fontsize=ftsize + 4)
ax.set_ylabel('Squared prediction risk', fontsize=ftsize + 4)
ax.set_yscale('log')

ax.tick_params(axis='y', labelsize=ftsize)
ax.tick_params(axis='x', labelsize=ftsize)
ax.spines['left'].set_color('tab:blue')
ax.grid(True, alpha=0.3)

ax.set_title(r'Asymptotic risks', fontsize=ftsize + 4)

for arr, color in [(theory_risk_teacher, colors_main[0]),
                   (theory_risk_pd,      colors_main[1]),
                   (theory_risk_pm,      colors_main[3])]:
    idx = np.argmin(arr)
    ax.plot(lambda_regs[idx], arr[idx],
            marker='*', color=color, markersize=24,
            markeredgecolor='white', markeredgewidth=1, zorder=5)

# xi
ax_twin = ax.twinx()

xi_line, = ax_twin.semilogx(lambda_regs, theory_xi_pm,
                              label=r'$\xi^{\star}_{\mathrm{pm}}$',
                              color='tab:red', linestyle='-', linewidth=lw)

ax_twin.set_ylabel(r'Optimal mixing parameter $\xi^{\star}$',
                   fontsize=ftsize + 4, color='tab:red')
ax_twin.tick_params(axis='y', labelsize=ftsize, labelcolor='tab:red')
ax_twin.spines['right'].set_color('tab:red')
ax_twin.spines['top'].set_visible(False)
ax_twin.spines['bottom'].set_visible(False)
ax_twin.spines['left'].set_visible(False)

ax.set_xticks([])
ax.set_xticks([], minor=True)
ticks = [0.001, 0.01, 0.1, 1, 10, 100]
ax.set_xticks(ticks)
ax.set_xticklabels(['0.001','0.01', '0.1', '1', '10', '100'])

ax.set_yticks([])
ax.set_yticks([], minor=True)
ticks_y = [1.4, 1.6, 1.8, 2, 2.2, 2.4]
ax.set_yticks(ticks_y)
ax.set_yticklabels(['1.4', '1.6', '1.8', '2', '2.2', '2.4'])

# --- Legend ---
all_lines = [test_teacher_line, test_pd_line, test_pm_line, xi_line]
all_labels = [l.get_label() for l in all_lines]
ax.legend(all_lines, all_labels,
          loc='upper center', bbox_to_anchor=(0.5, -0.15),
          ncol=4, fontsize=ftsize + 2)

plt.tight_layout()

plt.savefig(
    "fresh_x_theory.pdf",
    format="pdf",
    bbox_inches='tight',
    edgecolor='none',
    transparent=True
)

plt.show()