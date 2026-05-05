import numpy as np
from scipy.special import expit
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Config
np.random.seed(2026)
n         = 3000
n0 = n1   = n // 2
lam       = 10 #0.01
MODE      = "uniform"

noise_rates = np.linspace(0, 0.48, 25)

# noise_rates = np.linspace(0.52, 0.98, 25)

# Gram matrix
def make_block(size, mode="uniform", rng=None):
    if rng is None:
        rng = np.random.default_rng()
    if mode == "uniform":
        Z = rng.uniform(0.3, 0.7, (size, size))
        B = (Z @ Z.T) / size
        np.fill_diagonal(B, 1.0)
    elif mode == "constant":
        B = np.full((size, size), 0.25)
        np.fill_diagonal(B, 1.0)
    return B

rng_feat = np.random.default_rng(2026)
K1 = make_block(n0, mode=MODE, rng=rng_feat)
K0 = make_block(n1, mode=MODE, rng=rng_feat)
K  = np.block([[K1,                np.zeros((n0, n1))],
               [np.zeros((n1, n0)), K0              ]])

y_true = np.array([0]*n0 + [1]*n1, dtype=float)
idx_c0 = np.where(y_true == 0)[0]
idx_c1 = np.where(y_true == 1)[0]

# Dual logistic regression
def objective(alpha, K, y, lam, n):
    f   = K @ alpha
    p   = expit(f)
    bce = -(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12)).mean()
    reg = 0.5 * lam * float(alpha @ f) / n
    return bce + reg

def gradient(alpha, K, y, lam, n):
    f = K @ alpha
    p = expit(f)
    return K @ (p - y + lam * alpha) / n

def train_dual(K, y_target, lam):
    n_    = len(y_target)
    grad0 = K @ (np.full(n_, 0.5) - y_target) / n_
    gtol  = max(1e-8, np.linalg.norm(grad0) / 1000)
    result = minimize(
        objective, x0=np.zeros(n_),
        args=(K, y_target, lam, n_),
        jac=gradient,
        method='L-BFGS-B',
        options={'maxiter': 10000, 'ftol': 1e-15, 'gtol': gtol}
    )
    return expit(K @ result.x)

def accuracy(p, y_true):
    return ((p >= 0.5).astype(int) == y_true.astype(int)).mean()

def compute_xi(p_T, p_S, noise_rate, mask_c1_noisy, mask_c0_clean):
    mn_T_c1 = p_T[mask_c1_noisy].mean()
    mn_S_c1 = p_S[mask_c1_noisy].mean()
    mn_T_c0 = p_T[mask_c0_clean].mean()
    mn_S_c0 = p_S[mask_c0_clean].mean()

    if noise_rate < 0.5 and mn_T_c1 < 0.5:
        xi_1 = 0.51 / mn_T_c1
    else:
        xi_1 = 1.0

    denom = mn_S_c1 - mn_T_c1
    if noise_rate < 0.5 and mn_T_c1 < 0.5:
        xi_2 = (0.51 - mn_T_c1) / denom if denom != 0.0 else 1.0
    else:
        xi_2 = 1.0
    if noise_rate > 0.5 and mn_S_c0 < mn_T_c0:
        xi_2 = (0.51 - mn_T_c1) / denom if denom != 0.0 else 1.0

    return xi_1, xi_2

### Main
acc_teacher   = []
acc_student   = []
acc_noisy_mix = []
acc_pred_avg  = []
xi1_list      = []
xi2_list      = []

rng_labels = np.random.default_rng(42)

for noise_rate in noise_rates:
    n_flip = int(noise_rate * n / 2)

    flip_idx_c0 = rng_labels.choice(idx_c0, n_flip, replace=False)
    flip_idx_c1 = rng_labels.choice(idx_c1, n_flip, replace=False)
    flip_mask   = np.zeros(n, dtype=bool)
    flip_mask[flip_idx_c0] = True
    flip_mask[flip_idx_c1] = True

    y_noisy    = y_true.copy()
    y_noisy[flip_mask] = 1.0 - y_noisy[flip_mask]
    clean_mask = ~flip_mask

    mask_c0_clean = (y_true == 0) & clean_mask
    mask_c1_noisy = (y_true == 1) & flip_mask

    p_T = train_dual(K, y_noisy, lam)
    p_S = train_dual(K, p_T,     lam)

    xi_1, xi_2 = compute_xi(p_T, p_S, noise_rate, mask_c1_noisy, mask_c0_clean)

    p_noisy_mix = (1 - xi_1) * y_noisy + xi_1 * p_T
    p_pred_avg  = (1 - xi_2) * p_T     + xi_2 * p_S

    acc_teacher.append(accuracy(p_T,           y_true))
    acc_student.append(accuracy(p_S,           y_true))
    acc_noisy_mix.append(accuracy(p_noisy_mix, y_true))
    acc_pred_avg.append(accuracy(p_pred_avg,   y_true))
    xi1_list.append(xi_1)
    xi2_list.append(xi_2)

    print(f"p={noise_rate:.2f} | T={acc_teacher[-1]:.3f}  S={acc_student[-1]:.3f}"
          f"  NM={acc_noisy_mix[-1]:.3f}  PA={acc_pred_avg[-1]:.3f}"
          f"  xi1={xi_1:.3f}  xi2={xi_2:.3f}")

acc_teacher   = np.array(acc_teacher)
acc_student   = np.array(acc_student)
acc_noisy_mix = np.array(acc_noisy_mix)
acc_pred_avg  = np.array(acc_pred_avg)
xi1_list      = np.array(xi1_list)
xi2_list      = np.array(xi2_list)

### Plot
plt.rcParams['text.usetex'] = False

colors_main = ['tab:blue', '#A0CBE8', 'tab:green', 'tab:olive']
markers     = ['o', 's', '^', 'D']
lw     = 3
ms     = 8
ftsize = 18

# Figure 1: accuracy
fig1, ax = plt.subplots(1, 1, figsize=(10, 8))

teacher_line, = ax.plot(
    noise_rates, acc_teacher,
    label=r'$y_{\mathrm{pd}}$',
    color=colors_main[0], linewidth=lw,
    linestyle='-',
    marker=markers[0], markersize=ms + 1, markevery=3
)

student_line, = ax.plot(
    noise_rates, acc_student,
    label=r'$y_{\mathrm{pd}}^{(2)}$',
    color=colors_main[1], linewidth=lw,
    linestyle='--',
    marker=markers[1], markersize=ms, markevery=3
)

noisy_mix_line, = ax.plot(
    noise_rates, acc_noisy_mix,
    label=r'$y_{\mathrm{pm}, \xi}$',
    color=colors_main[3], linewidth=lw,
    linestyle='-',
    marker=markers[2], markersize=ms + 1, markevery=3
)

pred_avg_line, = ax.plot(
    noise_rates, acc_pred_avg,
    label=r'$y_{\mathrm{pm}, \xi}^{(2)}$',
    color=colors_main[2], linewidth=lw,
    linestyle='-.',
    alpha=1,
    marker=markers[3], markersize=ms, markevery=3
)

ax.set_xlabel(r'Label error rate (%)', fontsize=ftsize + 4)
ax.set_ylabel('Classification Accuracy', fontsize=ftsize + 4)
ax.set_title(rf'Accuracy gain of mixing prediction', fontsize=ftsize + 4)
ax.tick_params(axis='both', labelsize=ftsize)
ax.grid(True, alpha=0.3)
ax.set_ylim(-0.05, 1.05)
ax.spines['left'].set_color('tab:blue')
ax.legend(loc='lower left', fontsize=ftsize)

ax.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.48])
ax.set_xticklabels(['0', '10', '20', '30', '40', '48'], fontsize=ftsize)

fig1.tight_layout()
fig1.savefig("noise_sweep_accuracy.pdf", format="pdf",
             bbox_inches='tight', edgecolor='none', transparent=True)
fig1.savefig("noise_sweep_accuracy.png", dpi=150,
             bbox_inches='tight', transparent=False)
print("Saved: noise_sweep_accuracy.pdf / .png")
plt.show()

# Figure 2: xi
fig2, ax_xi = plt.subplots(1, 1, figsize=(10, 8))

xi1_line, = ax_xi.plot(noise_rates, xi1_list, label=r'$\xi_{\mathrm{pm}}$',
                        color='tab:orange', linestyle='-', linewidth=lw, alpha=0.7,
                        marker='o', markersize=ms, markevery=3)
xi2_line, = ax_xi.plot(noise_rates, xi2_list, label=r'$\xi_{\mathrm{pm}}^{(2)}$',
                        color='tab:red', linestyle='-', linewidth=lw, alpha=0.7,
                        marker='s', markersize=ms, markevery=3)

ax_xi.set_xlabel(r'Label error rate (%)', fontsize=ftsize + 4)
ax_xi.set_ylabel(r'Mixing parameter $\xi$', fontsize=ftsize + 4)
ax_xi.set_title(r'Mixing parameters', fontsize=ftsize + 4)
ax_xi.tick_params(axis='both', labelsize=ftsize)
ax_xi.grid(True, alpha=0.3)
ax_xi.legend(loc='upper left', fontsize=ftsize)

ax_xi.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.48])
ax_xi.set_xticklabels(['0', '10', '20', '30', '40', '48'], fontsize=ftsize)

fig2.tight_layout()
fig2.savefig("noise_sweep_xi.pdf", format="pdf",
             bbox_inches='tight', edgecolor='none', transparent=True)
fig2.savefig("noise_sweep_xi.png", dpi=150,
             bbox_inches='tight', transparent=False)
plt.show()