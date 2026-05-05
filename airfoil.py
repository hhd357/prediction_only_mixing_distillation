### AIRFOIL loading and pre-processing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
import os
import glob
import scipy.io

from ucimlrepo import fetch_ucirepo

airfoil_self_noise = fetch_ucirepo(id=291)

df = airfoil_self_noise.data.original

# Define X covariates columns
x_columns = [
    'frequency',
    'attack-angle',
    'chord-length',
    'free-stream-velocity',
    'suction-side-displacement-thickness'
]

# Define y target column
y_column = 'scaled-sound-pressure'

# Extract X and y
X = df[x_columns]
y = df[y_column]

print("Original shape:", X.shape)

np.random.seed(2026)

n = len(X)
indices = np.random.permutation(n)

# Randomly split into old, fresh, val, and test sets
split_old   = int(n * 0.1)
split_fresh = int(n * 0.2)
split_val   = int(n * 0.65)
split_test  = int(n * 0.7)

old_idx   = indices[:split_old]
fresh_idx = indices[split_old:split_fresh]
val_idx   = indices[split_val:split_test]
test_idx  = indices[split_test:]

X_old   = X.iloc[old_idx]
X_fresh = X.iloc[fresh_idx]
X_val   = X.iloc[val_idx]
X_test  = X.iloc[test_idx]

y_old   = y.iloc[old_idx]
y_fresh = y.iloc[fresh_idx]
y_val   = y.iloc[val_idx]
y_test  = y.iloc[test_idx]

# Cleaning data
old_mask = X_old.notna().all(axis=1) & y_old.notna()
X_old   = X_old[old_mask]
y_old   = y_old[old_mask]

fresh_mask = X_fresh.notna().all(axis=1) & y_fresh.notna()
X_fresh = X_fresh[fresh_mask]
y_fresh = y_fresh[fresh_mask]

val_mask = X_val.notna().all(axis=1) & y_val.notna()
X_val   = X_val[val_mask]
y_val   = y_val[val_mask]

test_mask = X_test.notna().all(axis=1) & y_test.notna()
X_test  = X_test[test_mask]
y_test  = y_test[test_mask]

# --- STANDARDIZATION (USING TRAIN SET) ---
# Compute means and stds on the training set 
X_train_mean = X_old.mean(axis=0)
X_train_std  = X_old.std(axis=0)

y_train_mean = y_old.mean()
y_train_std  = y_old.std()

# Handle zero variance to avoid division by zero
X_train_std = X_train_std.replace(0, 1.0)
if y_train_std == 0:
    y_train_std = 1.0

# Standardize X across all sets using training set info
X_old   = (X_old   - X_train_mean) / X_train_std
X_fresh = (X_fresh - X_train_mean) / X_train_std
X_val   = (X_val   - X_train_mean) / X_train_std
X_test  = (X_test  - X_train_mean) / X_train_std

# Standardize y across all sets using training set info
y_old   = (y_old   - y_train_mean) / y_train_std
y_fresh = (y_fresh - y_train_mean) / y_train_std
y_val   = (y_val   - y_train_mean) / y_train_std
y_test  = (y_test  - y_train_mean) / y_train_std

n_old   = len(X_old)
n_fresh = len(X_fresh)
n_val   = len(X_val)
n_test  = len(X_test)
p = X_old.shape[1]

print("Number of covariates:", p)
print(f"Number of valid records before split: {len(X)}")
print(f"n_old:   {n_old}")
print(f"n_fresh: {n_fresh}")
print(f"n_val:   {n_val}")
print(f"n_test:  {n_test}")

### MAIN COMPUTATION CODE
p = X_old.shape[1]

I_p = np.eye(p)

# Test error measured on test set
def test_mse(beta):
    return np.mean((y_test - X_test @ beta)**2)

# Initialize arrays
A_same_arr = []
B_same_arr = []
C_same_arr = []
xi_same_arr = []

A_fresh_arr = []
B_fresh_arr = []
C_fresh_arr = []
xi_fresh_arr = []
xi_tune_arr = []

Atune_arr = []
Btune_arr = []
Ctune_arr = []

R_arr = []
R_pd_same_arr = []
R_pd_fresh_arr = []
R_sd_fresh_arr = []
R_sd_same_arr = []
R_sd_fresh_tune_arr = []

# Regularized parameter range
lambda_regs = np.logspace(np.log10(1e-5), np.log10(100), 100)

num_lams = len(lambda_regs)

XtX_old = X_old.T @ X_old
XtX_fresh = X_fresh.T @ X_fresh
XtX_val = X_val.T @ X_val
I_p = np.eye(p)

for ilam, lambda_reg in enumerate(lambda_regs):
    # Teacher ridge
    Omega_old = XtX_old / n_old + lambda_reg * I_p
    beta_hat = np.linalg.solve(Omega_old, X_old.T @ y_old / n_old)

    # Fresh-X PD student
    Omega_fresh = XtX_fresh / n_fresh + lambda_reg * I_p
    M_fresh = np.linalg.solve(Omega_fresh, XtX_fresh / n_fresh)
    beta_pd_fresh = M_fresh @ beta_hat

    # Fresh-X PM student
    A_fresh = test_mse(beta_hat)
    A_fresh_arr.append(A_fresh)

    B_fresh = test_mse(beta_pd_fresh)
    B_fresh_arr.append(B_fresh)

    C_fresh = ((y_test - X_test @ beta_hat).T @ (y_test - X_test @ beta_pd_fresh)) / n_test
    C_fresh_arr.append(C_fresh)

    xi_fresh = (A_fresh - C_fresh)/(A_fresh + B_fresh - 2*C_fresh)
    xi_fresh_arr.append(xi_fresh)

    beta_sd_fresh = xi_fresh * beta_pd_fresh + (1 - xi_fresh) * beta_hat
    R_sd_fresh = test_mse(beta_sd_fresh)

    # Estimated xi and risk
    Atune = (1/n_val) * np.linalg.norm(y_val - X_val @ beta_hat)**2
    Atune_arr.append(Atune)

    Btune = (1/n_val) * np.linalg.norm(y_val - X_val @ beta_pd_fresh)**2
    Btune_arr.append(Btune)

    Ctune = ((y_val - X_val @ beta_hat) @ (y_val - X_val @ beta_pd_fresh)) / n_val
    Ctune_arr.append(Ctune)

    xi_tune = (Atune - Ctune)/(Atune + Btune - 2*Ctune)
    xi_tune_arr.append(xi_tune)

    beta_sd_tune = xi_tune * beta_pd_fresh + (1 - xi_tune) * beta_hat

    R = test_mse(beta_hat)
    R_pd_fresh = test_mse(beta_pd_fresh)
    R_sd_tune = test_mse(beta_sd_tune)
    R_arr.append(R)
    R_pd_fresh_arr.append(R_pd_fresh)
    R_sd_fresh_arr.append(R_sd_fresh)
    R_sd_fresh_tune_arr.append(R_sd_tune)

##### PLOT
plt.style.use('default')

fig, ax = plt.subplots(1, 1, figsize=(10, 8))

colors_main = ['tab:blue', '#A0CBE8', 'tab:green', 'tab:olive']

lw = 3
ftsize = 18

# Test risk curves
test_teacher_line, = ax.semilogx(lambda_regs, R_arr, label=r'$R$', color=colors_main[0], linewidth=lw)
test_pd_fresh_line, = ax.semilogx(lambda_regs, R_pd_fresh_arr, label=r'$R_{\text{pd}}$', color=colors_main[1], linewidth=lw)
test_sd_fresh_line, = ax.semilogx(lambda_regs, R_sd_fresh_arr, label=r'$R_{\text{pm}}^{\star}$', color=colors_main[3], linewidth=lw)
test_sd_tune_line, = ax.semilogx(lambda_regs, R_sd_fresh_tune_arr, label=r'$\widehat{R}_{\text{pm}}^{\star}$', color=colors_main[3],
                             linestyle='-.', marker='o', markersize=6, linewidth=lw, alpha=0.7, markevery=7)

ax.set_xlabel(r'Ridge penalty $\lambda$', fontsize=ftsize + 4)
ax.set_ylabel('Squared prediction risk', fontsize=ftsize + 4)
ax.set_yscale('log')

ax.set_yticks([])
ax.set_yticks([], minor=True)
ticks = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2]
ax.set_yticks(ticks)
ax.set_yticklabels(['0.3','0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1', '1.1', '1.2'], fontsize=ftsize)

ax.tick_params(axis='y', labelsize=ftsize)
ax.tick_params(axis='x', labelsize=ftsize) 
ax.spines['left'].set_color('tab:blue')
ax.grid(True, alpha=0.3)

# Title
ax.set_title('Airfoil', fontsize=ftsize + 4)

# Secondary y-axis for xi
ax_twin = ax.twinx()

test_xiemp_line, = ax_twin.semilogx(lambda_regs, xi_fresh_arr, label=r'$\xi^{\star}_{\text{pm}}$', color='tab:red', linestyle='-', \
                                    linewidth=lw)
ax_twin.set_ylabel(r'Optimal mixing parameter', fontsize=ftsize + 4, color='tab:red')
ax_twin.tick_params(axis='y', labelsize=ftsize, labelcolor='tab:red')
ax_twin.spines['right'].set_color('tab:red')

ax_twin.set_xticks([])
ax_twin.set_xticks([], minor=True)
ticks = [1e-5, 1e-4, 0.001, 0.01, 0.1, 1, 10, 100]
ax_twin.set_xticks(ticks)

# Find minimum of R_pd_fresh and add star
idx_min_pd_fresh = np.argmin(R_pd_fresh_arr)
lambda_min_pd_fresh = lambda_regs[idx_min_pd_fresh]
R_pd_fresh_min = R_pd_fresh_arr[idx_min_pd_fresh]
ax.plot(lambda_min_pd_fresh, R_pd_fresh_min, marker='*', color=colors_main[1], markersize=24, markeredgecolor='white', \
        markeredgewidth=1, zorder=5)

# Find minimum of R_sd_fresh and add star
idx_min_sd_fresh = np.argmin(R_sd_fresh_arr)
lambda_min_sd_fresh = lambda_regs[idx_min_sd_fresh]
R_sd_fresh_min = R_sd_fresh_arr[idx_min_sd_fresh]
ax.plot(lambda_min_sd_fresh, R_sd_fresh_min, marker='*', color=colors_main[3], markersize=24, markeredgecolor='white', \
        markeredgewidth=1, zorder=5)

# Find minimum of R and add star
idx_min_R = np.argmin(R_arr)
lambda_min_R = lambda_regs[idx_min_R]
R_min = R_arr[idx_min_R]
ax.plot(lambda_min_R, R_min, marker='*', color=colors_main[0], markersize=24, markeredgecolor='white', markeredgewidth=1, zorder=5)

# Legend box
all_test_lines = [test_teacher_line, test_pd_fresh_line, test_sd_fresh_line, test_sd_tune_line, test_xiemp_line]
all_test_labels = [l.get_label() for l in all_test_lines]
ax.legend(all_test_lines, all_test_labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5, fontsize=ftsize + 2)

ax_twin.spines['top'].set_visible(False)
ax_twin.spines['bottom'].set_visible(False)
ax_twin.spines['left'].set_visible(False)

plt.tight_layout()
plt.show()
