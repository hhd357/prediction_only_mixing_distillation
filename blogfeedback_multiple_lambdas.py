### BLOG FEEDBACK loading and pre-processing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
import os
import glob

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00304/BlogFeedback.zip"
zip_path = "BlogFeedback.zip"
data_dir = "BlogFeedback"

if not os.path.exists(zip_path):
    urllib.request.urlretrieve(url, zip_path)

if not os.path.exists(data_dir):
    os.makedirs(data_dir)
    os.system(f"unzip {zip_path} -d {data_dir}")

file_path = os.path.join(data_dir, "blogData_train.csv")

df = pd.read_csv(file_path, header=None)

# Define X and y
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

print("Original shape:", X.shape)

# Seed 2026
np.random.seed(2026)

n = len(X)
indices = np.random.permutation(n)

# Randomly split into old, fresh, val, and test sets
split_old   = int(n * 0.05)
split_fresh = int(n * 0.15)
split_val   = int(n * 0.69)
split_test  = int(0.7 * n)

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

def test_mse(beta):
    return np.mean((y_test - X_test @ beta)**2)

lambda_regs    = np.logspace(np.log10(1e-2), np.log10(1e2), 50)
lambda_s_fixed = [1, 10, 1e2, 1e3]

XtX_old   = X_old.T @ X_old
XtX_fresh = X_fresh.T @ X_fresh
I_p       = np.eye(p)

R_arr     = []
results   = {ls: {"R_pd": [], "R_pm": [], "xi": []} for ls in lambda_s_fixed}

for ilam, lambda_reg in enumerate(lambda_regs):
    Omega_old = XtX_old / n_old + lambda_reg * I_p
    beta_hat  = np.linalg.solve(Omega_old, X_old.T @ y_old / n_old)
    A         = test_mse(beta_hat)
    R_arr.append(A)

    for lambda_s in lambda_s_fixed:
        Omega_fresh = XtX_fresh / n_fresh + lambda_s * I_p
        M_fresh     = np.linalg.solve(Omega_fresh, XtX_fresh / n_fresh)
        beta_pd     = M_fresh @ beta_hat

        B  = test_mse(beta_pd)
        C  = (y_test - X_test @ beta_hat) @ (y_test - X_test @ beta_pd) / n_test
        xi = (A - C) / (A + B - 2 * C)

        beta_pm = xi * beta_pd + (1 - xi) * beta_hat
        R_pm    = test_mse(beta_pm)

        results[lambda_s]["R_pd"].append(B)
        results[lambda_s]["R_pm"].append(R_pm)
        results[lambda_s]["xi"].append(xi)

### PLOT
plt.style.use('default')

fig, ax = plt.subplots(1, 1, figsize=(10, 8))

colors_main  = ['tab:blue', '#A0CBE8', 'tab:green', 'tab:olive']
colors_ls    = ['tab:orange', 'tab:green', 'tab:purple', 'tab:brown']

lw = 3
ftsize = 18

test_teacher_line, = ax.semilogx(lambda_regs, R_arr, label=r'$R$', color=colors_main[0], linewidth=lw)

all_test_lines = [test_teacher_line]

for i, lambda_s in enumerate(lambda_s_fixed):
    R_pd_fresh_arr = results[lambda_s]["R_pd"]
    R_sd_fresh_arr = results[lambda_s]["R_pm"]
    col            = colors_ls[i]

    test_pd_fresh_line, = ax.semilogx(lambda_regs, R_pd_fresh_arr,
                                       label=rf'$\lambda_s={lambda_s:.0e}$',
                                       color=col, linewidth=lw, linestyle='--')
    test_sd_fresh_line, = ax.semilogx(lambda_regs, R_sd_fresh_arr,
                                       label='_nolegend_',
                                       color=col, linewidth=lw, linestyle='-')
    all_test_lines.extend([test_pd_fresh_line, test_sd_fresh_line])

    # Find minimum of R_sd_fresh and add star
    idx_min_sd_fresh = np.argmin(R_sd_fresh_arr)
    ax.plot(lambda_regs[idx_min_sd_fresh], R_sd_fresh_arr[idx_min_sd_fresh], marker='*',
            color=col, markersize=24, markeredgecolor='white', markeredgewidth=1, zorder=5)

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
ax.set_xticks([], minor=True)
ax.spines['left'].set_color('tab:blue')
ax.grid(True, alpha=0.3)
ax.set_title('Blog Feedback', fontsize=ftsize + 4)

# Find minimum of R and add star
idx_min_R = np.argmin(R_arr)
ax.plot(lambda_regs[idx_min_R], R_arr[idx_min_R], marker='*', color=colors_main[0],
        markersize=24, markeredgecolor='white', markeredgewidth=1, zorder=5)

# Legend: color entries for each lambda_s, then linestyle entries for R_pd vs R_pm
color_handles = [test_teacher_line] + [
    plt.Line2D([0], [0], color=colors_ls[i], linewidth=lw, label=rf'$\lambda_s={lambda_s:.2f}$')
    for i, lambda_s in enumerate(lambda_s_fixed)
]
style_handles = [
    plt.Line2D([0], [0], color='black', linewidth=lw, linestyle='--', label=r'$R_{\text{pd}}$'),
    plt.Line2D([0], [0], color='black', linewidth=lw, linestyle='-',  label=r'$R_{\text{pm}}^{\star}$'),
]
ax.legend(handles=color_handles + style_handles,
          loc='upper center', bbox_to_anchor=(0.5, -0.15),
          ncol=3, fontsize=ftsize + 2)

plt.tight_layout()

plt.show()