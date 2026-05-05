# ResNet-18 features pretrained on ImageNet, with CIFAR-10 dataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision
from torchvision import transforms
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load CIFAR-10 dataset
transform_base = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
])

# Load full train and test sets
train_dataset_full = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_base)
test_dataset_full = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_base)

# Load pretrained ResNet-18
model = torchvision.models.resnet18(pretrained=True)
model.fc = nn.Identity()  # Remove final FC layer, output 512-dim features
model = model.to(device)
model.eval()

### SUBSAMPLING THE DATASET
K = 10
n_train = 2000
n_fresh = 4000
n_val = 200
n_test = 2000
p = 512

noise_rate = 0.0  # fraction of training labels to corrupt

np.random.seed(2026)

# Subsample — train, fresh, val all disjoint, drawn from train_dataset_full
all_train_indices = np.random.choice(len(train_dataset_full), n_train + n_fresh + n_val, replace=False)
train_indices = all_train_indices[:n_train]
fresh_indices = all_train_indices[n_train:n_train + n_fresh]
val_indices   = all_train_indices[n_train + n_fresh:]
test_indices  = np.random.choice(len(test_dataset_full), n_test, replace=False)

train_dataset = Subset(train_dataset_full, train_indices)
fresh_dataset = Subset(train_dataset_full, fresh_indices)
val_dataset   = Subset(train_dataset_full, val_indices)
test_dataset  = Subset(test_dataset_full,  test_indices)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
fresh_loader = DataLoader(fresh_dataset, batch_size=32, shuffle=False)
val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False)

# Extract features
def extract_features(loader):
    features = []
    labels = []
    with torch.no_grad():
        for images, lbls in loader:
            images = images.to(device)
            feats = model(images)
            features.append(feats.cpu().numpy())
            labels.append(lbls.numpy())
    return np.vstack(features), np.hstack(labels)

# Extract features
print("Extracting train features...")
X_train, y_train = extract_features(train_loader)
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

print("Extracting fresh features...")
X_fresh, y_fresh = extract_features(fresh_loader)
print(f"X_fresh shape: {X_fresh.shape}, y_fresh shape: {y_fresh.shape}")

print("Extracting val features...")
X_val, y_val = extract_features(val_loader)
print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")

print("Extracting test features...")
X_test, y_test = extract_features(test_loader)
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# Apply label noise to training labels only
y_train_noisy = y_train.copy()
n_noisy = int(noise_rate * n_train)
noisy_indices = np.random.choice(n_train, n_noisy, replace=False)
for idx in noisy_indices:
    original_label = y_train_noisy[idx]
    wrong_labels = np.delete(np.arange(K), original_label)
    y_train_noisy[idx] = np.random.choice(wrong_labels)

print(f"Label noise: {n_noisy}/{n_train} training labels corrupted ({noise_rate*100:.1f}%)")
print(f"Actual noise rate: {np.mean(y_train_noisy != y_train)*100:.1f}%")

# One-hot encode
encoder = OneHotEncoder(sparse_output=False)
y_train_multi = encoder.fit_transform(y_train_noisy.reshape(-1, 1))  
y_fresh_multi = encoder.transform(y_fresh.reshape(-1, 1))           
y_val_multi   = encoder.transform(y_val.reshape(-1, 1))              
y_test_multi  = encoder.transform(y_test.reshape(-1, 1))           

### MAIN COMPUTATION CODE
lambda_regs = np.logspace(np.log10(1e-3), np.log10(1e2), 100)
num_lams = len(lambda_regs)

# Test error measured on test set
def test_mse(beta_all):
    return (1/n_test) * sum(np.linalg.norm(y_test_multi[:, i] - X_test @ beta_all[i])**2 for i in range(K))

# Initialize arrays
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

def classification_accuracy(beta_all):
    """Compute accuracy: argmax of predictions vs true labels."""
    beta_stack = np.array(beta_all).T  # (p, K) = (512, 10)
    y_pred_scores = X_test @ beta_stack  # (n_test, K)
    y_pred_classes = np.argmax(y_pred_scores, axis=1)
    acc = np.mean(y_pred_classes == y_test)
    return acc

for ilam, lambda_reg in enumerate(lambda_regs):
    beta_hat_all = []
    beta_pd_all = []
    beta_sd_all = []
    beta_sd_tune_all = []

    Omega = X_train.T @ X_train / n_train + lambda_reg * np.eye(p)
    Omega_fresh = X_fresh.T @ X_fresh / n_fresh + lambda_reg * np.eye(p)
    M = np.linalg.solve(Omega, X_train.T @ X_train / n_train)
    M_fresh = np.linalg.solve(Omega_fresh, X_fresh.T @ X_fresh / n_fresh)

    for i in range(K):
        y_train_i = y_train_multi[:, i]
        beta_hat_i = np.linalg.solve(Omega, X_train.T @ y_train_i) / n_train
        beta_hat_all.append(beta_hat_i)
        beta_pd_i = M_fresh @ beta_hat_i
        beta_pd_all.append(beta_pd_i)

    #A, B, C, xi_emp calculation using test samples
    A_emp = (1/n_test) * sum(np.linalg.norm(y_test_multi[:, i] - X_test @ beta_hat_all[i])**2 for i in range(K))
    B_emp = (1/n_test) * sum(np.linalg.norm(y_test_multi[:, i] - X_test @ beta_pd_all[i])**2 for i in range(K))
    C_emp = (1/n_test) * sum((y_test_multi[:, i] - X_test @ beta_hat_all[i]).T @ (y_test_multi[:, i] - X_test @ beta_pd_all[i]) for i in range(K))
    xi_emp = (A_emp - C_emp)/(A_emp + B_emp - 2*C_emp)


    #Tuning
    Atune = (1/n_val) * sum(np.linalg.norm(y_val_multi[:, i] - X_val @ beta_hat_all[i])**2 for i in range(K))
    Atune_arr.append(Atune)

    Btune = (1/n_val) * sum(np.linalg.norm(y_val_multi[:, i] - X_val @ beta_pd_all[i])**2 for i in range(K))
    Btune_arr.append(Btune)

    Ctune = (1/n_val) * sum((y_val_multi[:, i] - X_val @ beta_hat_all[i]) @ (y_val_multi[:, i] - X_val @ beta_pd_all[i]) for i in range(K))
    Ctune_arr.append(Ctune)

    xi_tune = (Atune - Ctune)/(Atune + Btune - 2*Ctune)
    xi_tune_arr.append(xi_tune)

    for i in range(K):
        beta_sd_i = (1 - xi_emp) * beta_hat_all[i] + xi_emp * beta_pd_all[i]
        beta_sd_all.append(beta_sd_i)
        beta_sd_tune_i = (1 - xi_tune) * beta_hat_all[i] + xi_tune * beta_pd_all[i]
        beta_sd_tune_all.append(beta_sd_tune_i)

    #empirical risks and tuning risk
    R = test_mse(beta_hat_all)
    R_pd = test_mse(beta_pd_all)
    R_sd = test_mse(beta_sd_all)
    R_sd_tune = test_mse(beta_sd_tune_all)

    xi_fresh_arr.append(xi_emp)
    xi_tune_arr.append(xi_tune)
    A_fresh_arr.append(A_emp)
    B_fresh_arr.append(B_emp)
    C_fresh_arr.append(C_emp)
    R_arr.append(R)
    R_pd_fresh_arr.append(R_pd)
    R_sd_fresh_arr.append(R_sd)
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
ax.set_title('CIFAR-10 on pre-trained ResNet-18', fontsize=ftsize + 4)

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