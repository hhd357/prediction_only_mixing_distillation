# ResNet-34 features pretrained on ImageNet, with Caltech-101 dataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, TensorDataset
import torchvision
from torchvision import transforms
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import FixedLocator, FixedFormatter


# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

K = 257   # Caltech-256 has 256 object classes + 1 clutter class = 257 total

# Transforms
transform_base = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Lambda(lambda img: img.convert("RGB")),  # ← force 3 channels
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load Caltech-256
full_dataset = torchvision.datasets.Caltech256(
    root='./data', download=True, transform=transform_base)

print(f"Total Caltech-256 samples: {len(full_dataset)}")

# Manual 80/20 train/test split
n_total     = len(full_dataset)
n_test_full = int(0.3 * n_total)
n_train_full = n_total - n_test_full

rng = np.random.default_rng(seed=2026)
all_indices  = rng.permutation(n_total)

train_dataset_full = Subset(full_dataset, all_indices[:n_train_full])
test_dataset_full  = Subset(full_dataset, all_indices[n_train_full:])

print(f"train_dataset_full: {len(train_dataset_full)} samples")
print(f"test_dataset_full : {len(test_dataset_full)}  samples")

# ResNet-34 feature extractor
model = torchvision.models.resnet34(pretrained=True)
model.fc = nn.Identity()   
model = model.to(device)
model.eval()

# Subsample dataset

K = 257
n_train    = 5000
n_fresh    = 15000
n_val      = 1000
n_test     = 6000

p          = 512
noise_rate = 0.4

# Fix test set
test_rng         = np.random.default_rng(seed=0)
all_test_indices = test_rng.permutation(len(test_dataset_full))
test_indices     = all_test_indices[:n_test]

# Fix val set
val_rng                = np.random.default_rng(seed=1)
all_val_indices        = val_rng.permutation(len(test_dataset_full))
remaining_test_indices = np.setdiff1d(all_val_indices, test_indices)
val_indices            = remaining_test_indices[:n_val]

# Fix train set
train_rng     = np.random.default_rng(seed=2)
train_perm    = train_rng.permutation(len(train_dataset_full))
train_indices = train_perm[:n_train]

# Fresh set
remaining_train_indices = np.setdiff1d(
    np.arange(len(train_dataset_full)), train_indices
)
fresh_rng     = np.random.default_rng(seed=2026)
fresh_indices = fresh_rng.choice(remaining_train_indices, n_fresh, replace=False)

print(f"train_dataset_full size : {len(train_dataset_full)}")
print(f"test_dataset_full  size : {len(test_dataset_full)}")
assert n_train + n_fresh <= len(train_dataset_full), \
    f"n_train + n_fresh ({n_train + n_fresh}) exceeds train_dataset_full ({len(train_dataset_full)})"
assert n_test + n_val <= len(test_dataset_full), \
    f"n_test + n_val ({n_test + n_val}) exceeds test_dataset_full ({len(test_dataset_full)})"

train_dataset = Subset(train_dataset_full, train_indices)
fresh_dataset = Subset(train_dataset_full, fresh_indices)
val_dataset   = Subset(test_dataset_full,  val_indices)
test_dataset  = Subset(test_dataset_full,  test_indices)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
fresh_loader = DataLoader(fresh_dataset, batch_size=32, shuffle=False)
val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False)

# Feature extraction
def extract_features(loader):
    features, labels = [], []
    with torch.no_grad():
        for images, lbls in loader:
            images = images.to(device)
            feats  = model(images)
            features.append(feats.cpu().numpy())
            labels.append(lbls.numpy())
    return np.vstack(features), np.hstack(labels)

print("Extracting train features...")
X_train, y_train = extract_features(train_loader)
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

print("Extracting fresh features...")
X_fresh, y_fresh = extract_features(fresh_loader)
print(f"X_fresh shape: {X_fresh.shape}, y_fresh shape: {y_fresh.shape}")

print("Extracting val features...")
X_val, y_val = extract_features(val_loader)
print(f"X_val   shape: {X_val.shape},   y_val   shape: {y_val.shape}")

print("Extracting test features...")
X_test, y_test = extract_features(test_loader)
print(f"X_test  shape: {X_test.shape},  y_test  shape: {y_test.shape}")

# One-hot encode
encoder       = OneHotEncoder(sparse_output=False)
y_train_multi = encoder.fit_transform(y_train.reshape(-1, 1))  # (n_train, K)
y_fresh_multi = encoder.transform(y_fresh.reshape(-1, 1))      # (n_fresh, K)
y_val_multi   = encoder.transform(y_val.reshape(-1, 1))        # (n_val,   K)
y_test_multi  = encoder.transform(y_test.reshape(-1, 1))       # (n_test,  K)

# Label corruption
def corrupt_labels(y, K, corruption_rate=0.1, seed=2026):
    """
    Randomly corrupt a fraction of labels by replacing them
    with a uniformly random DIFFERENT class.

    Args:
        y:               (n,) integer labels
        K:               number of classes
        corruption_rate: fraction of labels to corrupt
        seed:            random seed (independent from all other RNGs above)

    Returns:
        y_corrupted:    (n,) corrupted integer labels
        corrupted_mask: (n,) bool — True where label was changed
    """
    rng         = np.random.default_rng(seed)
    y_corrupted = y.copy()
    n           = len(y)

    n_corrupt   = int(n * corruption_rate)
    corrupt_idx = rng.choice(n, n_corrupt, replace=False)

    for idx in corrupt_idx:
        other_classes    = np.delete(np.arange(K), y[idx])
        y_corrupted[idx] = rng.choice(other_classes)

    corrupted_mask              = np.zeros(n, dtype=bool)
    corrupted_mask[corrupt_idx] = True
    return y_corrupted, corrupted_mask

# Corrupt train
y_train_corrupted, train_corrupted_mask = corrupt_labels(
    y_train, K, corruption_rate=noise_rate, seed=2026
)
y_train_corrupted_multi = encoder.transform(y_train_corrupted.reshape(-1, 1))

# Corrupt val and test (just to mimic the teacher's predictions)
y_val_corrupted,  val_corrupted_mask  = corrupt_labels(
    y_val,  K, corruption_rate=noise_rate, seed=2026
)
y_test_corrupted, test_corrupted_mask = corrupt_labels(
    y_test, K, corruption_rate=noise_rate, seed=2026
)
y_val_corrupted_multi  = encoder.transform(y_val_corrupted.reshape(-1, 1))
y_test_corrupted_multi = encoder.transform(y_test_corrupted.reshape(-1, 1))

# Reassign y_train to corrupted; keep val/test clean for evaluation
y_train_original = y_train.copy()
y_val_original   = y_val.copy()
y_test_original  = y_test.copy()

y_train       = y_train_corrupted
y_train_multi = y_train_corrupted_multi

# Check
print(f"\nSplit summary:")
print(f"  X_train : {X_train.shape}  (noise_rate={noise_rate:.0%})")
print(f"  X_fresh : {X_fresh.shape}  (clean, unseen during training)")
print(f"  X_val   : {X_val.shape}")
print(f"  X_test  : {X_test.shape}")
print(f"  Corrupted train labels : {train_corrupted_mask.sum()} / {len(y_train)}")
print(f"  Corrupted val   labels : {val_corrupted_mask.sum()}  / {len(y_val)}")
print(f"  Corrupted test  labels : {test_corrupted_mask.sum()} / {len(y_test)}")

### MAIN COMPUTATION CODE
seeds   = [2026]
lr   = 1e-3 # teacher lr
lr_s = 5e-2 # student lr

lambda_list = np.logspace(-4, 1, 11)

# ── Seed ─────────────────────────────────────────────────────────────────────
def set_seed(seed=2026):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

def linear_probe_ce(X_train, y_train, X_val, y_val, X_test, y_test,
                    K=10, lambda_reg=1e-4, lr=1e-2,
                    epochs=200, batch_size=128, seed=2026):
    set_seed(seed)

    X_tr = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_tr = torch.tensor(y_train, dtype=torch.long).to(device)
    X_va = torch.tensor(X_val,   dtype=torch.float32).to(device)
    y_va = torch.tensor(y_val,   dtype=torch.long).to(device)
    X_te = torch.tensor(X_test,  dtype=torch.float32).to(device)
    y_te = torch.tensor(y_test,  dtype=torch.long).to(device)

    train_dl = DataLoader(TensorDataset(X_tr, y_tr),
                          batch_size=batch_size, shuffle=True)

    head = nn.Linear(X_train.shape[1], K).to(device)
    nn.init.xavier_uniform_(head.weight)
    nn.init.zeros_(head.bias)

    optimizer = optim.SGD(head.parameters(), lr=lr,
                          momentum=0.9, weight_decay=lambda_reg)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    criterion = nn.CrossEntropyLoss()

    train_accs, val_accs, test_accs, losses = [], [], [], []

    for epoch in range(epochs):
        head.train()
        epoch_loss = 0.0
        for Xb, yb in train_dl:
            optimizer.zero_grad()
            loss = criterion(head(Xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(yb)

        scheduler.step()

        head.eval()
        with torch.no_grad():
            train_preds = head(X_tr).argmax(dim=1)
            val_preds   = head(X_va).argmax(dim=1)
            test_preds  = head(X_te).argmax(dim=1)

        train_accs.append((train_preds == y_tr).float().mean().item())
        val_accs.append(  (val_preds   == y_va).float().mean().item())
        test_accs.append( (test_preds  == y_te).float().mean().item())
        losses.append(epoch_loss / len(y_tr))

    final_W = head.weight.detach().cpu().numpy().T
    final_b = head.bias.detach().cpu().numpy()
    return final_W, final_b, train_accs, val_accs, test_accs, losses


def linear_probe_soft(X_train, Y_soft, X_val, y_val, X_test, y_test,
                      K=10, lambda_reg=1e-4, lr=1e-2,
                      epochs=200, batch_size=128, seed=2026):
    set_seed(seed)

    X_tr     = torch.tensor(X_train, dtype=torch.float32).to(device)
    Y_soft_t = torch.tensor(Y_soft,  dtype=torch.float32).to(device)
    X_va     = torch.tensor(X_val,   dtype=torch.float32).to(device)
    y_va     = torch.tensor(y_val,   dtype=torch.long).to(device)
    X_te     = torch.tensor(X_test,  dtype=torch.float32).to(device)
    y_te     = torch.tensor(y_test,  dtype=torch.long).to(device)

    train_dl = DataLoader(TensorDataset(X_tr, Y_soft_t),
                          batch_size=batch_size, shuffle=True)

    student = nn.Linear(X_train.shape[1], K).to(device)
    nn.init.xavier_uniform_(student.weight)
    nn.init.zeros_(student.bias)

    optimizer = optim.SGD(student.parameters(), lr=lr,
                          momentum=0.9, weight_decay=lambda_reg)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    def soft_ce_loss(logits, soft_targets):
        log_probs = torch.log_softmax(logits, dim=1)
        return -(soft_targets * log_probs).sum(dim=1).mean()

    train_accs, val_accs, test_accs, losses = [], [], [], []

    for epoch in range(epochs):
        student.train()
        epoch_loss = 0.0
        for Xb, Yb_soft in train_dl:
            optimizer.zero_grad()
            loss = soft_ce_loss(student(Xb), Yb_soft)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(Xb)

        scheduler.step()

        student.eval()
        with torch.no_grad():
            train_preds = student(X_tr).argmax(dim=1)
            val_preds   = student(X_va).argmax(dim=1)
            test_preds  = student(X_te).argmax(dim=1)
            y_tr_hard   = Y_soft_t.argmax(dim=1)

        train_accs.append((train_preds == y_tr_hard).float().mean().item())
        val_accs.append(  (val_preds   == y_va     ).float().mean().item())
        test_accs.append( (test_preds  == y_te     ).float().mean().item())
        losses.append(epoch_loss / len(Y_soft_t))

    final_W = student.weight.detach().cpu().numpy().T
    final_b = student.bias.detach().cpu().numpy()
    return final_W, final_b, train_accs, val_accs, test_accs, losses

def get_logits(W, b, X):
    logits = X @ W + b
    exp    = np.exp(logits - logits.max(axis=1, keepdims=True))
    return logits, exp / exp.sum(axis=1, keepdims=True)

def make_onehot(y, K):
    n  = len(y)
    oh = np.zeros((n, K), dtype=np.float32)
    oh[np.arange(n), y] = 1.0
    return oh

# Xi grid 
_xi_base  = np.linspace(-20, 20, 1000)
xi_sweep  = np.sort(np.unique(np.concatenate([_xi_base, [0.0, 1.0]])))

# Storage over lambda values
avg_teacher       = []
avg_pd            = []
avg_student_soft  = []
avg_student_hard  = []
avg_best_xi_soft  = []
avg_best_xi_hard  = []

for lam in lambda_list:
    print("\n" + "=" * 60)
    print(f"LAMBDA = {lam:.4e}")
    print("=" * 60)

    seed_teacher_test      = []
    seed_pd_test           = []
    seed_student_soft_test = []
    seed_student_hard_test = []
    seed_best_xi_soft      = []
    seed_best_xi_hard      = []

    for seed in seeds:
        print(f"  SEED {seed}")

        # Teacher: trained on X_train / y_train
        W, b, _, val_accs, test_accs, _ = linear_probe_ce(
            X_train, y_train, X_val, y_val, X_test, y_test,
            K=K, lambda_reg=lam, lr=lr, epochs=200, batch_size=128, seed=seed
        )
        print(f"    Teacher  Val: {val_accs[-1]:.4f}  Test: {test_accs[-1]:.4f}")
        seed_teacher_test.append(test_accs[-1])

        # Teacher soft labels on X_fresh
        head_teacher = nn.Linear(X_train.shape[1], K).to(device)
        head_teacher.weight.data = torch.tensor(W.T, dtype=torch.float32).to(device)
        head_teacher.bias.data   = torch.tensor(b,   dtype=torch.float32).to(device)
        head_teacher.eval()

        with torch.no_grad():
            X_fr_tensor      = torch.tensor(X_fresh, dtype=torch.float32).to(device)
            Y_fresh_soft     = torch.softmax(head_teacher(X_fr_tensor), dim=1)
            Y_fresh_soft_np  = Y_fresh_soft.cpu().numpy()

        # PD student: trained on X_fresh with teacher soft labels
        W_s, b_s, _, val_accs_s, test_accs_s, _ = linear_probe_soft(
            X_fresh, Y_fresh_soft_np, X_val, y_val, X_test, y_test,
            K=K, lambda_reg=lam, lr=lr_s, epochs=200, batch_size=128, seed=seed
        )
        print(f"    Soft-student  Val: {val_accs_s[-1]:.4f}  Test: {test_accs_s[-1]:.4f}")
        seed_pd_test.append(test_accs_s[-1])

        # Precompute probabilities for xi sweep
        _, probs_teacher_val  = get_logits(W,   b,   X_val)
        _, probs_student_val  = get_logits(W_s, b_s, X_val)
        _, probs_teacher_test = get_logits(W,   b,   X_test)
        _, probs_student_test = get_logits(W_s, b_s, X_test)

        # Noisy one-hots for val/test (corrupted labels)
        y_val_noisy_oh  = make_onehot(y_val_corrupted,  K)
        y_test_noisy_oh = make_onehot(y_test_corrupted, K)

        # Xi sweep
        acc_pred_val  = np.array([
            (((1-xi)*probs_teacher_val + xi*probs_student_val).argmax(axis=1) == y_val).mean()
            for xi in xi_sweep
        ])
        acc_noisy_val = np.array([
            (((1-xi)*y_val_noisy_oh + xi*probs_teacher_val).argmax(axis=1) == y_val).mean()
            for xi in xi_sweep
        ])

        best_xi_soft  = xi_sweep[acc_pred_val.argmax()]
        best_xi_noisy = xi_sweep[acc_noisy_val.argmax()]

        bt_pred  = (1 - best_xi_soft)  * probs_teacher_test + best_xi_soft  * probs_student_test
        bt_noisy = (1 - best_xi_noisy) * y_test_noisy_oh    + best_xi_noisy * probs_teacher_test

        seed_student_soft_test.append((bt_pred.argmax(axis=1)  == y_test).mean())
        seed_student_hard_test.append((bt_noisy.argmax(axis=1) == y_test).mean())
        seed_best_xi_soft.append(best_xi_soft)
        seed_best_xi_hard.append(best_xi_noisy)

        print(f"    Best-xi pred avg   xi={best_xi_soft:.4f}  Test: {seed_student_soft_test[-1]:.4f}")
        print(f"    Best-xi noisy mix  xi={best_xi_noisy:.4f}  Test: {seed_student_hard_test[-1]:.4f}")

    avg_teacher.append(      np.mean(seed_teacher_test))
    avg_pd.append(           np.mean(seed_pd_test))
    avg_student_soft.append( np.mean(seed_student_soft_test))
    avg_student_hard.append( np.mean(seed_student_hard_test))
    avg_best_xi_soft.append( np.mean(seed_best_xi_soft))
    avg_best_xi_hard.append( np.mean(seed_best_xi_hard))

avg_teacher      = np.array(avg_teacher)
avg_pd           = np.array(avg_pd)
avg_student_soft = np.array(avg_student_soft) # our reported target
avg_student_hard = np.array(avg_student_hard)
avg_best_xi_soft = np.array(avg_best_xi_soft) # our reported target
avg_best_xi_hard = np.array(avg_best_xi_hard)

# ── Summary table ─────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print(f"LAMBDA SWEEP SUMMARY")
print("=" * 80)
print(f"{'Lambda':<12} {'Teacher':>10} {'Soft-KD':>10} {'PredAvg':>10} "
      f"{'NoisyMix':>10} {'xi*_pred':>10} {'xi*_noisy':>11}")
print("-" * 80)
for i, lam in enumerate(lambda_list):
    print(f"{lam:<12.4e} "
          f"{avg_teacher[i]:>10.4f} "
          f"{avg_pd[i]:>10.4f} "
          f"{avg_student_soft[i]:>10.4f} "
          f"{avg_student_hard[i]:>10.4f} "
          f"{avg_best_xi_soft[i]:>10.4f} "
          f"{avg_best_xi_hard[i]:>11.4f}")
    
### PLOT
plt.rcParams['text.usetex'] = False

fig, ax = plt.subplots(1, 1, figsize=(10, 8))

colors_main = ['tab:blue', '#A0CBE8', 'tab:green', 'tab:olive']
lw     = 3
ftsize = 18

# Misclassification error
err_teacher      = 1 - avg_teacher
err_pd           = 1 - avg_pd
err_student_soft = 1 - avg_student_soft
err_student_hard = 1 - avg_student_hard

teacher_line,      = ax.semilogx(lambda_list, err_teacher,      label=r'$y_{\mathrm{pd}}$',
                                  color=colors_main[0], linewidth=lw)
pd_line,           = ax.semilogx(lambda_list, err_pd,           label=r'$y_{\mathrm{pd}, 2}$',
                                  color=colors_main[1], linewidth=lw)
student_soft_line, = ax.semilogx(lambda_list, err_student_soft, label=r'$y_{\mathrm{sd}, 2}$',
                                  color=colors_main[2], linewidth=lw, alpha = 0.7)

for line, err in zip(
        [teacher_line, pd_line, student_soft_line],
        [err_teacher,  err_pd,  err_student_soft]):
    idx = np.argmin(err)
    ax.plot(lambda_list[idx], err[idx],
            marker='*', color=line.get_color(), markersize=24,
            markeredgecolor='white', markeredgewidth=1, zorder=5)

print(f"Best error - Teacher:       {err_teacher.min():.4f} at lambda={lambda_list[err_teacher.argmin()]:.4e}")
print(f"Best error - Soft-KD:       {err_pd.min():.4f} at lambda={lambda_list[err_pd.argmin()]:.4e}")
print(f"Best error - Pred avg:      {err_student_soft.min():.4f} at lambda={lambda_list[err_student_soft.argmin()]:.4e}")
print(f"Best error - Noisy mix:     {err_student_hard.min():.4f} at lambda={lambda_list[err_student_hard.argmin()]:.4e}")

ax.set_xlabel(r'Ridge penalty $\lambda$', fontsize=ftsize + 4)
ax.set_ylabel('Test misclassification error', fontsize=ftsize + 4)
ax.set_title(rf'Caltech-256 ($p={noise_rate:.1f}$)', fontsize=ftsize + 4)
ax.tick_params(axis='both', labelsize=ftsize)
ax.grid(True, alpha=0.3)

ax.set_ylim(0.2, 1.0)

ax.spines['left'].set_color('tab:blue')

# Right axis: xi curve
ax_twin = ax.twinx()

xi_soft_line, = ax_twin.semilogx(lambda_list, avg_best_xi_soft, label=r'$\xi^{\star}_{\mathrm{sd}, 2}$',
                                   color='tab:red', linestyle='-', linewidth=lw)

ax_twin.set_ylabel(r'Optimal mixing parameter', fontsize=ftsize + 4, color='tab:red')
ax_twin.tick_params(axis='y', labelsize=ftsize, labelcolor='tab:red')
ax_twin.spines['right'].set_color('tab:red')
ax_twin.spines['top'].set_visible(False)
ax_twin.spines['bottom'].set_visible(False)
ax_twin.spines['left'].set_visible(False)

ax_twin.set_xticks([])
ax_twin.set_xticks([], minor=True)

ticks = [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1]
ax.xaxis.set_major_locator(FixedLocator(ticks))
ax.xaxis.set_major_formatter(FixedFormatter([r'$10^{-4}$', r'$10^{-3}$', r'$10^{-2}$',
                                              r'$10^{-1}$', r'$10^{0}$',  r'$10^{1}$']))
ax.tick_params(axis='x', labelsize=ftsize)
ax_twin.set_ylim(-20.5, 10.5)

# Legend box
all_lines  = [teacher_line, pd_line, student_soft_line,
              xi_soft_line]
all_labels = [l.get_label() for l in all_lines]
ax.legend(all_lines, all_labels, loc='upper center',
          bbox_to_anchor=(0.5, -0.15), ncol=5, fontsize=ftsize + 2)

plt.tight_layout()
plt.show()   