# Prediction-only distillation with optimal mixing in ridge-regularized linear and logistic regression

## Scripts for computing theoretical, estimated, empirical risks and plot

- Ridge on Real-world regression tasks and CIFAR10
  - 'blogfeedback.py'
  - 'airfoil.py'
  - 'blogfeedback_multiple_lambdas.py'
  - 'CIFAR10_ridge.py'
 
- Caltech-101 linear probing
  - 'Caltech-101.py'
  - 'Caltech-101_multiple_lambdas.py'

- Caltech-256 linear probing
  - 'Caltech-256.py'
 
- CIFAR-100 linear probing
  - 'CIFAR100_random.py'
  - 'CIFAR100_hierarchical.py'
 
- Synthetic ridge experiments
  - 'synthetic_ridge_over_SNR.py'
  - 'synthetic_ridge_DE_curve.py'
  - 'synthetic_ridge_mono_test.py'

- Synthetic logistic experiments
  - 'synthetic_logistic.py'

## Computation details

All the experiments are run on Google Colab Pro.

## Dependencies

Package | Version
--- | ---
torch | 2.2.1
matplotlib | 3.10.8
numpy | 1.26.4
pandas | 2.2.3
python | 3.11.8
scikit-learn | 1.6.1
scipy | 1.12.0
