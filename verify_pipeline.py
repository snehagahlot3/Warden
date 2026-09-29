import numpy as np
import json

print('=' * 70)
print('VERIFICATION: All processed data files load correctly')
print('=' * 70)

X_train = np.load('data/processed/X_train.npy')
y_train = np.load('data/processed/y_train.npy')
X_val = np.load('data/processed/X_val.npy')
y_val = np.load('data/processed/y_val.npy')
X_test = np.load('data/processed/X_test.npy')
y_test = np.load('data/processed/y_test.npy')

print(f'X_train: {X_train.shape}, dtype={X_train.dtype}')
print(f'y_train: {y_train.shape}, dtype={y_train.dtype}')
print(f'X_val:   {X_val.shape}, dtype={X_val.dtype}')
print(f'y_val:   {y_val.shape}, dtype={y_val.dtype}')
print(f'X_test:  {X_test.shape}, dtype={X_test.dtype}')
print(f'y_test:  {y_test.shape}, dtype={y_test.dtype}')

total = len(X_train) + len(X_val) + len(X_test)
print(f'\nTotal sequences: {total:,}')

with open('data/processed/pipeline_config.json') as f:
    config = json.load(f)
print(f'\nPipeline config loaded.')
print(f'Source: {config["source_dataset"]}')
print(f'Window size: {config["window_size"]} flows')
print(f'Feature count: {config["num_features"]}')
print(f'Train attack pct: {config["train_attack_pct"]}%')
print(f'Val attack pct: {config["val_attack_pct"]}%')
print(f'Test attack pct: {config["test_attack_pct"]}%')

with open('data/processed/feature_names.json') as f:
    feat_names = json.load(f)
print(f'\nFeature names ({len(feat_names)} features)')
print(f'First 5: {feat_names[:5]}')

# Verify shapes match expected WARDEN-T format
assert X_train.shape == (2513, 10, 81), f"X_train shape mismatch: {X_train.shape}"
assert X_val.shape == (838, 10, 81), f"X_val shape mismatch: {X_val.shape}"
assert X_test.shape == (838, 10, 81), f"X_test shape mismatch: {X_test.shape}"
assert y_train.shape == (2513,), f"y_train shape mismatch: {y_train.shape}"
assert y_val.shape == (838,), f"y_val shape mismatch: {y_val.shape}"
assert y_test.shape == (838,), f"y_test shape mismatch: {y_test.shape}"

print('\n' + '=' * 70)
print('ALL ASSERTIONS PASSED - DATA PIPELINE COMPLETE')
print('=' * 70)
print(f'\nWARDEN-T data is ready for model input:')
print(f'  Input shape:  [samples, 10, 81]')
print(f'  Target shape: [samples]')
print(f'  Sequence length: 10')
print(f'  Features per state: 81')
