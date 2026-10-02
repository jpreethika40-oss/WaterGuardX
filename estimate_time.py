

"""Estimate LSTM training time before full run."""
import sys, time
sys.path.insert(0, 'src')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_loader import load_raw
from preprocessing import run_preprocessing
from datasets import WaterSensorDataset
from lstm import LSTMClassifier
from training import set_seed
from config import SEED, SEQUENCE_LENGTH

set_seed(SEED)
DEVICE = 'cpu'

df = load_raw()
train_df, val_df, test_df, scaler, feat_cols = run_preprocessing(df, save=False)

train_ds = WaterSensorDataset(train_df, feat_cols, label_strategy='last')
val_ds   = WaterSensorDataset(val_df,   feat_cols, label_strategy='last')
INPUT_SIZE = train_ds.n_features

print(f"Train windows: {len(train_ds):,}")
print(f"Val   windows: {len(val_ds):,}")
print(f"Input shape:   (batch, {SEQUENCE_LENGTH}, {INPUT_SIZE})")

# Time one epoch for each batch size / hidden size combo
configs = [
    {'hidden_size': 64,  'num_layers': 1, 'batch_size': 256},
    {'hidden_size': 64,  'num_layers': 2, 'batch_size': 256},
    {'hidden_size': 128, 'num_layers': 1, 'batch_size': 256},
    {'hidden_size': 128, 'num_layers': 2, 'batch_size': 256},
    {'hidden_size': 128, 'num_layers': 2, 'batch_size': 512},
    {'hidden_size': 64,  'num_layers': 2, 'batch_size': 512},
]

n_pos = (train_ds.labels == 1).sum()
n_neg = (train_ds.labels == 0).sum()
pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32)
criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

print(f"\n{'Config':<45} {'1 epoch (s)':>12} {'30ep est (min)':>15} {'60ep est (min)':>15}")
print("-" * 90)

for cfg in configs:
    model = LSTMClassifier(INPUT_SIZE, cfg['hidden_size'],
                           cfg['num_layers'], 0.3, True)
    loader = DataLoader(train_ds, batch_size=cfg['batch_size'],
                        shuffle=True, num_workers=0)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    model.train()
    t0 = time.perf_counter()
    for X, y in loader:
        logits = model(X)
        loss   = criterion(logits, y.float())
        opt.zero_grad(); loss.backward(); opt.step()
    epoch_s = time.perf_counter() - t0

    label = f"h={cfg['hidden_size']} layers={cfg['num_layers']} bs={cfg['batch_size']}"
    print(f"  {label:<43} {epoch_s:>12.2f} {epoch_s*30/60:>15.1f} {epoch_s*60/60:>15.1f}")

# Also time val epoch
val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
model_v = LSTMClassifier(INPUT_SIZE, 128, 2, 0.3, True)
t0 = time.perf_counter()
model_v.eval()
with torch.no_grad():
    for X, y in val_loader:
        _ = model_v(X)
val_s = time.perf_counter() - t0
print(f"\n  Val epoch (h=128, layers=2, bs=256): {val_s:.2f}s")

print(f"\n  6 configs x ~30 epochs each (train+val per epoch):")
# rough: worst case h=128, layers=2, bs=256
worst_train = None
for cfg in configs:
    if cfg['hidden_size'] == 128 and cfg['num_layers'] == 2 and cfg['batch_size'] == 256:
        worst_train = cfg
# re-time
model_w = LSTMClassifier(INPUT_SIZE, 128, 2, 0.3, True)
loader_w = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=0)
opt_w = torch.optim.AdamW(model_w.parameters(), lr=1e-3)
model_w.train()
t0 = time.perf_counter()
for X, y in loader_w:
    logits = model_w(X); loss = criterion(logits, y.float())
    opt_w.zero_grad(); loss.backward(); opt_w.step()
ep_s = time.perf_counter() - t0

total_search_min = 6 * 30 * (ep_s + val_s) / 60
final_train_min  = 60 * (ep_s + val_s) / 60
print(f"  HP search (6 x 30ep):  ~{total_search_min:.0f} minutes")
print(f"  Final train (60ep):    ~{final_train_min:.0f} minutes")
print(f"  TOTAL ESTIMATE:        ~{total_search_min + final_train_min:.0f} minutes")
