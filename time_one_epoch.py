"""Time one single training epoch precisely."""
import sys, time
sys.path.insert(0, 'src')
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_loader import load_raw
from preprocessing import run_preprocessing
from datasets import WaterSensorDataset
from lstm import LSTMClassifier
from config import SEED

torch.manual_seed(SEED)

df = load_raw()
train_df, val_df, _, scaler, feat_cols = run_preprocessing(df, save=False)

train_ds = WaterSensorDataset(train_df, feat_cols, label_strategy='last')
val_ds   = WaterSensorDataset(val_df,   feat_cols, label_strategy='last')

n_pos = (train_ds.labels == 1).sum()
n_neg = (train_ds.labels == 0).sum()
pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32)
criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

print(f"Train windows: {len(train_ds):,}")
print(f"Val   windows: {len(val_ds):,}")
print()

for hidden, layers, bs in [(64, 2, 256), (128, 2, 256), (128, 2, 512)]:
    model = LSTMClassifier(train_ds.n_features, hidden, layers, 0.3, True)
    loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=0)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Train epoch
    model.train()
    t0 = time.perf_counter()
    for X, y in loader:
        logits = model(X)
        loss = criterion(logits, y.float())
        opt.zero_grad(); loss.backward(); opt.step()
    train_s = time.perf_counter() - t0

    # Val epoch
    model.eval()
    t0 = time.perf_counter()
    with torch.no_grad():
        for X, y in val_loader:
            _ = model(X)
    val_s = time.perf_counter() - t0

    epoch_s = train_s + val_s
    print(f"h={hidden} layers={layers} bs={bs}: "
          f"train={train_s:.1f}s  val={val_s:.1f}s  "
          f"total/epoch={epoch_s:.1f}s  "
          f"10ep={epoch_s*10/60:.1f}min  "
          f"3configs+final(10ep)={epoch_s*40/60:.1f}min")
