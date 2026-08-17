"""
Standalone evaluation for the already-trained LSTM-AE checkpoint.
Does NOT retrain — loads lstm_ae_best.pt and scores the test set.

Edit DATA_DIR below to point at the folder containing:
    X_normal_test_norm.npy, X_anomaly_test_norm.npy, y_anomaly_test.npy
It does not need to be the same folder as X_train.npy/X_val.npy.
"""

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from pathlib import Path
import json

from train_lstm_ae import LSTMAutoencoder, CONFIG

# Project folder
BASE_DIR = Path(__file__).resolve().parent

# Test data is in the same folder as this script
DATA_DIR = BASE_DIR

# Trained model checkpoint
CKPT_DIR = BASE_DIR / "lstm_ae_run"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("DATA_DIR:", DATA_DIR)
print("CKPT_DIR:", CKPT_DIR)
print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

X_normal = np.load(
    DATA_DIR / "X_normal_test_norm.npy"
).astype(np.float32)

X_anomaly = np.load(
    DATA_DIR / "X_anomaly_test_norm.npy"
).astype(np.float32)

y_anomaly = np.load(
    DATA_DIR / "y_anomaly_test.npy"
).astype(int)

assert y_anomaly.shape[0] == X_anomaly.shape[0], \
    f"Label count {y_anomaly.shape[0]} != anomaly window count {X_anomaly.shape[0]}"
if not np.all(y_anomaly > 0):
    print(f"WARNING: {np.sum(y_anomaly == 0)} entries in y_anomaly_test.npy are 0/non-positive.")

y_normal = np.zeros(X_normal.shape[0], dtype=int)
y_anomaly_binary = (y_anomaly > 0).astype(int)

X_test = np.concatenate([X_normal, X_anomaly], axis=0)
y_test = np.concatenate([y_normal, y_anomaly_binary], axis=0)
print(f"Test set: {X_normal.shape[0]} normal, {X_anomaly.shape[0]} anomaly "
      f"({y_test.mean()*100:.2f}% positive)")

model = LSTMAutoencoder(
    n_features=CONFIG["n_features"],
    hidden_dim=CONFIG["hidden_dim"],
    latent_dim=CONFIG["latent_dim"],
    num_layers=CONFIG["num_layers"],
    dropout=CONFIG["dropout"],
).to(device)
model.load_state_dict(torch.load(CKPT_DIR / "lstm_ae_best.pt", map_location=device))
model.eval()

batch_size = 512
scores = []
with torch.no_grad():
    for i in range(0, len(X_test), batch_size):
        x = torch.from_numpy(X_test[i:i + batch_size]).to(device)
        recon, _ = model(x)
        err = ((recon - x) ** 2).mean(dim=[1, 2])
        scores.append(err.cpu().numpy())
scores = np.concatenate(scores)

auc = roc_auc_score(y_test, scores)
ap = average_precision_score(y_test, scores)

print("\n--- Test Set Evaluation ---")
print(f"ROC-AUC:              {auc:.3f}")
print(f"Average Precision:    {ap:.3f}")
print(f"Normal error (mean):  {scores[y_test==0].mean():.4f}")
print(f"Anomaly error (mean): {scores[y_test==1].mean():.4f}")

np.save(CKPT_DIR / "test_scores.npy", scores)
with open(CKPT_DIR / "test_metrics.json", "w") as f:
    json.dump({"roc_auc": auc, "average_precision": ap}, f, indent=2)


