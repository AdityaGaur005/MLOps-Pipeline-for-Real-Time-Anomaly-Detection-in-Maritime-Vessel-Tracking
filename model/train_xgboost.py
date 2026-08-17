"""
Hybrid anomaly classifier: engineered features + LSTM-AE reconstruction error
-> XGBoost, evaluated with grouped CV by MMSI (vessel ID) to avoid identity leakage.

REQUIRES an MMSI array aligned with your window arrays:
    mmsi_normal_test.npy  -> (N_normal,) vessel ID per normal-test window
    mmsi_anomaly_test.npy -> (N_anomaly,) vessel ID per anomaly-test window
If you don't have these, see the FALLBACK note printed at runtime — results
without grouping are optimistic and should not be trusted as final numbers.
"""

import numpy as np
import torch
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
import xgboost as xgb

try:
    from sklearn.model_selection import StratifiedGroupKFold
    HAS_GROUP_CV = True
except ImportError:
    HAS_GROUP_CV = False

from train_lstm_ae import LSTMAutoencoder, CONFIG

# ---------------------------------------------------------------------------
DATA_DIR = Path(r"C:\Users\Aman Hooda\Desktop\gaur")           # <-- SET THIS
CKPT_DIR = Path(r"C:\Users\Aman Hooda\Desktop\gaur\lstm_ae_run")
N_FOLDS = 5
RANDOM_STATE = 42
# ---------------------------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_labeled_data():
    X_normal = np.load(DATA_DIR / "X_normal_test_norm.npy").astype(np.float32)
    X_anomaly = np.load(DATA_DIR / "X_anomaly_test_norm.npy").astype(np.float32)
    y_anomaly = np.load(DATA_DIR / "y_anomaly_test.npy").astype(int)

    y_normal = np.zeros(X_normal.shape[0], dtype=int)
    y_anomaly_binary = (y_anomaly > 0).astype(int)

    X = np.concatenate([X_normal, X_anomaly], axis=0)
    y = np.concatenate([y_normal, y_anomaly_binary], axis=0)

    mmsi_normal_path = DATA_DIR / "mmsi_normal_test.npy"
    mmsi_anomaly_path = DATA_DIR / "mmsi_anomaly_test.npy"
    if mmsi_normal_path.exists() and mmsi_anomaly_path.exists():
        mmsi_normal = np.load(mmsi_normal_path)
        mmsi_anomaly = np.load(mmsi_anomaly_path)
        groups = np.concatenate([mmsi_normal, mmsi_anomaly], axis=0)
        print("MMSI files found — using grouped CV (correct, leakage-safe).")
    else:
        groups = None
        print("FALLBACK: no MMSI files found. Using ungrouped StratifiedKFold.\n"
              "  WARNING: if the same vessel contributes windows to both the train\n"
              "  and validation side of a fold, reported AUC/AP will be optimistically\n"
              "  biased. Treat these numbers as an upper bound, not a final result,\n"
              "  until you can supply per-window MMSI.")

    print(f"Labeled set: {len(y)} windows, {y.sum()} positive ({y.mean()*100:.2f}%)")
    return X, y, groups


def engineer_features(X):
    """
    X: (N, 30, 7) normalized windows.
    Feature order matches R&D log: [lat, lon, sog, cog, computed_speed,
    acceleration, heading_change]
    Returns (N, n_features) array of hand-crafted summary stats.
    """
    lat, lon = X[:, :, 0], X[:, :, 1]
    sog = X[:, :, 2]
    cog = X[:, :, 3]
    comp_speed = X[:, :, 4]
    accel = X[:, :, 5]
    heading_chg = X[:, :, 6]

    feats = {}
    for name, arr in [("sog", sog), ("comp_speed", comp_speed),
                       ("accel", accel), ("heading_chg", heading_chg), ("cog", cog)]:
        feats[f"{name}_mean"] = arr.mean(axis=1)
        feats[f"{name}_std"] = arr.std(axis=1)
        feats[f"{name}_max"] = arr.max(axis=1)
        feats[f"{name}_min"] = arr.min(axis=1)

    # speed/accel volatility — sharp changes are more anomaly-indicative than raw levels
    feats["accel_absmax"] = np.abs(accel).max(axis=1)
    feats["heading_chg_absmax"] = np.abs(heading_chg).max(axis=1)
    feats["heading_chg_count_large"] = (np.abs(heading_chg) > np.abs(heading_chg).mean()
                                         + 2 * heading_chg.std()).sum(axis=1)

    # net displacement vs. path length -> low ratio = loitering/erratic path,
    # high ratio = straight-line travel (normal transit)
    lat_diff = np.diff(lat, axis=1)
    lon_diff = np.diff(lon, axis=1)
    step_dist = np.sqrt(lat_diff ** 2 + lon_diff ** 2)
    path_length = step_dist.sum(axis=1)
    net_disp = np.sqrt((lat[:, -1] - lat[:, 0]) ** 2 + (lon[:, -1] - lon[:, 0]) ** 2)
    feats["straightness"] = net_disp / (path_length + 1e-6)
    feats["path_length"] = path_length

    return np.column_stack(list(feats.values())), list(feats.keys())


def get_reconstruction_scores(X):
    model = LSTMAutoencoder(
        n_features=CONFIG["n_features"], hidden_dim=CONFIG["hidden_dim"],
        latent_dim=CONFIG["latent_dim"], num_layers=CONFIG["num_layers"],
        dropout=CONFIG["dropout"],
    ).to(device)
    model.load_state_dict(torch.load(CKPT_DIR / "lstm_ae_best.pt", map_location=device))
    model.eval()

    scores = []
    batch_size = 512
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            x = torch.from_numpy(X[i:i + batch_size]).to(device)
            recon, _ = model(x)
            err = ((recon - x) ** 2).mean(dim=[1, 2])
            scores.append(err.cpu().numpy())
    return np.concatenate(scores)


def run_cv(X_feat, y, groups):
    n_pos, n_neg = y.sum(), len(y) - y.sum()
    scale_pos_weight = n_neg / n_pos

    if groups is not None and HAS_GROUP_CV:
        splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        split_args = (X_feat, y, groups)
    else:
        splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        split_args = (X_feat, y)

    oof_scores = np.zeros(len(y))
    fold_aucs, fold_aps = [], []

    for fold, (train_idx, val_idx) in enumerate(splitter.split(*split_args), 1):
        X_tr, X_val = X_feat[train_idx], X_feat[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        clf = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr",
            tree_method="hist",
            random_state=RANDOM_STATE,
            early_stopping_rounds=20,
        )
        clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

        val_scores = clf.predict_proba(X_val)[:, 1]
        oof_scores[val_idx] = val_scores

        auc = roc_auc_score(y_val, val_scores)
        ap = average_precision_score(y_val, val_scores)
        fold_aucs.append(auc)
        fold_aps.append(ap)
        print(f"Fold {fold}/{N_FOLDS}: AUC={auc:.3f}  AP={ap:.3f}  "
              f"(n_val={len(val_idx)}, pos={y_val.sum()})")

    overall_auc = roc_auc_score(y, oof_scores)
    overall_ap = average_precision_score(y, oof_scores)
    print(f"\nOut-of-fold overall: AUC={overall_auc:.3f}  AP={overall_ap:.3f}")
    print(f"Fold mean:           AUC={np.mean(fold_aucs):.3f} (+/-{np.std(fold_aucs):.3f})  "
          f"AP={np.mean(fold_aps):.3f} (+/-{np.std(fold_aps):.3f})")
    return oof_scores, overall_auc, overall_ap


if __name__ == "__main__":
    X, y, groups = load_labeled_data()

    print("\nEngineering features...")
    X_engineered, feat_names = engineer_features(X)

    print("Scoring windows with LSTM-AE...")
    recon_scores = get_reconstruction_scores(X)

    X_full = np.column_stack([X_engineered, recon_scores])
    feat_names = feat_names + ["lstm_recon_error"]
    print(f"Total features: {len(feat_names)}")

    print("\nRunning cross-validated XGBoost...")
    oof_scores, auc, ap = run_cv(X_full, y, groups)

    # Fit final model on all data for deployment / feature importance inspection
    n_pos, n_neg = y.sum(), len(y) - y.sum()
    final_clf = xgb.XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=n_neg / n_pos, eval_metric="aucpr",
        tree_method="hist", random_state=RANDOM_STATE,
    )
    final_clf.fit(X_full, y)

    importances = sorted(zip(feat_names, final_clf.feature_importances_),
                          key=lambda t: -t[1])
    print("\nTop 10 features by importance:")
    for name, imp in importances[:10]:
        print(f"  {name:25s} {imp:.4f}")

    final_clf.save_model(str(CKPT_DIR / "xgb_hybrid.json"))
    print(f"\nSaved final model to {CKPT_DIR / 'xgb_hybrid.json'}")
