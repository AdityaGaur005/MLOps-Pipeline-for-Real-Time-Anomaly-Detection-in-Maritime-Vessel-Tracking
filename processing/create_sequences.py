import pandas as pd
import numpy as np
from tqdm import tqdm
import os

years = {
    '2017': "file patht.parquet",
    '2018': "file path.parquet",
    '2019': "file path.parquet",
    '2020': "file path.parquet"
}


# Sequence parameters
WINDOW_SIZE = 30          # number of consecutive AIS points per sequence
STRIDE = 15               # overlap (sliding window step)
FEATURES = [
    'lat', 'lon',
    'speed_over_ground_knots',
    'course_over_ground_deg',
    'computed_speed_knots',
    'acceleration_knots_per_sec',
    'heading_change_deg'
]


def create_normal_sequences(df, window_size, stride, feature_cols):
    """
    Extract sliding windows only from vessels that have NO incident labels.
    Returns a numpy array of shape (num_windows, window_size, num_features).
    """
    sequences = []
    
    # Group by vessel
    grouped = df.sort_values(['MMSI', 'datetime_hst']).groupby('MMSI')
    
    # Loop over each vessel
    for mmsi, group in tqdm(grouped, total=df['MMSI'].nunique(), desc="Processing vessels"):
        # Skip vessel if it has ANY incident-labeled points (because we train only on normal)
        if group['is_incident'].any():
            continue
        
        # Extract the feature matrix for this vessel
        data = group[feature_cols].values  # shape (T, num_features)
        T = data.shape[0]
        if T < window_size:
            continue
        
        # Slide a window across the trajectory
        for start in range(0, T - window_size + 1, stride):
            window = data[start:start + window_size]
            sequences.append(window)
    
    return np.array(sequences, dtype=np.float32)

# Process each year and save the normal sequences to .npy files
for year, path in years.items():
    out_file = f"normal_sequences_{year}.npy"
    if os.path.exists(out_file):
        print(f"{out_file} already exists, skipping {year}")
        continue
    
    print(f"\nLoading {year} data from {path}...")
    df = pd.read_parquet(path)
    print(f"Loaded {len(df):,} rows, {df['MMSI'].nunique():,} vessels")
    
    print(f"Creating normal sequences for {year}...")
    seq = create_normal_sequences(df, WINDOW_SIZE, STRIDE, FEATURES)
    print(f"Created {seq.shape[0]:,} sequences")
    
    np.save(out_file, seq)
    print(f"Saved to {out_file}")



print("Loading training sequences...")
train_seqs = []
for year in ['2017', '2018', '2019']:
    fname = f"normal_sequences_{year}.npy"
    data = np.load(fname)
    print(f"{year}: {data.shape[0]:,} sequences")
    train_seqs.append(data)

X_train_full = np.concatenate(train_seqs, axis=0)
print(f"\nTotal training sequences: {X_train_full.shape[0]:,} (shape: {X_train_full.shape[1]} x {X_train_full.shape[2]})")

# Save combined training set
np.save("X_train_full.npy", X_train_full)
print("Saved combined training set to X_train_full.npy")



"""
output:
rint("Loading training sequences...")
train_seqs = []
for year in ['2017', '2018', '2019']:
    fname = f"normal_sequences_{year}.npy"
    data = np.load(fname)
    print(f"{year}: {data.shape[0]:,} sequences")
    train_seqs.append(data)

X_train_full = np.concatenate(train_seqs, axis=0)
print(f"\nTotal training sequences: {X_train_full.shape[0]:,} (shape: {X_train_full.shape[1]} x {X_train_full.shape[2]})")

# Save combined training set
np.save("X_train_full.npy", X_train_full)
print("Saved combined training set to X_train_full.npy")


Loading 2017 data from C:\Users\hawaii_2017.parquet...
Loaded 22,215,608 rows, 1,135 vessels
Creating normal sequences for 2017...
Processing vessels: 100%|█████████████████████████████████████████████████████████████| 1135/1135 [00:14<00:00, 76.83it/s]
Created 1,213,571 sequences
Saved to normal_sequences_2017.npy

Loading 2018 data from C:\Users\hawaii_2018.parquet...
Loaded 21,333,680 rows, 1,194 vessels
Creating normal sequences for 2018...
Processing vessels: 100%|████████████████████████████████████████████████████████████| 1194/1194 [00:11<00:00, 108.13it/s]
Created 1,202,965 sequences
Saved to normal_sequences_2018.npy

Loading 2019 data from C:\Users\hawaii_2019.parquet...
Loaded 22,670,306 rows, 1,145 vessels
Creating normal sequences for 2019...
Processing vessels: 100%|█████████████████████████████████████████████████████████████| 1145/1145 [00:11<00:00, 96.77it/s]
Created 1,224,831 sequences
Saved to normal_sequences_2019.npy

Loading 2020 data from C:\Users\hawaii_2020.parquet...
Loaded 22,476,746 rows, 1,000 vessels
Creating normal sequences for 2020...
Processing vessels: 100%|█████████████████████████████████████████████████████████████| 1000/1000 [00:12<00:00, 79.37it/s]
Created 1,253,579 sequences
Saved to normal_sequences_2020.npy
Loading training sequences...
2017: 1,213,571 sequences
2018: 1,202,965 sequences
2019: 1,224,831 sequences

Total training sequences: 3,641,367 (shape: 30 x 7)
Saved combined training set to X_train_full.npy
"""
