import numpy as np

FEATURES = [
    'lat',
    'lon',
    'speed_over_ground_knots',
    'course_over_ground_deg',
    'computed_speed_knots',
    'acceleration_knots_per_sec',
    'heading_change_deg'
]

# Load the combined training sequences
file_path = r"X_train_full.npy"

print("Loading training sequences...")
X_train_full = np.load(file_path)

print("Shape:", X_train_full.shape)
print("Dtype:", X_train_full.dtype)

# Compute mean and std from training set
mean = X_train_full.mean(axis=(0, 1))
std = X_train_full.std(axis=(0, 1))

print("\nFeature means:")
for f, m in zip(FEATURES, mean):
    print(f"  {f:30s}: {m:.4f}")

print("\nFeature stds:")
for f, s in zip(FEATURES, std):
    print(f"  {f:30s}: {s:.4f}")

# Normalize
X_train_norm = (X_train_full - mean) / (std + 1e-8)

# Save outputs
np.save(r"C:\Users\Aditya Gaur\Downloads\.vscode\maritime\X_train_norm.npy", X_train_norm)
np.save(r"C:\Users\Aditya Gaur\Downloads\.vscode\maritime\norm_mean.npy", mean)
np.save(r"C:\Users\Aditya Gaur\Downloads\.vscode\maritime\norm_std.npy", std)

print(f"\nNormalized training shape: {X_train_norm.shape}")
print("Normalization complete.")