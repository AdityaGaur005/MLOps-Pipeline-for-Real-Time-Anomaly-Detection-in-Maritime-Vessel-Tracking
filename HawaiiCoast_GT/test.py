import numpy as np

X = np.load(r"C:\Users\Aditya Gaur\Downloads\.vscode\maritime\HawaiiCoast_GT\X_train_full.npy")

print("Shape:", X.shape)
print("Lat/Lon correlation:",
      np.corrcoef(X[:, :, 0].flatten(), X[:, :, 1].flatten())[0, 1])
print("Exact duplicate:",
      np.array_equal(X[:, :, 0], X[:, :, 1]))
