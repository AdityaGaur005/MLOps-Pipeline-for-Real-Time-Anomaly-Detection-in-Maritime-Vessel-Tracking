import numpy as np
a = np.load(r"C:\Users\Aditya Gaur\Downloads\.vscode\maritime\model\X_train.npy")
b = np.load(r"C:\Users\Aditya Gaur\Downloads\.vscode\maritime\HawaiiCoast_GT\X_train_norm.npy")  # adjust path
c = np.load(r"C:\Users\Aditya Gaur\Downloads\.vscode\maritime\HawaiiCoast_GT\X_train_full.npy")  # adjust path
print(a.shape, b.shape, c.shape)
print(np.allclose(a[:1000], b[:1000]))
print(np.allclose(a[:1000], c[:1000]))
print(np.allclose(b[:1000], c[:1000]))
print(a.mean(), a.std())  # should be ~0, ~1 if normalized
print(b.mean(), b.std())  # should be ~0, ~1 if normalized
print(c.mean(), c.std())  # should be ~0, ~1 if normalized