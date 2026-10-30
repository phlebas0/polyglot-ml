from pathlib import Path
import numpy as np

# synthetic regression dataset
rng = np.random.default_rng(42)

n_train = 1000
n_test = 200
n_features = 20

X_train = rng.normal(size=(n_train, n_features))
X_test = rng.normal(size=(n_test, n_features))

# regression coefficients
beta_true = rng.normal(size=n_features)

y_train = X_train @ beta_true + rng.normal(scale=0.5, size=n_train)
y_test = X_test @ beta_true + rng.normal(scale=0.5, size=n_test)

# export identical datasets for every language
data_dir = Path(__file__).resolve().parent

np.savetxt(
    data_dir / "train.csv",
    np.column_stack((X_train, y_train)),
    delimiter=",",
    fmt="%.17g",
)

np.savetxt(
    data_dir / "test.csv",
    np.column_stack((X_test, y_test)),
    delimiter=",",
    fmt="%.17g",
)

print("complete.")
