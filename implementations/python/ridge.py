from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
# load common inputs
train = np.loadtxt(ROOT / "data/train.csv", delimiter=",")
test = np.loadtxt(ROOT / "data/test.csv", delimiter=",")

X_train, y_train = train[:, :-1], train[:, -1]
X_test, y_test = test[:, :-1], test[:, -1]

lam = 1.0


def fit_ridge(X, y, lam):
    A = X.T @ X + lam * np.eye(X.shape[1])
    b = X.T @ y

    # Cholesky factorisation: A = L @ L.T
    L = np.linalg.cholesky(A)

    # Solve two triangular systems
    z = np.linalg.solve(L, b)
    beta = np.linalg.solve(L.T, z)

    return beta


# fit model and make predictions
beta = fit_ridge(X_train, y_train, lam)
predictions = X_test @ beta

rmse = np.sqrt(np.mean((y_test - predictions) ** 2))

# check that the fitted coefficients satisfy the regularised normal equations
residual = (
    X_train.T @ X_train + lam * np.eye(X_train.shape[1])
) @ beta - X_train.T @ y_train

assert np.linalg.norm(residual) < 1e-8

# save outputs
output = ROOT / "results"
output.mkdir(exist_ok=True)

np.savetxt(output / "python_coefficients.csv", beta, fmt="%.17g")
np.savetxt(output / "python_predictions.csv", predictions, fmt="%.17g")

print(f"test RMSE: {rmse:.6f}")
print(f"normal-equation residual: {np.linalg.norm(residual):.2e}")
print("complete")
