from pathlib import Path
import numpy as np
from scipy.linalg import solve_triangular

# locate repository root
ROOT = Path(__file__).resolve().parents[2]

# load common inputs
train = np.loadtxt(ROOT / "data/train.csv", delimiter=",")
test = np.loadtxt(ROOT / "data/test.csv", delimiter=",")

X_train, y_train = train[:, :-1], train[:, -1]
X_test, y_test = test[:, :-1], test[:, -1]


# ridge regression using Cholesky factorisation
def fit_ridge(X, y, lam):
    A = X.T @ X + lam * np.eye(X.shape[1])
    b = X.T @ y

    # Cholesky factorisation: A = L @ L.T
    L = np.linalg.cholesky(A)

    # solve two triangular systems
    z = solve_triangular(L, b, lower=True)
    beta = solve_triangular(L.T, z, lower=False)

    return beta


# fit model and make predictions
lam = 1.0
beta = fit_ridge(X_train, y_train, lam)
predictions = X_test @ beta

rmse = np.sqrt(np.mean((y_test - predictions) ** 2))

# validate the solution against the regularised normal equations
residual = X_train.T @ (X_train @ beta - y_train) + lam * beta
assert np.linalg.norm(residual) < 1e-8

# save outputs
output = ROOT / "results"
output.mkdir(exist_ok=True)

np.savetxt(output / "python_coefficients.csv", beta, fmt="%.17g")
np.savetxt(output / "python_predictions.csv", predictions, fmt="%.17g")

print(f"test RMSE: {rmse:.6f}")
print(f"normal equation residual: {np.linalg.norm(residual):.2e}")
print("complete")
