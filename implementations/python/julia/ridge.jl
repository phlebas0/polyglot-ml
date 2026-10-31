using LinearAlgebra
using DelimitedFiles
using Statistics
using Printf

# locate repository root
root = normpath(joinpath(@__DIR__, "..", ".."))

# load common inputs
train = readdlm(joinpath(root, "data", "train.csv"), ',', Float64)
test = readdlm(joinpath(root, "data", "test.csv"), ',', Float64)

X_train, y_train = train[:, 1:(end-1)], train[:, end]
X_test, y_test = test[:, 1:(end-1)], test[:, end]


# ridge regression using Cholesky factorisation
function fit_ridge(X, y, lam)
    A = X' * X + lam * I
    b = X' * y

    # Cholesky factorisation: A = L * L'
    L = cholesky(Symmetric(A)).L

    # solve two triangular systems
    z = L \ b
    beta = L' \ z

    return beta
end


# fit model and make predictions
lam = 1.0
beta = fit_ridge(X_train, y_train, lam)
predictions = X_test * beta

rmse = sqrt(mean((y_test - predictions) .^ 2))

# validate the solution against the regularised normal equations
residual = X_train' * (X_train * beta - y_train) + lam * beta
@assert norm(residual) < 1e-8

# compare against the Python implementation
python_beta = vec(readdlm(
    joinpath(root, "results", "python_coefficients.csv"),
    ',', Float64
))

python_predictions = vec(readdlm(
    joinpath(root, "results", "python_predictions.csv"),
    ',', Float64
))

coefficient_diff = maximum(abs.(beta - python_beta))
prediction_diff = maximum(abs.(predictions - python_predictions))

@assert coefficient_diff < 1e-9
@assert prediction_diff < 1e-9

# save outputs
output = joinpath(root, "results")
mkpath(output)

writedlm(joinpath(output, "julia_coefficients.csv"), beta, ',')
writedlm(joinpath(output, "julia_predictions.csv"), predictions, ',')

@printf("test RMSE: %.6f\n", rmse)
@printf("normal equation residual: %.2e\n", norm(residual))
@printf("maximum coefficient difference: %.2e\n", coefficient_diff)
@printf("maximum prediction difference: %.2e\n", prediction_diff)
println("complete")
