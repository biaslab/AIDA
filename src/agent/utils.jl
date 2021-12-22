using LinearAlgebra, Optim
using Distributions: Bernoulli
using SpecialFunctions: erf
using StatsFuns: normcdf

# CDF of 0 mean, 1D Gaussian
# ϕ(μ, σ) = 0.5(1.0 + (erf(μ / (σ * √(2)))))
ϕ(μ, σ) = normcdf(0.0, σ, μ)
ϕ(μ) = ϕ(μ,1)

# Binary entropy. Epsilons for stability
h(p) = -p * log(p + eps()) - (1 - p) * log(1 - p + eps())

# Approximate conditional entropy term
C = √((π * log(2) / 2))
H(μ, σ) = (C / √(σ^2 + C^2)) * exp(-0.5 * (μ^2 / (σ^2 + C^2)))

# BALD objective. Sometimes we need to dispatch on tuples
BALD(μ, σ) = h(ϕ(μ, σ)) - H(μ, σ)
BALD(θ) = BALD(θ[1][1],θ[2][1])

# Cross entropy
CE(μ,Σ) = KL(ϕ(μ, Σ), 1 - eps()) + h(ϕ(μ, Σ))
CE(θ) = CE(θ[1][1],θ[2][1])

# KL between Bernoullis. Goal prior has param set to 1 - eps(). eps for stability
KL(p, q) = p * log(eps() + p / q) + (1 - p) * log(eps() + (1 - p) / (1 - q))


# Squared exponential kernel for multivariate inputs. Thanks Hoang!
function se_kernel(x1, x2, σ, l)
    K = sum(norm.(x1, 2) .^ 2, dims = 1)' .- 2 * x1'x2 .+ sum(norm.(x2, 2) .^ 2, dims = 1)
    K = σ^2 * exp.(-K / (2 * l^2))
end

# Prediction from GP classifier
function predict(x1, grid_element, y1,Σ_11, σ, l)
    x2 = collect(grid_element)
    #n = size(x1, 2)
    Σ_22 = se_kernel(x2, x2, σ, l)
    Σ_12 = se_kernel(x1, x2, σ, l)

    y_scaled = (y1 .- 0.5) * 2.0

    μ_star = find_μ(Σ_11,y1)
    logp,∇_logp,∇∇_logp = ∇_log_prob(y_scaled,μ_star)
    W_ = Diagonal(-∇∇_logp)

    μ_pred = Σ_12' * ∇_logp
    Σ_pred = Σ_22 - Σ_12' * inv(inv(W_) + Σ_11) * Σ_12

    return μ_pred, Σ_pred
end

# Objective function for finding the mean of the latent function
function objective(a,y_scaled,Σ_11)
    μ = Σ_11'* a
    logp,∇_logp,∇∇_logp = ∇_log_prob(y_scaled,μ)
    0.5*a' * μ - sum(logp)
end

# Gradient based optimization to find the mean of the latent function
function find_μ(Σ, y)
    # Make outputs -1/1 instead of 0/1
    y_scaled = (y .- 0.5) * 2.0

    n = size(y,1)
    μ_test = zeros(n)
    a = zeros(n)

    obj_prev = Inf
    obj_cur = 99999

    cur_iter = 0
    max_iter = 10
    # Based off of Rasmussen algorithm 3.1
    while abs(obj_prev - obj_cur > 1e-6) && cur_iter < max_iter
	logp,∇_logp,∇∇_logp = ∇_log_prob(y_scaled,μ_test)
	W_ = Diagonal( -∇∇_logp)

	L = cholesky(Hermitian(I(n) + sqrt(W_) * Σ * sqrt(W_)))
	b = W_*μ_test + ∇_logp
	Δ_a = b - sqrt(W_) * (L.U \ ( L.L \ ( sqrt(W_) * Σ * b))) - a

    # Linesearch to determine the step size
	γ_star = optimize(γ -> objective(a + γ*Δ_a,y_scaled,Σ), 0.0, 2.0, abs_tol = 1e-4, method = Optim.Brent()).minimizer

	# update the mean
	a = a + γ_star * Δ_a
	μ_test = Σ'* a

	# bookkeeping
	obj_prev = obj_cur
	obj_cur = objective(a,y_scaled,Σ)
	cur_iter += 1
    end
    μ_test
end

# Returns derivatives of log likelihood
function ∇_log_prob(y,μ)
    z = y.*μ # technically divided by σ but we keep σ = 1
    logp = log.(ϕ.(z))
    ∇_logp = (y ./ (sqrt(2. * π))) .* exp.(-0.5*z.*z - logp)
    ∇∇_logp = -∇_logp .* (μ + ∇_logp)
    logp,∇_logp,∇∇_logp
end

# Compute EFE at a point
# xc is a tuple since it comes from a grid made with Iterators.product
function choose_point(x1, xc, y1, σ, l)
    x2 = collect(xc)
    μ_pred, Σ_pred = predict(x1, x2, y1, σ, l)
    -BALD(μ_pred[1], Σ_pred[1]) + KL(ϕ(μ_pred[1], Σ_pred[1]), 1 - eps()) + h(ϕ(μ_pred[1], Σ_pred[1]))
end

# Marginal likelihood (log evidence) for hyper param optimization
function log_evidence(x1, y1, θ)
    σ, l = θ
    n = size(x1, 2)

    y_scaled = (y1 .- 0.5) * 2.0
    Σ_11 = se_kernel(x1, x1, σ, l) + I(n) * 0.1
    μ_star = find_μ(Σ_11, y1)

    logp,∇_logp,∇∇_logp = ∇_log_prob(y_scaled,μ_star)

    W_ = Diagonal( -∇∇_logp)
    L = cholesky(Hermitian(I(n) + sqrt(W_)*sqrt(W_) .* Σ_11))
    b = W_*μ_star + ∇_logp
    a = b - sqrt(W_) * (L.U \ ( L.L \ ( sqrt(W_) * Σ_11 * b)))

    ll = 0.5*a'*μ_star + sum(log.( diag( L.L) - logp))
    -ll
end

# Optimize hyperparams of SE kernel
function optimize_hyperparams(x1, y1, θ)
    # We bound params to ensure the optimization is stable. When the agent only has negative responses
    # available, the optimizer can send parameters to extreme (> 1e19) values.
    params = optimize(θ -> log_evidence(x1, y1, θ), [0.1,0.1],[1.,1],θ,Fminbox(),
		     Optim.Options( iterations=1000, g_tol=1e-4))
    # Note, we increased tolerance on the solver to run faster.
    σ,l = params.minimizer
end

function get_new_decomp(grid, x1, y1, σ, l)
    n = size(x1,2)
    Σ_11 = se_kernel(x1, x1, σ, l) + I(n) * 1e-6
    # Compute the EFE grid
    pred_grid = predict.(Ref(x1), grid, Ref(y1),Ref(Σ_11), σ, l)
    epi_grid = -BALD.(pred_grid)
    inst_grid = CE.(pred_grid)

    epi_grid, inst_grid
end


# Grid search over EFE values with inhibition of return, inspired by eye movements
function get_new_proposal(grid, x1, y1, current, σ, l)
    # Compute the EFE grid
    Σ_11 = se_kernel(x1, x1, σ, l) + I(n) * 1e-6
    epi_grid, inst_grid = get_new_decomp(grid, x1, y1, σ, l)
    value_grid = epi_grid + inst_grid
    # Ensure that we propose a new trial and not the same one twice in a row
    value_grid[collect(grid).==[(current[1], current[2])]] .= Inf

    # Find the minimum and try it out
    idx = argmin(value_grid)
    x2 = collect(grid)[idx]
    x2, value_grid
end

