using LinearAlgebra,Optim
using Distributions: Bernoulli
using SpecialFunctions: erf

# CDF of 1D Gaussian
ϕ(μ,σ) = 0.5( 1. + (erf( μ / (σ * √(2)) )))

# Binary entropy. Epsilons for stability
h(p) = -p*log(p+eps()) - (1-p)*log(1-p + eps())

# Approximate conditional entropy term
C = √((π*log(2)/2))
H(μ,σ) = (C / √(σ + C^2))* exp(-0.5 * (μ^2 / (σ + C^2)))

# BALD objective. Sometimes we need to dispatch on tuples
BALD(μ,σ) = h(ϕ(μ,σ)) - H(μ,σ)
BALD((μ,σ)) = h(ϕ(μ,σ)) - H(μ,σ)

# KL between Bernoullis. Goal prior has param set to 1 - eps(). eps for stability
KL(p,q) = p * log(eps() + p/q) + (1-p) * log(eps()+ (1-p)/(1-q))

# Squared exponential kernel for multivariate inputs. Thanks Hoang!
function se_kernel(x1,x2,σ,l)
    K = sum(norm.(x1,2).^2,dims=1)' .- 2*x1'x2 .+ sum(norm.(x2,2).^2,dims=1);
    K =  σ^2*exp.(-K/(2*l^2));
end

# Prediction from GP classifier
function predict(x1,x2,y1,σ,l)
    n = size(x1,2)
    Σ_11 = se_kernel(x1, x1, σ, l) + I(n)*0.1
    Σ_22 = se_kernel(x2, x2, σ, l)
    Σ_12 = se_kernel(x1, x2, σ, l)

    chol11 = cholesky(Hermitian(Σ_11))

    # Rasmussen tricks for stability and efficiency
    solved = chol11.U\(chol11.L\Σ_12)

    μ_star = find_μ(Σ_11,x1,y1)
    μ_pred = Σ_12' * (y1 - sigmoid.(μ_star))
    Σ_pred = Σ_22 - Σ_12' * inv(inv(W(μ_star)) + Σ_11) * Σ_12
    ## Add epsilon to prevent 0 variances
    #Σ_pred= Hermitian(Σ_22 - solved' * Σ_12) .+ I(size(x2,2)) * 1e-6
    #μ_pred = solved' * y1
    return μ_pred, Σ_pred
end

# Newton optimization to find the mean of the latent function
function find_μ(Σ,x,y)
    # W is of the proposed mean. Because we can get the precision as the Hessian of the log likelihood
    iter = 10
    # Make outputs -1/1 instead of 0/1
    y_scaled = (y .- 0.5) * 2.
    μ_test = zeros(size(y,1))
    for i in 1:iter # Need to make a check for convergence here
        μ_test = Σ*inv(I(size(y,1)) + W(μ_test) * Σ) * ( y_scaled - sigmoid.(μ_test) + W(μ_test)*μ_test)
    end
    μ_test
end


# Help function to construct W matrix
function W(x)
    d = sigmoid.(x) .* (1 .- sigmoid.(x))
    Diagonal(d[:])
end


# Compute EFE at a point
# xc is a tuple since it comes from a grid made with Iterators.product
function choose_point(x1,xc,y1,σ,l)
    x2 = collect(xc)
    μ_pred,Σ_pred =predict(x1,x2,y1,σ,l)
    -BALD(μ_pred[1],Σ_pred[1]) + KL(ϕ(μ_pred[1],Σ_pred[1]),1-eps()) + h(ϕ(μ_pred[1],Σ_pred[1]))
end

# Marginal likelihood (log evidence) for hyper param optimization
function log_evidence(x1,y1,θ)
    σ,l = θ
    n = size(x1,2)

    Σ_11 = se_kernel(x1, x1, σ, l) + I(n) * 0.1
    μ_star = find_μ(Σ_11,x1,y1)

    ll =  -0.5 * μ_star' * inv(Σ_11) * μ_star- 0.5 * logdet(Σ_11) - 0.5 * logdet(W(μ_star) + inv(Σ_11)) + y1' * μ_star -sum(log.(1 .+ exp.(μ_star)))
    -ll
end

function sigmoid(x)
    1 / (1 + exp(-x))
end


# Optimize hyperparams of SE kernel
function optimize_hyperparams(x1,y1,θ)
    params = optimize(θ -> log_evidence(x1,y1,θ), θ)
    σ,l = params.minimizer
end


# Grid search over EFE values with inhibition of return, inspired by eye movements
function get_new_proposal(grid,x1,y1,current,σ,l)
    # Compute the EFE grid
    value_grid = choose_point.(Ref(x1),grid,Ref(y1),σ,l)
    # Ensure that we propose a new trial and not the same one twice in a row
    value_grid[collect(grid) .== [(current[1],current[2])]] .= Inf

    # Find the minimum and try it out
    idx = argmin(value_grid)
    x2 = collect(grid)[idx]
    x2#,value_grid
end
