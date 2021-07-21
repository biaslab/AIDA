using ForneyLab
using Flux
import Turing

# Turn a vector into a set of weights and biases.
function unpack(nn_params::AbstractVector)
    W1 = reshape(nn_params[1:6], 2, 3);   
    b1 = nn_params[7:8]
    W2 = reshape(nn_params[9:14], 3, 2); 
    b2 = nn_params[14:16]
    Wo = reshape(nn_params[17:19], 1, 3);
    bo = nn_params[20:20]
    return W1, b1, W2, b2, Wo, bo
end

# Construct a neural network using Flux and return a predicted value.
function nn_forward(xs, nn_params::AbstractVector)
    W1, b1, W2, b2, Wo, bo = unpack(nn_params)
    nn = Chain(Dense(W1, b1, tanh),
               Dense(W2, b2, tanh),
               Dense(Wo, bo, σ))
    return nn(xs)
end;

# Specify the probabalistic model.
Turing.@model function prefernce_learning(gs, fdb, reg=3.3)
    # Create the weight and bias vector.
    nn_params ~ Turing.MvNormal(zeros(20), reg .* ones(20))
    
    # Calculate predictions for the inputs given the weights
    # and biases
    preds = nn_forward(gs, nn_params)
    
    # Observe each prediction.
    for i in 1:length(fdb)
        fdb[i] ~ Turing.Bernoulli(preds[i])
    end
end;

# define inference graph
function build_est_graph(n_gs, nn_params, n_samples=1)
    graph = ForneyLab.FactorGraph()

    @RV gs ~ GaussianMeanVariance(
              placeholder(:mu_gs, dims=(n_gs,)), 
              placeholder(:Sigma_gs, dims=(n_gs,n_gs)))


    z = Vector{Variable}(undef, n_samples)
    y = Vector{Variable}(undef, n_samples)
    for n in 1:n_samples
        @eval $(Symbol("func$n"))(x) = nn_forward(x, nn_params)
        @RV z[n] ~ Nonlinear{Sampling}(gs,g=eval(Symbol("func$n")),in_variates=[Multivariate],out_variate=Univariate)
        if n == n_samples
            @RV y[n] ~ GaussianMeanVariance(z[n], 1e-4)
            placeholder(y[n], :y, index=n)
        else
            @RV y[n] ~ Bernoulli(z[n])
            placeholder(y[n], :y, index=n)
        end
    end
    
    return graph
end