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
function nn_forward(xs::AbstractArray, nn_params::AbstractVector)
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


# Specify the probabalistic model.
Turing.@model function planning(nn_params, fdb, context)
    # Create the weight and bias vector.
    gs ~ Turing.MvNormal([1.0, 1.0, context], [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1e-5])
    
    # Calculate predictions for the inputs given the weights
    # and biases
    preds = nn_forward(gs, nn_params)
    
    # Observe each prediction.
    for i in 1:length(fdb)
        fdb[i] ~ Turing.Bernoulli(preds[i])
    end
end;


# Returns paramaters of a neural netwrok that provides the highest log posterior.
function learning(gains, contexts, appraisals, N=1000; g_jitter=1e-4)
    # adding jitter to gains might be useful when dealing with small number of observations
    gains = [g .+ sqrt(g_jitter)*randn(length(g)) for g in gains]
    inputs = [vcat(g, c) for (g, c) in zip(gains, contexts)]
    ch = Turing.sample(prefernce_learning(hcat(inputs...), appraisals, 1.0), Turing.HMC(0.05, 10), N);
    # Extract all weight and bias parameters.
    theta = Turing.MCMCChains.group(ch, :nn_params).value;
    # Find the index that provided the highest log posterior in the chain.
    _, i = findmax(ch[:lp])
    i = i.I[1]
    θ = Float64.(theta[i, :])
end

# nn_params = learning(gains, contexts, appraisals);

# # Alternative (in case you don't want to use learning function)
# gains₊ = [g .+ sqrt(1e-4)*randn(length(g)) for g in gains]
# inputs = [vcat(g, c) for (g, c) in zip(gains, contexts)]
# ch = Turing.sample(prefernce_learning(hcat(inputs...), appraisals, 1.0), Turing.HMC(0.05, 10), 1000);


# sumstats = Turing.summarize(ch, Turing.mean, Turing.std)
# mθ, vθ   = sumstats.nt.mean, sumstats.nt.std;

# theta = Turing.MCMCChains.group(ch, :nn_params).value;
# # Find the index that provided the highest log posterior in the chain.
# _, i = findmax(ch[:lp])
# i = i.I[1]
# θ = theta[i, :];

# predictions = []
# for n in 1:length(appraisals)
#     push!(predictions, nn_forward(inputs[n], θ))
# end

# # Check for errors on training set
# appraisals_ = [rand(Distributions.Bernoulli(p[1])) for p in predictions]
# err = sum(abs.(appraisals .- appraisals_))


# infer gains
# function sample_gains(context, nn_params, N=1000)
#     ch = Turing.sample(planning(nn_params, [1.0], context), Turing.HMC(0.01, 4), N);
#     gains_context = Turing.MCMCChains.group(ch, :gs).value
#     # Find the index that provided the highest log posterior in the chain.
#     _, i = findmax(ch[:lp])
#     i = i.I[1]
#     Float64.(gains_context[i, :])
# end