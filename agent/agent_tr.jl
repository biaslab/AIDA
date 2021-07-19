using Turing
using Flux
using Plots

# Turn a vector into a set of weights and biases.
function unpack(nn_params::AbstractVector)
    W1 = reshape(nn_params[1:6], 3, 2);   
    b1 = nn_params[7:9]
    
    W2 = reshape(nn_params[10:15], 2, 3); 
    b2 = nn_params[16:17]
    
    Wo = reshape(nn_params[18:19], 1, 2); 
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
@model function prefernce_learning(gs, fdb, reg=3.3)
    # Create the weight and bias vector.
    nn_params ~ MvNormal(zeros(20), reg .* ones(20))
    
    # Calculate predictions for the inputs given the weights
    # and biases
    preds = nn_forward(gs, nn_params)
    
    # Observe each prediction.
    for i in 1:length(fdb)
        fdb[i] ~ Bernoulli(preds[i])
    end
end;