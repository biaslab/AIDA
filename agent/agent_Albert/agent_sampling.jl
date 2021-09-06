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