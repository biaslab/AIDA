using Turing

import Distributions.Uniform
function generate_data(n_samples, θ_0)
    d = Uniform(0, 1)
    ω = rand(d, n_samples)
    logistic(x, θ) = 1/(1+ exp(-(x'θ)))
    r = logistic.(ω, θ_0*ones(n_samples))
    return ω, r
end

θ̂ = 0.6
inputs, outputs = generate_data(10000, θ̂)

logit_θ(x) = 1/(1+ exp(-(θ̂*x)))

# Declare our Turing model.
@model function agent(r)
    # Our prior belief about the probability of heads in a coin.
    u_0 ~ Beta(3, 2)
    ω_0 ~ Normal(0, 1)
    ω_1 = 1/(1+exp(-ω_0-u_0))
    # The number of observations.
    r ~ Normal(logit_θ(ω_1), 0.001)
end

chn = sample(agent(0.5), HMC(0.1, 5), 1000)

 # Define a simple Normal model with unknown mean and variance.
@model function gdemo(x, y)
    s ~ InverseGamma(2, 3)
    m ~ Normal(0, sqrt(s))
    x ~ Normal(m, sqrt(s))
    y ~ Normal(m, sqrt(s))
end

c1 = sample(gdemo(1.5, 2), SMC(), 1000)
с2 = sample(gdemo(1.5, 2), HMC(0.1, 5), 1000)
