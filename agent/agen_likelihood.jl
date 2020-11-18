using ForneyLab
using LinearAlgebra
using ProgressMeter

regress(x, θ) = 1/(1+ exp(-(x'θ)))
sigmoid(x) = 1/(1+ exp(-(x)))
σ(x) = 1/(1+ exp(-(x)))
σplus(x, y) = 1/(1+ exp(-((x + y))))

function learner()
    n_samples = 1
end

function learner()

end

function agent_init()
    fg = FactorGraph()

    # State prior
    @RV ω_0 ~ GaussianMeanPrecision(placeholder(:m_ω_0), placeholder(:w_ω_0))
    @RV θ ~ GaussianMeanPrecision(placeholder(:θ̂), huge)

    @RV u ~ GaussianMeanPrecision(placeholder(:m_u), placeholder(:w_u))
    @RV us ~ Nonlinear{Sampling}(u, g=σ)
    @RV un ~ GaussianMeanPrecision(us, 1e2)
    @RV ωs ~ Nonlinear{Sampling}(ω_0, un, g=σplus)
    @RV ωn ~ GaussianMeanPrecision(ωs, 1e2)
    @RV r ~ Nonlinear{Sampling}(ωn, θ, g=regress)
    @RV rf ~ GaussianMeanPrecision(r, placeholder(:w_rf))
    # Data placeholder
    placeholder(rf, :rf)

    # Reset state for next step
    q = PosteriorFactorization(ωs, us, ωn, ids=[:ΩS :US :ΩN])
    algo = messagePassingAlgorithm(free_energy=true)
    source_code = algorithmSourceCode(algo, free_energy=true)
end

fg = FactorGraph()

# State prior
@RV ω_0 ~ GaussianMeanPrecision(placeholder(:m_ω_0), placeholder(:w_ω_0))
@RV θ ~ GaussianMeanPrecision(placeholder(:m_θ), placeholder(:w_θ))

@RV u ~ GaussianMeanPrecision(placeholder(:m_u), placeholder(:w_u))
@RV us ~ Nonlinear{Sampling}(u, g=σ)
@RV un ~ GaussianMeanPrecision(us, placeholder(:w_un))
@RV ωs ~ Nonlinear{Sampling}(ω_0, un, g=σplus)
@RV ωn ~ GaussianMeanPrecision(ωs, 1e5)
@RV r ~ Nonlinear{Sampling}(ωn, θ, g=regress)
@RV rf ~ GaussianMeanPrecision(r, placeholder(:w_rf))
# Data placeholder
placeholder(rf, :rf)

# Reset state for next step
q = PosteriorFactorization(ωs, us, ωn, ids=[:ΩS :US :ΩN])
algo = messagePassingAlgorithm(free_energy=true)
source_code = algorithmSourceCode(algo, free_energy=true)

#code = estimator()
eval(Meta.parse(source_code))

marginals = Dict()
fe = []
priors = Dict(:m_ω_0 => 0.5, :w_ω_0 => 1.0,
              :rf => 1.0, :w_rf => 1.0,
              :m_u => 0.0, :w_u => 1.0,
              :m_θ =>0.6, :w_θ => 1.0,
              :w_un => 1e5)

marginals = Dict{Symbol, ProbabilityDistribution}()
marginals[:us] = vague(SampleList)
marginals[:u] = ProbabilityDistribution(Univariate, GaussianMeanPrecision, m=0.0, w=1.0)
marginals[:ω] =  ProbabilityDistribution(Univariate, GaussianMeanPrecision, m=0.0, w=1.0)
marginals[:ωn] =  ProbabilityDistribution(Univariate, GaussianMeanPrecision, m=0.0, w=1.0)
marginals[:r] = vague(SampleList)

fe = []
messages = initΩN()
@showprogress for i in 1:10
    stepΩS!(priors, marginals)
    stepΩN!(priors, marginals)
    stepUS!(priors, marginals)
    push!(fe, freeEnergy(priors, marginals))
end



import Distributions.Uniform
function generate_data(n_samples, θ_0)
    d = Uniform(0, 1)
    ω = rand(d, n_samples)
    logistic(x, θ) = 1/(1+ exp(-(x'θ)))
    r = logistic.(ω, θ_0*ones(n_samples))
    return ω, r
end

inputs, outputs = generate_data(10000, 0.6)

scatter(inputs, outputs)

using Flux
import Flux: @epochs
using Flux: Data.DataLoader

m = Chain(Dense(1, 1, σ))


loss(x, y) = Flux.mse(m(x), y)

function loss_all(dataloader)
    l = 0f0
    for (x,y) in dataloader
        l += loss(x, y)
    end
    l/length(dataloader)
end

ps = Flux.params(m)

opt = Descent(0.1)

train_data = DataLoader(inputs, outputs)
evalcb = () -> @show(loss_all(train_data))
@epochs 100 Flux.train!(loss, ps, train_data, opt)
