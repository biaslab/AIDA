using ForneyLab
using LinearAlgebra

βN(x) = x
sumβN(x, y) = x + y
sigmoid(x) = 1/(1+ exp(-(x)))

function init_estimator()
    n_samples = 1
    fg = FactorGraph()

    # State prior
    @RV u_0 ~ GaussianMeanPrecision(placeholder(:m_u_0), placeholder(:w_u_0))

    # Transition and observation model
    ω = Vector{Variable}(undef, n_samples)
    ωsample = Vector{Variable}(undef, n_samples)
    ωn = Vector{Variable}(undef, n_samples)
    p = Vector{Variable}(undef, n_samples)
    u = Vector{Variable}(undef, n_samples)
    un = Vector{Variable}(undef, n_samples)
    utr = Vector{Variable}(undef, n_samples)
    β = Vector{Variable}(undef, n_samples)

    u_i_min = u_0

    for i in 1:n_samples

        @RV ω[i] ~ Beta(1.0, 1.0)
        @RV ωsample[i] ~ Nonlinear{Sampling}(ω[i], g=βN)
        @RV ωn[i] ~ GaussianMeanPrecision(ωsample[i], 100.0)

        @RV utr[i] ~ GaussianMeanPrecision(u_i_min, 10.0)
        @RV u[i] ~ Nonlinear{Sampling}(ωn[i], utr[i], g=sumβN)
        @RV un[i] ~ GaussianMeanPrecision(u[i], 100.0)
        @RV p[i] ~ Nonlinear{Sampling}(un[i], g=sigmoid)
        @RV β[i] ~ Bernoulli(p[i])
        # Data placeholder
        placeholder(β[i], :β, index=i)

        # Reset state for next step
        u_i_min = un[i]
    end
    q = PosteriorFactorization(ω, ωn, [u_0; un], ids=[:Ω :ΩN :U])
    algo = messagePassingAlgorithm(free_energy=true, id=:E)
    source_code = algorithmSourceCode(algo, free_energy=true)
    eval(Meta.parse(source_code))
end

function estimate(n_samples; prediction, target, vmp_iter)
    data = Dict(:β => target, :m_u_0 => prediction[1], :w_u_0 => prediction[2])
    # Initial posterior factors
    marginals = Dict{Symbol, ProbabilityDistribution}()
    for t = 1:n_samples
        marginals[:u_*t] = vague(SampleList)
        marginals[:utr_*t] = vague(GaussianMeanPrecision)
        marginals[:un_*t] = vague(GaussianMeanPrecision)
        marginals[:ω_*t] = vague(GaussianMeanPrecision)
        marginals[:ωn_*t] = vague(GaussianMeanPrecision)
        marginals[:ωsample_*t] = vague(GaussianMeanPrecision)
    end
    fe = []
    for i in 1:vmp_iter
        stepEU!(data, marginals)
        stepEΩ!(data, marginals)
        stepEΩN!(data, marginals)
        push!(fe, freeEnergyE(data, marginals))
    end
    return marginals, fe
end

function init_actor()
    fg = FactorGraph()
    # State prior
    @RV u_0 ~ GaussianMeanPrecision(placeholder(:m_u_0), placeholder(:w_u_0))
    @RV u ~ GaussianMeanPrecision(u_0, 1.0)
    @RV usum = u + placeholder(:ω̂)
    @RV un ~ GaussianMeanPrecision(usum, 100.0)
    @RV p ~ Nonlinear{Sampling}(un, g=sigmoid)
    @RV r ~ Bernoulli(p)

    # Data placeholder
    placeholder(r, :r)

    q = PosteriorFactorization([u_0; u], un, ids=[:U :UN])
    algo = messagePassingAlgorithm(free_energy=true, id=:A)
    source_code = algorithmSourceCode(algo, free_energy=true)
    eval(Meta.parse(source_code))
end

function act(action; feedback, prior, vmp_iter)
    marginals = Dict{Symbol, ProbabilityDistribution}(:un => vague(GaussianMeanPrecision))
    data = Dict(:ω̂ => action, :r=>0.0, :m_u_0 => prior[1], :w_u_0 => prior[2])
    messages_1 = Array{Message}(undef, 6)
    messages_2 = Array{Message}(undef, 4)
    fe = []
    for i in 1:vmp_iter
        stepAU!(data, marginals, messages_1)
        stepAUN!(data, marginals, messages_2)
        push!(fe, freeEnergyA(data, marginals))
    end

    return marginals, fe
end
