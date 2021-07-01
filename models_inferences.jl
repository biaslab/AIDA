using Revise
using Rocket
using ReactiveMP
using GraphPPL
using Distributions
using LinearAlgebra
import ProgressMeter
using WAV
using Plots

# Extension for Addition node 
@marginalrule typeof(+)(:in1_in2) (m_out::UnivariateNormalDistributionsFamily, m_in1::UnivariateNormalDistributionsFamily, m_in2::UnivariateNormalDistributionsFamily) = begin
    xi_out = weightedmean(m_out)
    W_out  = precision(m_out)
    xi_in1 = weightedmean(m_in1)
    W_in1  = precision(m_in1)
    xi_in2 = weightedmean(m_in2)
    W_in2  = precision(m_in2)
    
    return MvNormalWeightedMeanPrecision([xi_in1+xi_out; xi_in2+xi_out], [W_in1+W_out W_out; W_out W_in2+W_out])
end

@rule typeof(+)(:in1, Marginalisation) (m_out::UnivariateNormalDistributionsFamily, m_in2::UnivariateNormalDistributionsFamily) = begin
    return NormalMeanVariance(mean(m_out) - mean(m_in2), var(m_out) + var(m_in2))
end
@rule typeof(+)(:in2, Marginalisation) (m_out::UnivariateNormalDistributionsFamily, m_in1::UnivariateNormalDistributionsFamily) = begin
    return NormalMeanVariance(mean(m_out) - mean(m_in1), var(m_out) + var(m_in1))
end

# "pure" AR model
@model function ar_model(n, order)

    x = datavar(Vector{Float64}, n)
    y = datavar(Float64, n)

    γ ~ GammaShapeRate(1.0, 1.0) where {q=MeanField()}
    θ ~ MvNormalMeanPrecision(zeros(order), Matrix{Float64}(I, order, order)) where {q=MeanField()}


    for i in 1:n
        y[i] ~ NormalMeanPrecision(dot(x[i], θ), γ) where {q=MeanField()}
    end

    return x, y, θ, γ
end
# AR inference
function inference_ar(inputs, outputs, order, niter)
    n = length(outputs)
    model, (x, y, θ, γ) = ar_model(n, order, options = (limit_stack_depth = 500, ))

    γ_buffer = nothing
    θ_buffer = nothing
    fe = Vector{Float64}()

    γsub = subscribe!(getmarginal(γ), (my) -> γ_buffer = my)
    θsub = subscribe!(getmarginal(θ), (mθ) -> θ_buffer = mθ)
    fesub = subscribe!(score(Float64, BetheFreeEnergy(), model), (f) -> push!(fe, f))

    setmarginal!(γ, GammaShapeRate(1.0, 1.0))

    ProgressMeter.@showprogress for i in 1:niter
        update!(x, inputs)
        update!(y, outputs)
    end


    return γ_buffer, θ_buffer, fe
end


# Auxilary model to infer stationary noise variance
@model function gaussian_model(n)

    y = datavar(Float64, n)

    γ   ~ GammaShapeRate(1.0, 1.0) where {q=MeanField()}
    x_0 ~ NormalMeanPrecision(0.0, 1.0) where {q=MeanField()}
    x   ~ NormalMeanPrecision(x_0, γ) where {q=MeanField()}

    for i in 1:n
        y[i] ~ NormalMeanPrecision(x, 1e4) where {q=MeanField()}
    end

    return y, x, γ
end
# Gaussian inference
function inference_gaussian(outputs, niter)
    n = length(outputs)
    model, (y, x, γ) = gaussian_model(n, options = (limit_stack_depth = 500, ))
    γ_buffer = nothing
    x_buffer = nothing
    fe = Vector{Float64}()

    γsub = subscribe!(getmarginal(γ), (my) -> γ_buffer = my)
    xsub = subscribe!(getmarginal(x), (mx) -> x_buffer = mx)
    fesub = subscribe!(score(Float64, BetheFreeEnergy(), model), (f) -> push!(fe, f))

    setmarginal!(γ, GammaShapeRate(1.0, 1.0))
    setmarginal!(x, NormalMeanPrecision(0.0, 1.0))

    ProgressMeter.@showprogress for i in 1:niter
        update!(y, outputs)
    end

    unsubscribe!(γsub)
    unsubscribe!(xsub)
    unsubscribe!(fesub)

    return x_buffer, γ_buffer, fe
end

# classical latent AR model
@model function lar_model(n, order, artype, c, τ)

    x = randomvar(n)
    y = datavar(Float64, n)
    ct  = constvar(c)

    γ ~ GammaShapeRate(1.0, 1e-4) where {q=MeanField()}
    θ ~ MvNormalMeanPrecision(randn(order), Matrix{Float64}(I, order, order)) where {q=MeanField()}
    x0 ~ MvNormalMeanPrecision(zeros(order), Matrix{Float64}(I, order, order)) where {q=MeanField()}

    x_prev = x0

    ar_nodes = Vector{FactorNode}(undef, n)

    for i in 1:n
        ar_nodes[i], x[i] ~ AR(x_prev, θ, γ) where { q = q(y, x)q(γ)q(θ), meta = ARMeta(artype, order, ARsafe()) }

        y[i] ~ NormalMeanVariance(dot(ct, x[i]), τ)

        x_prev = x[i]
    end

    return y, x, θ, γ, ar_nodes
end

function lar_inference(data, order, niter, τ)
    n = length(data)
    artype = Multivariate
    c = zeros(order); c[1] = 1.0
    model, (y, x, θ, γ, ar_nodes) = lar_model(n, order, artype, c, τ)

    γ_buffer = nothing
    θ_buffer = nothing
    x_buffer = Vector{Marginal}(undef, n)
    fe = Vector{Float64}()

    γsub = subscribe!(getmarginal(γ), (mγ) -> γ_buffer = mγ)
    θsub = subscribe!(getmarginal(θ), (mθ) -> θ_buffer = mθ)
    xsub = subscribe!(getmarginals(x), (mx) -> copyto!(x_buffer, mx))
    fesub = subscribe!(score(Float64, BetheFreeEnergy(), model), (f) -> push!(fe, f))

    setmarginal!(γ, GammaShapeRate(1.0, 1e-4))
    setmarginal!(θ, MvNormalMeanPrecision(zeros(order), Matrix{Float64}(I, order, order)))

    for i in 1:n
        setmarginal!(ar_nodes[i], :y_x, MvNormalMeanPrecision(zeros(2*order), Matrix{Float64}(I, 2*order, 2*order)))
    end
    for i in 1:niter
        update!(y, data)
    end
    return γ_buffer, θ_buffer, x_buffer, fe
end

# to optimize
function lar_batch_learning(segments, ar_order, vmp_its, τ)
    totseg = size(segments, 1)
    l      = size(segments, 2)
    
    rmx = zeros(totseg, l)
    rvx = zeros(totseg, l)
    rmθ = zeros(totseg, ar_order)
    rvθ = zeros(totseg, ar_order, ar_order)
    rγ = fill(tuple(.0, .0), totseg)
    fe  = zeros(totseg, vmp_its)
    
    ProgressMeter.@showprogress for segnum in 1:totseg
        γ, θ, xs, fe[segnum, :] = lar_inference(segments[segnum, :], ar_order, vmp_its, τ)

        mx, vx                            = mean.(xs), cov.(xs)
        mθ, vθ                            = mean(θ), cov(θ)
        rmx[segnum, :], rvx[segnum, :]    = first.(mx), first.(vx)
        rmθ[segnum, :], rvθ[segnum, :, :] = mθ, vθ
        rγ[segnum]                        = shape(γ), rate(γ)
    end
    rmx, rvx, rmθ, rvθ, rγ
end

# Coupled AR auxilary piplelines
struct InitARMessages{M} <: ReactiveMP.AbstractPipelineStage 
    message :: M
end
ReactiveMP.apply_pipeline_stage(stage::InitARMessages, factornode, tag::Type{Val{:x}}, stream) = stream |> start_with(Message(stage.message, false, true))
ReactiveMP.apply_pipeline_stage(stage::InitARMessages, factornode, tag::Type{Val{:y}}, stream) = stream |> start_with(Message(stage.message, false, true))
ReactiveMP.apply_pipeline_stage(stage::InitARMessages, factornode, tag, stream)                = stream

# Coupled AR model
@model function coupled_model(n, prior_η, prior_τ, order_1, order_2, artype, c1, c2)

    # z for ar_1
    z  = randomvar(n)
    z1 = randomvar(n)
    # x for ar_2
    x  = randomvar(n)
    x1 = randomvar(n)
    o = datavar(Float64, n)
    
    ct1  = constvar(c1)
    ct2  = constvar(c2)
    
    # AR_1
    γ ~ GammaShapeRate(1.0, 1e-4) where {q=MeanField()}
    θ ~ MvNormalMeanPrecision(zeros(order_1), Matrix{Float64}(I, order_1, order_1)) where {q=MeanField()}
    z0 ~ MvNormalMeanPrecision(zeros(order_1), Matrix{Float64}(I, order_1, order_1)) where {q=MeanField()}
    
    # AR_2
    τ ~ GammaShapeRate(prior_τ[1], prior_τ[2]) where {q=MeanField()}
#     η ~ MvNormalMeanPrecision(prior_η[1], 1e12*Matrix{Float64}(I, order_2, order_2)) where {q=MeanField()}
    η ~ MvNormalMeanPrecision(prior_η[1], prior_η[2]) where {q=MeanField()}
    x0 ~ MvNormalMeanPrecision(zeros(order_2), 1e2*Matrix{Float64}(I, order_2, order_2)) where {q=MeanField()}

    z_prev = z0
    x_prev = x0

    ar_1_nodes = Vector{FactorNode}(undef, n)
    ar_2_nodes = Vector{FactorNode}(undef, n)

    for i in 1:n
        ar_1_nodes[i], z[i] ~ AR(z_prev, θ, γ) where { 
            q = q(y, x)q(γ)q(θ), meta = ARMeta(artype, order_1, ARsafe()), 
            pipeline = InitARMessages(MvNormalMeanPrecision(zeros(order_1), Matrix{Float64}(I, order_1, order_1)))
        }
        z1[i] ~ dot(ct1, z[i])
        
        ar_2_nodes[i], x[i] ~ AR(x_prev, η, τ) where { 
            q = q(y, x)q(γ)q(θ), meta = ARMeta(artype, order_2, ARsafe()) ,
            pipeline = InitARMessages(MvNormalMeanPrecision(zeros(order_2), Matrix{Float64}(I, order_2, order_2)))
        }
        x1[i] ~ dot(ct2, x[i])
        
        o[i] ~ NormalMeanVariance(z1[i] + x1[i], 1e-12)

        x_prev = x[i]
        z_prev = z[i]
    end
    
    scheduler = schedule_updates(x, θ, γ, z, η, τ)

    return o, z, z1, θ, γ, x, x1, η, τ, ar_1_nodes, ar_2_nodes, scheduler
end

# Coupled AR inference
function coupled_inference(data, prior_η, prior_τ, order_1, order_2, niter)
    n = length(data)
    artype = Multivariate
    c1 = zeros(order_1); c1[1] = 1.0
    c2 = zeros(order_2); c2[1] = 1.0
    model, (o, z, z1, θ, γ, x, x1, η, τ, ar_1_nodes, ar_2_nodes, scheduler) = coupled_model(n, prior_η, prior_τ, order_1, order_2, artype, c1, c2, options=(limit_stack_depth=100, ))

    γ_buffer = nothing
    θ_buffer = nothing
    x_buffer = Vector{Marginal}(undef, n)
    
    τ_buffer = nothing
    η_buffer = nothing
    z_buffer = Vector{Marginal}(undef, n)
    fe = Vector{Float64}()

    γsub = subscribe!(getmarginal(γ), (mγ) -> γ_buffer = mγ)
    θsub = subscribe!(getmarginal(θ), (mθ) -> θ_buffer = mθ)
    xsub = subscribe!(getmarginals(x), (mx) -> copyto!(x_buffer, mx))
    
    τsub = subscribe!(getmarginal(τ), (mτ) -> τ_buffer = mτ)
    ηsub = subscribe!(getmarginal(η), (mη) -> η_buffer = mη)
    zsub = subscribe!(getmarginals(z), (mz) -> copyto!(z_buffer, mz))
    
    fe_scheduler = PendingScheduler()
    fesub = subscribe!(score(Float64, BetheFreeEnergy(), model, fe_scheduler), (f) -> push!(fe, f))
    setmarginal!(γ, GammaShapeRate(1e-12, 1e-12))
    setmarginal!(θ, MvNormalMeanPrecision(zeros(order_1), Matrix{Float64}(I, order_1, order_1)))
    
    setmarginal!(τ, GammaShapeRate(prior_τ[1], prior_τ[2]))
#     setmarginal!(η, MvNormalMeanPrecision(prior_η[1], prior_η[2])) # better
    setmarginal!(η, MvNormalMeanPrecision(prior_η[1],  1e4*Matrix{Float64}(I, order_2, order_2)))

    for i in 1:n
#         setmarginal!(x1[i], NormalMeanPrecision(0.0, 1.0))
#         setmarginal!(z1[i], NormalMeanPrecision(0.0, 1.0))
        setmarginal!(ar_1_nodes[i], :y_x, MvNormalMeanPrecision(zeros(2*order_1), Matrix{Float64}(I, 2*order_1, 2*order_1)))
        setmarginal!(ar_2_nodes[i], :y_x, MvNormalMeanPrecision(zeros(2*order_2), Matrix{Float64}(I, 2*order_2, 2*order_2)))
    end
    
    for i in 1:niter
        update!(o, data)
        # iffy approach
        for i in 1:n
            release!(scheduler)
        end
        release!(fe_scheduler)
    end
    return γ_buffer, θ_buffer, z_buffer, τ_buffer, η_buffer, x_buffer, fe
end

function batch_coupled_learning(segments, priors_η, priors_τ, ar_1_order, ar_2_order, vmp_its)
    totseg = size(segments, 1)
    l      = size(segments, 2)
    rmx = zeros(totseg, l)
    rvx = zeros(totseg, l)
    rmθ = zeros(totseg, ar_1_order)
    rvθ = zeros(totseg, ar_1_order, ar_1_order)
    rγ = fill(tuple(.0, .0), totseg)
    
    rmz = zeros(totseg, l)
    rvz = zeros(totseg, l)
    rmη = zeros(totseg, ar_2_order)
    rvη = zeros(totseg, ar_2_order, ar_2_order)
    rτ = fill(tuple(.0, .0), totseg)
    
    fe  = zeros(totseg, vmp_its)
    
    ProgressMeter.@showprogress for segnum in 1:totseg
        prior_η                           = (priors_η[1][segnum, :], priors_η[2][segnum, :, :])
        prior_τ                           = priors_τ[segnum]
        γ, θ, zs, τ, η, xs, fe[segnum, :] = coupled_inference(segments[segnum, :], prior_η, prior_τ, ar_1_order, ar_2_order, vmp_its)
        mz, vz                            = mean.(zs), cov.(zs)
        mθ, vθ                            = mean(θ), cov(θ)
        rmz[segnum, :], rvz[segnum, :]    = first.(mz), first.(vz)
        rmθ[segnum, :], rvθ[segnum, :, :] = mθ, vθ
        rγ[segnum]                        = shape(γ), rate(γ)
        
        mx, vx                            = mean.(xs), cov.(xs)
        mη, vη                            = mean(η), cov(η)
        rmx[segnum, :], rvx[segnum, :]    = first.(mx), first.(vx)
        rmη[segnum, :], rvη[segnum, :, :] = mη, vη
        rτ[segnum]                        = shape(τ), rate(τ)
    end
    rmz, rvz, rmθ, rvθ, rγ, rmx, rvx, rmη, rvη, rτ, fe
end