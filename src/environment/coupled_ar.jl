struct DummyDistribution
end

Distributions.entropy(dist::DummyDistribution) = ReactiveMP.InfCountingReal(0.0, -1)

@marginalrule typeof(+)(:in1_in2) (m_out::PointMass{Float64}, m_in1::NormalMeanVariance{Float64}, m_in2::NormalMeanVariance{Float64}, ) = begin 
    return DummyDistribution()
end

struct InitARMessages{M} <: ReactiveMP.AbstractPipelineStage 
    message :: M
end
ReactiveMP.apply_pipeline_stage(stage::InitARMessages, factornode, tag::Type{Val{:x}}, stream) = stream |> start_with(Message(stage.message, false, true))
ReactiveMP.apply_pipeline_stage(stage::InitARMessages, factornode, tag::Type{Val{:y}}, stream) = stream |> start_with(Message(stage.message, false, true))
ReactiveMP.apply_pipeline_stage(stage::InitARMessages, factornode, tag, stream)                = stream


struct PrintFormConstraint <: ReactiveMP.AbstractFormConstraint
    name :: String
end

ReactiveMP.default_form_check_strategy(::PrintFormConstraint)   = ReactiveMP.FormConstraintCheckLast()

ReactiveMP.is_point_mass_form_constraint(::PrintFormConstraint) = false

function ReactiveMP.constrain_form(constraint::PrintFormConstraint, something)
    println(constraint.name, " ", something)
    return something
end

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
    η ~ MvNormalMeanPrecision(prior_η[1], prior_η[2]) where {q=MeanField()}
    x0 ~ MvNormalMeanPrecision(zeros(order_2), Matrix{Float64}(I, order_2, order_2)) where {q=MeanField()} # synthetic

    z_prev = z0
    x_prev = x0

    ar_1_nodes = Vector{FactorNode}(undef, n)
    ar_2_nodes = Vector{FactorNode}(undef, n)
    
    pipeline_1 = InitARMessages(MvNormalMeanPrecision(zeros(order_1), Matrix{Float64}(I, order_1, order_1)))
    pipeline_2 = InitARMessages(MvNormalMeanPrecision(zeros(order_2), Matrix{Float64}(I, order_2, order_2)))
    
    for i in 1:n
        ar_1_nodes[i], z[i] ~ AR(z_prev, θ, γ) where { 
            q = q(y, x)q(γ)q(θ), meta = ARMeta(artype, order_1, ARsafe()),
            pipeline=pipeline_1
        }
        z1[i] ~ dot(ct1, z[i])
        
        ar_2_nodes[i], x[i] ~ AR(x_prev, η, τ) where { 
            q = q(y, x)q(γ)q(θ), meta = ARMeta(artype, order_2, ARsafe()) ,
            pipeline=pipeline_2
        }
        x1[i] ~ dot(ct2, x[i])
        
        o[i] ~ z1[i] + x1[i]
        x_prev = x[i]
        z_prev = z[i]
    end
    
    scheduler = nothing

    return o, z, z1, θ, γ, x, x1, η, τ, ar_1_nodes, ar_2_nodes, scheduler
end

# Coupled AR inference
function coupled_inference(data, prior_η, prior_τ, order_1, order_2, niter)
    n = length(data)
    artype = Multivariate
    c1 = zeros(order_1); c1[1] = 1.0
    c2 = zeros(order_2); c2[1] = 1.0
    model, (o, z, z1, θ, γ, x, x1, η, τ, ar_1_nodes, ar_2_nodes, scheduler) = coupled_model(model_options(limit_stack_depth=100, ), n, prior_η, prior_τ, order_1, order_2, artype, c1, c2)
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
    
    fesub = subscribe!(score(Float64, BetheFreeEnergy(), model), (f) -> push!(fe, f))
    setmarginal!(γ, GammaShapeRate(1.0, 1.0))
    setmarginal!(θ, MvNormalMeanPrecision(zeros(order_1), Matrix{Float64}(I, order_1, order_1)))
    
    setmarginal!(τ, GammaShapeRate(prior_τ[1], prior_τ[2]))
    setmarginal!(η, MvNormalMeanPrecision(prior_η[1], prior_η[2])) # better

    marginal_ar_1 = MvNormalMeanPrecision(zeros(2*order_1), Matrix{Float64}(I, 2*order_1, 2*order_1))
    marginal_ar_2 = MvNormalMeanPrecision(zeros(2*order_2), Matrix{Float64}(I, 2*order_2, 2*order_2))
    
    for i in 1:n
        setmarginal!(ar_1_nodes[i], :y_x, marginal_ar_1)
        setmarginal!(ar_2_nodes[i], :y_x, marginal_ar_2)
    end
    for i in 1:niter
        update!(o, data)
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

# TVAR + AR model
@model function coupled_model_tvar(n, prior_η, prior_τ, order_1, order_2, artype, c1, c2, ω)

    θ = randomvar(n)
    z  = randomvar(n)
    z1 = randomvar(n)

    x  = randomvar(n)
    x1 = randomvar(n)
    o = datavar(Float64, n)
    
    ct1  = constvar(c1)
    ct2  = constvar(c2)
    
    # AR_1
    γ ~ GammaShapeRate(1.0, 1e-4) where {q=MeanField()}
    θ0 ~ MvNormalMeanPrecision(zeros(order_1), diageye(order_1)) where {q=MeanField()}
    z0 ~ MvNormalMeanPrecision(zeros(order_1), diageye(order_1)) where {q=MeanField()}
    
    # AR_2
    τ ~ GammaShapeRate(prior_τ[1], prior_τ[2]) where {q=MeanField()}
    η ~ MvNormalMeanPrecision(prior_η[1], prior_η[2]) where {q=MeanField()}
    x0 ~ MvNormalMeanPrecision(zeros(order_2), diageye(order_2)) where {q=MeanField()} # synthetic

    z_prev = z0
    x_prev = x0
    θ_prev = θ0

    ar_1_nodes = Vector{FactorNode}(undef, n)
    ar_2_nodes = Vector{FactorNode}(undef, n)
    
    pipeline_1 = InitARMessages(MvNormalMeanPrecision(zeros(order_1), diageye(order_1)))
    pipeline_2 = InitARMessages(MvNormalMeanPrecision(zeros(order_2), diageye(order_2)))
    
    for i in 1:n
        θ[i] ~ MvNormalMeanPrecision(θ_prev, inv(ω)*diageye(order_1))
        ar_1_nodes[i], z[i] ~ AR(z_prev, θ[i], γ) where { 
            q = q(y, x)q(γ)q(θ), meta = ARMeta(artype, order_1, ARsafe()),
            pipeline=pipeline_1
        }
        
        z1[i] ~ dot(ct1, z[i])
        
        ar_2_nodes[i], x[i] ~ AR(x_prev, η, τ) where { 
            q = q(y, x)q(γ)q(θ), meta = ARMeta(artype, order_2, ARsafe()) ,
            pipeline=pipeline_2
        }
        x1[i] ~ dot(ct2, x[i])
        
        o[i] ~ z1[i] + x1[i]
        x_prev = x[i]
        z_prev = z[i]
        θ_prev = θ[i]
    end
    
    scheduler = nothing

    return o, z, z1, θ, γ, x, x1, η, τ, ar_1_nodes, ar_2_nodes, scheduler
end

# TVAR + AR inference
function coupled_inference_tvar(data, prior_η, prior_τ, order_1, order_2, niter; ω=1e-4)
    n = length(data)
    artype = Multivariate
    c1 = zeros(order_1); c1[1] = 1.0
    c2 = zeros(order_2); c2[1] = 1.0
    model, (o, z, z1, θ, γ, x, x1, η, τ, ar_1_nodes, ar_2_nodes, scheduler) = coupled_model_tvar(model_options(limit_stack_depth=100, ), n, prior_η, prior_τ, order_1, order_2, artype, c1, c2, ω)

    γ_buffer = nothing
    θ_buffer = Vector{Marginal}(undef, n)
    x_buffer = Vector{Marginal}(undef, n)
    
    τ_buffer = nothing
    η_buffer = nothing
    z_buffer = Vector{Marginal}(undef, n)
    fe = Vector{Float64}()

    γsub = subscribe!(getmarginal(γ), (mγ) -> γ_buffer = mγ)
    θsub = subscribe!(getmarginals(θ), (mθ) -> copyto!(θ_buffer, mθ))
    xsub = subscribe!(getmarginals(x), (mx) -> copyto!(x_buffer, mx))
    
    τsub = subscribe!(getmarginal(τ), (mτ) -> τ_buffer = mτ)
    ηsub = subscribe!(getmarginal(η), (mη) -> η_buffer = mη)
    zsub = subscribe!(getmarginals(z), (mz) -> copyto!(z_buffer, mz))
    
    fesub = subscribe!(score(Float64, BetheFreeEnergy(), model), (f) -> push!(fe, f))
    
    setmarginal!(γ, GammaShapeRate(1.0, 1.0))
    setmarginal!(τ, GammaShapeRate(prior_τ[1], prior_τ[2]))
    setmarginal!(η, MvNormalMeanPrecision(prior_η[1], prior_η[2])) # better

    marginal_ar_1 = MvNormalMeanPrecision(zeros(2*order_1), diageye(2*order_1))
    marginal_ar_2 = MvNormalMeanPrecision(zeros(2*order_2), diageye(2*order_2))
    marginal_θ = MvNormalMeanPrecision(zeros(order_1), diageye(order_1))
    for i in 1:n
        setmarginal!(θ[i], marginal_θ)
        setmarginal!(ar_1_nodes[i], :y_x, marginal_ar_1)
        setmarginal!(ar_2_nodes[i], :y_x, marginal_ar_2)
    end
    
    for i in 1:niter
        update!(o, data)
    end
    return γ_buffer, θ_buffer, z_buffer, τ_buffer, η_buffer, x_buffer, fe
end