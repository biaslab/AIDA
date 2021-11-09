# classical latent AR model

function fill_dict(dists::Dict, ar_order)
    dists_ = Dict{Symbol, Any}(:mθ => zeros(ar_order), 
                               :vθ => Matrix{Float64}(I, ar_order, ar_order), 
                               :aγ => 1.0, :bγ => 1e-4, 
                               :aτ => 1.0, :bτ => 1.0)
    for key in keys(dists)
        dists_[key] = dists[key]
    end

    return dists_
end

# LAR unknown meaasurement noise
@model function lar_model(n, order, artype, c, priors)
    
    x = randomvar(n)
    y = datavar(Float64, n)
    ct  = constvar(c)

    γ ~ GammaShapeRate(priors[:aγ],  priors[:bγ]) where {q=MeanField()}
    θ ~ MvNormalMeanPrecision(priors[:mθ], priors[:vθ]) where {q=MeanField()}
    
    x0 ~ MvNormalMeanPrecision(zeros(order), diageye(order)) where {q=MeanField()}

    τ = if haskey(priors, :aτ) && haskey(priors, :bτ) && !haskey(priors, :τ)
        τ_tmp = randomvar()
        τ_tmp ~ GammaShapeRate(priors[:aτ], priors[:bτ]) where {q=MeanField()}
        τ_tmp
    else
        priors[:τ]
    end

    x_prev = x0

    ar_nodes = Vector{FactorNode}(undef, n)

    for i in 1:n
        ar_nodes[i], x[i] ~ AR(x_prev, θ, γ) where { q = q(y, x)q(γ)q(θ), meta = ARMeta(artype, order, ARsafe()) }

        y[i] ~ NormalMeanPrecision(dot(ct, x[i]), τ) where {q=MeanField()}

        x_prev = x[i]
    end

    return y, x, θ, γ, τ, ar_nodes
end

function lar_inference(data, niter; priors=Dict(), marginals=Dict())
    n = length(data)
    artype = Multivariate
    order = priors[:order]
    haskey(priors, :order) || error(":order key must be specified in priors dict")
    c = zeros(order); c[1] = 1.0
    priors = fill_dict(priors, order)
    model, (y, x, θ, γ, τ, ar_nodes) = lar_model(n, order, artype, c, priors)
    marginals = fill_dict(marginals, order)

    γ_buffer = nothing
    τ_buffer = nothing
    θ_buffer = nothing
    x_buffer = Vector{Marginal}(undef, n)
    fe = Vector{Float64}()

    if haskey(priors, :aτ) && haskey(priors, :bτ) && !haskey(priors, :τ)
        subscribe!(getmarginal(τ), (mτ) -> τ_buffer = mτ)
        setmarginal!(τ, GammaShapeRate(marginals[:aτ], marginals[:bτ]))
    end
    subscribe!(getmarginal(γ), (mγ) -> γ_buffer = mγ)
    subscribe!(getmarginal(θ), (mθ) -> θ_buffer = mθ)
    subscribe!(getmarginals(x), (mx) -> copyto!(x_buffer, mx))
    subscribe!(score(Float64, BetheFreeEnergy(), model), (f) -> push!(fe, f))

    setmarginal!(γ, GammaShapeRate(marginals[:aγ], marginals[:bγ]))
    setmarginal!(θ, MvNormalMeanPrecision(marginals[:mθ], marginals[:vθ]))

    for i in 1:n
        setmarginal!(ar_nodes[i], :y_x, MvNormalMeanPrecision(ones(2*order), diageye(2*order)))
    end

    for i in 1:niter
        update!(y, data)
    end

    return γ_buffer, τ_buffer, θ_buffer, x_buffer, fe
end

# to optimize
function lar_batch_learning(segments, vmp_its, priors::Dict, marginals=Dict())
    haskey(priors, :order) || error(":order key must be specified in priors dict")
    ar_order = priors[:order]
    totseg = size(segments, 1)
    l      = size(segments, 2)
    rmx = zeros(totseg, l)
    rvx = zeros(totseg, l)
    rmθ = zeros(totseg, ar_order)
    rvθ = zeros(totseg, ar_order, ar_order)
    rγ = fill(tuple(.0, .0), totseg)
    rτ = fill(tuple(.0, .0), totseg)
    fe  = zeros(totseg, vmp_its)
    
    ProgressMeter.@showprogress for segnum in 1:totseg
    
        γ, τ, θ, xs, fe[segnum, :]        = lar_inference(segments[segnum, :], vmp_its, priors=priors, marginals=marginals)
        mx, vx                            = mean.(xs), cov.(xs)
        mθ, vθ                            = mean(θ), cov(θ)
        rmx[segnum, :], rvx[segnum, :]    = first.(mx), first.(vx)
        rmθ[segnum, :], rvθ[segnum, :, :] = mθ, vθ
        rγ[segnum]                        = shape(γ), rate(γ)
        rτ[segnum]                        = haskey(priors, :aτ) ? (shape(τ), rate(τ)) : (.0, .0);
    end
    rmx, rvx, rmθ, rvθ, rγ, !haskey(priors, :τ) ? rτ : nothing
end