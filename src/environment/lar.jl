# classical latent AR model
@model function lar_model(n, order, artype, c, τ, priors)

    if isempty(priors)
        priors[:mθ] = randn(order)
        priors[:vθ] = Matrix{Float64}(I, order, order)
        priors[:aγ] = 1.0
        priors[:bγ] = 1e-4
    end
    
    x = randomvar(n)
    y = datavar(Float64, n)
    ct  = constvar(c)

    γ ~ GammaShapeRate(priors[:aγ],  priors[:bγ]) where {q=MeanField()}
    θ ~ MvNormalMeanPrecision(priors[:mθ], priors[:vθ]) where {q=MeanField()}
    x0 ~ MvNormalMeanPrecision(zeros(order), diageye(order)) where {q=MeanField()}

    x_prev = x0

    ar_nodes = Vector{FactorNode}(undef, n)

    for i in 1:n
        ar_nodes[i], x[i] ~ AR(x_prev, θ, γ) where { q = q(y, x)q(γ)q(θ), meta = ARMeta(artype, order, ARsafe()) }

        y[i] ~ NormalMeanVariance(dot(ct, x[i]), τ)

        x_prev = x[i]
    end

    return y, x, θ, γ, ar_nodes
end

function lar_inference(data, order, niter, τ; priors=Dict(), marginals=Dict())
    n = length(data)
    artype = Multivariate
    c = zeros(order); c[1] = 1.0
    model, (y, x, θ, γ, ar_nodes) = lar_model(n, order, artype, c, τ, priors)

    
    if isempty(marginals)
        marginals[:mθ] = zeros(order)
        marginals[:vθ] = Matrix{Float64}(I, order, order)
        marginals[:aγ] = 1.0
        marginals[:bγ] = 1e-4
    end
    
    γ_buffer = nothing
    θ_buffer = nothing
    x_buffer = Vector{Marginal}(undef, n)
    fe = Vector{Float64}()

    γsub = subscribe!(getmarginal(γ), (mγ) -> γ_buffer = mγ)
    θsub = subscribe!(getmarginal(θ), (mθ) -> θ_buffer = mθ)
    xsub = subscribe!(getmarginals(x), (mx) -> copyto!(x_buffer, mx))
    fesub = subscribe!(score(Float64, BetheFreeEnergy(), model), (f) -> push!(fe, f))

    setmarginal!(γ, GammaShapeRate(marginals[:aγ], marginals[:bγ]))
    setmarginal!(θ, MvNormalMeanPrecision(marginals[:mθ], diageye(order)))

    for i in 1:n
        setmarginal!(ar_nodes[i], :y_x, MvNormalMeanPrecision(zeros(2*order), diageye(2*order)))
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


# LAR unknown meaasurement noise
@model function lar_model_ex(n, order, artype, c)

    x = randomvar(n)
    y = datavar(Float64, n)

    γ ~ GammaShapeRate(0.00001, 1.0) where {q=MeanField()}
    θ ~ MvNormalMeanPrecision(randn(order), diageye(order)) where {q=MeanField()}
    x0 ~ MvNormalMeanPrecision(100.0 * ones(order), diageye(order)) where {q=MeanField()}
    τ ~ GammaShapeRate(1.0, 1.0) where {q=MeanField()}

    x_prev = x0

    ct  = constvar(c)

    ar_nodes = Vector{FactorNode}(undef, n)

    for i in 1:n
        ar_nodes[i], x[i] ~ AR(x_prev, θ, γ) where { q = q(y, x)q(γ)q(θ), meta = ARMeta(artype, order, ARsafe()) }

        y[i] ~ NormalMeanPrecision(dot(ct, x[i]), τ) where {q=MeanField()}

        x_prev = x[i]
    end

    return x, y, θ, γ, τ, ar_nodes
end

function lar_inference_ex(data, order, niter)
    n = length(data)
    artype = Multivariate
    c = zeros(order); c[1] = 1.0
    model, (x, y, θ, γ, τ, ar_nodes) = lar_model_ex(n, order, artype, c)

    γ_buffer = nothing
    τ_buffer = nothing
    θ_buffer = nothing
    x_buffer = Vector{Marginal}(undef, n)
    fe = Vector{Float64}()

    γsub = subscribe!(getmarginal(γ), (mγ) -> γ_buffer = mγ)
    τsub = subscribe!(getmarginal(τ), (mτ) -> τ_buffer = mτ)
    θsub = subscribe!(getmarginal(θ), (mθ) -> θ_buffer = mθ)
    xsub = subscribe!(getmarginals(x), (mx) -> copyto!(x_buffer, mx))
    fesub = subscribe!(score(Float64, BetheFreeEnergy(), model), (f) -> push!(fe, f))

    setmarginal!(γ, GammaShapeRate(1.0, 1.0))
    setmarginal!(τ, GammaShapeRate(1.0, 1.0))
    setmarginal!(θ, MvNormalMeanPrecision(zeros(order), diageye(order)))

    for i in 1:n
        setmarginal!(ar_nodes[i], :y_x, MvNormalMeanPrecision(100.0 * ones(2*order), diageye(2*order)))
    end

    for i in 1:niter
        update!(y, data)
    end

    unsubscribe!(γsub)
    unsubscribe!(τsub)
    unsubscribe!(θsub)
    unsubscribe!(xsub)
    unsubscribe!(fesub)

    return γ_buffer, τ_buffer, θ_buffer, x_buffer, fe
end