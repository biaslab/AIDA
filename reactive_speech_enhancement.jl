using Revise
using Rocket
using ReactiveMP
using GraphPPL
using Distributions
using LinearAlgebra
import ProgressMeter
using WAV
using Plots

@model function lar_model(n, order, artype, c, a_τ, b_τ)

    x = randomvar(n)
    y = datavar(Float64, n)
    ct  = constvar(c)
    aτ  = constvar(a_τ)
    bτ  = constvar(b_τ)

    γ ~ GammaShapeRate(1.0, 1e-5) where {q=MeanField()}
    θ ~ MvNormalMeanPrecision(randn(order), Matrix{Float64}(I, order, order)) where {q=MeanField()}
    x0 ~ MvNormalMeanPrecision(zeros(order), Matrix{Float64}(I, order, order)) where {q=MeanField()}

    x_prev = x0
    
    τ ~ GammaShapeRate(aτ, bτ) where {q=MeanField()}

    ar_nodes = Vector{FactorNode}(undef, n)

    for i in 1:n
        ar_nodes[i], x[i] ~ AR(x_prev, θ, γ) where { q = q(y, x)q(γ)q(θ), meta = ARMeta(artype, order, ARsafe()) }

        y[i] ~ NormalMeanPrecision(dot(ct, x[i]), τ) where {q=MeanField()}

        x_prev = x[i]
    end

    return x, y, θ, γ, τ, ar_nodes
end


function inference(data, order, niter, a_τ, b_τ)
    n = length(data)
    artype = Multivariate
    c = zeros(order); c[1] = 1.0
    model, (x, y, θ, γ, τ, ar_nodes) = lar_model(n, order, artype, c, a_τ, b_τ)

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

    setmarginal!(γ, GammaShapeRate(0.0001, 1.0))
    setmarginal!(τ, GammaShapeRate(1.0, 1.0))
    setmarginal!(θ, MvNormalMeanPrecision(zeros(order), Matrix{Float64}(I, order, order)))

    for i in 1:n
        setmarginal!(ar_nodes[i], :y_x, MvNormalMeanPrecision(zeros(2*order), Matrix{Float64}(I, 2*order, 2*order)))
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



# clean speech
cl, fs = wavread("audio/1_ds.wav")
cl, fs = wavread("NOIZEUS/sp10.wav")
# white noise
σ² = 0.0001
wn = sqrt(σ²)*randn(length(cl))
# noised speech
ns = cl .+ wn
wavwrite(ns, fs, "audio/1_ds_noised.wav")
ns, fs = wavread("NOIZEUS/sp10_train_sn0.wav")

# dividing into 10ms frames with 2.5ms overlap
start = 1
l = Int(round(0.01*fs))
overlap = Int(round(0.0025*fs))
totseg = Int(ceil(length(ns)/(l-overlap))) - 1
segment = zeros(totseg, l)
zseg = zeros(totseg, l)
for i in 1:totseg - 1
    global start
    try
        segment[i,1:l]=ns[start:start+l-1]
        zseg[i, 1:l] = cl[start:start+l-1]
        start = (l-overlap)*i+1
    catch e
        # println(e)
    end

end
segment[totseg, 1:length(ns)-start+1] = ns[start:length(ns)]
zseg[totseg, 1:length(cl)-start+1] = cl[start:length(cl)];

ar_order = 1
vmp_its = 20

rmx = zeros(totseg, l)
rvx = zeros(totseg, l)
rmθ = zeros(totseg, ar_order)
rvθ = zeros(totseg, ar_order, ar_order)
rmγ = zeros(totseg)
rmτ = zeros(totseg)

# Storage for fe
fe = zeros(totseg, vmp_its)
a_τ_0, b_τ_0 = 1e-3, 1e-3
ProgressMeter.@showprogress for segnum in 1:totseg
    γ, τ, θ, xs, fe[segnum, :] = inference(segment[segnum, :], ar_order, vmp_its, a_τ_0, b_τ_0)
    mx, vx = mean.(xs), cov.(xs)
    mθ, vθ = mean(θ), cov(θ)
    rmx[segnum, :], rvx[segnum, :] = first.(mx), first.(vx)
    rmθ[segnum, :], rvθ[segnum, :, :] = mθ, vθ
    rmγ[segnum], rmτ[segnum] = mean(γ), mean(τ)
    a_τ_0, b_τ_0 = shape(τ), rate(τ)
end

# Reconstructing the signal
cleanSpeech = zeros(length(ns))
cleanSpeech[1:l] = rmx[1, 1:l]

start = l + 1
for i in 2:totseg - 1
    cleanSpeech[start:start+(l-overlap)] = rmx[i,overlap:end]
    start = start + l - overlap - 1
end
cleanSpeech[start:start+l-1] = rmx[totseg,1:l]
cleanSpeech = cleanSpeech[1:length(ns)];

plot(ns)
plot!(cleanSpeech)
wavwrite(cleanSpeech, fs, "audio/1_filtered.wav")

# Dmitry
# import Base.Iterators: partition, flatten, drop
# function bar(array, window, overlap)
#     w = partition(array, window)
#     return flatten(vcat(first(w), drop.(drop(w, 1), overlap)))
# end

using Plots
fe_est = sum(fe, dims=1)' ./ totseg
plot(fe_est)

ar_meta = ARMeta(2, Multivariate, ARsafe)

ae = score(AverageEnergy(), AR, Val{(:y_x, :θ, :γ)}, (as_marginal(MvNormalMeanCovariance(zeros(2), ones(2))), as_marginal(NormalMeanVariance(0.0, 1.0)), as_marginal(GammaShapeRate(1.0, 1.0))), ar_meta)
