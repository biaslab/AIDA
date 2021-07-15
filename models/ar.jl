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

