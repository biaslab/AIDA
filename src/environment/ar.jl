# "pure" AR model
@model function ar_model(n, order, aγ, bγ)

    x = datavar(Vector{Float64}, n)
    y = datavar(Float64, n)

    γ ~ GammaShapeRate(aγ, bγ) where {q=MeanField()}
    θ ~ MvNormalMeanPrecision(zeros(order), Matrix{Float64}(I, order, order)) where {q=MeanField()}


    for i in 1:n
        y[i] ~ NormalMeanPrecision(dot(x[i], θ), γ) where {q=MeanField()}
    end

    return x, y, θ, γ
end

# AR inference
function ar_inference(inputs, outputs, order, niter, aγ=1.0, bγ=1.0)
    n = length(outputs)
    model, (x, y, θ, γ) = ar_model(n, order, aγ, bγ, options = (limit_stack_depth = 500, ))

    γ_buffer = nothing
    θ_buffer = nothing
    fe = Vector{Float64}()

    subscribe!(getmarginal(γ), (my) -> γ_buffer = my)
    subscribe!(getmarginal(θ), (mθ) -> θ_buffer = mθ)
    subscribe!(score(Float64, BetheFreeEnergy(), model), (f) -> push!(fe, f))

    setmarginal!(γ, GammaShapeRate(aγ, bγ))

    for i in 1:niter
        update!(x, inputs)
        update!(y, outputs)
    end


    return γ_buffer, θ_buffer, fe
end

