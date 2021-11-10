# "pure" AR model
@model function ar_model(n, μθ, Λθ, aγ, bγ)

    x = datavar(Vector{Float64}, n)
    y = datavar(Float64, n)

    γ ~ GammaShapeRate(aγ, bγ) where {q=MeanField()}
    θ ~ MvNormalMeanPrecision(μθ, Λθ) where {q=MeanField()}


    for i in 1:n
        y[i] ~ NormalMeanPrecision(dot(x[i], θ), γ) where {q=MeanField()}
    end

    return x, y, θ, γ
end

# AR inference
function ar_inference(inputs, outputs, order, niter; priors=Dict(:μθ => zeros(order), :Λθ => diageye(order), :aγ => 1e-4, :bγ => 1.0))
    n = length(outputs)
    @unpack μθ, Λθ, aγ, bγ = priors
    model, (x, y, θ, γ) = ar_model(n, μθ, Λθ, aγ, bγ, options = (limit_stack_depth = 100, ))

    γ_buffer = nothing
    θ_buffer = nothing
    fe = Vector{Float64}()

    subscribe!(getmarginal(γ), (my) -> γ_buffer = my)
    subscribe!(getmarginal(θ), (mθ) -> θ_buffer = mθ)
    subscribe!(score(Float64, BetheFreeEnergy(), model), (f) -> push!(fe, f))

    setmarginal!(γ, GammaShapeRate(aγ, bγ))

    ProgressMeter.@showprogress for i in 1:niter
        update!(x, inputs)
        update!(y, outputs)
    end


    return γ_buffer, θ_buffer, fe
end

