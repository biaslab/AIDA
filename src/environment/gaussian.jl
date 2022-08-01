# Auxilary model to infer stationary noise variance
@model [default_factorisation = MeanField()] function gaussian_model(n, aτ, bτ)

    y = datavar(Float64, n)

    x   ~ NormalMeanPrecision(0.0, 1.0)
    τ   ~ GammaShapeRate(aτ, bτ)

    for i in 1:n
        y[i] ~ NormalMeanPrecision(x, τ)
    end

    return y, x, τ
end

# Gaussian inference
function gaussian_inference(outputs, niter; priors=Dict(:aτ => 1e-4, :bτ => 1.0))
    n = length(outputs)
    @unpack aτ, bτ = priors
    model, (y, x, τ) = gaussian_model(model_options(limit_stack_depth = 500, ), n, aτ, bτ)
    τ_buffer = nothing
    x_buffer = nothing
    fe = Vector{Float64}()

    subscribe!(getmarginal(τ), (mτ) -> τ_buffer = mτ)
    subscribe!(getmarginal(x), (mx) -> x_buffer = mx)
    subscribe!(score(Float64, BetheFreeEnergy(), model), (f) -> push!(fe, f))

    setmarginal!(τ, GammaShapeRate(1.0, 1.0))
    setmarginal!(x, NormalMeanPrecision(0.0, 1.0))

    for _ in 1:niter
        update!(y, outputs)
    end

    return x_buffer, τ_buffer, fe
end