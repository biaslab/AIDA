using Distributions: Bernoulli
using LinearAlgebra: diagm

function generate_user_response(w::Union{Tuple,AbstractVector}; μ = [0.8, 0.2], σ = [0.004, 0.004], scale = 2.0, binary = true)

    @assert length(w) == 2 "The user preferences are currently only defined for 2-dimensional gains."

    # Exponentiated and weighted negative squared distance
    f(x) = exp((x - μ)' * diagm(1 ./ σ ) * (x - μ))

    # P is proportional to distance from goal, multiplied by a scaling factor
    # Scaling factor ensures we can get higher than 0.5 when w = μ
    p(y) = scale / (1 + f(y))

    # Compute probability
    p = p(w)

    # Check that scaling didn't break things
    @assert p ≤ 1 "p is not correctly normalised. Try a different scaling factor"

    if binary
        return 1.0 * rand(Bernoulli(p))
    else
        return p
    end
end
