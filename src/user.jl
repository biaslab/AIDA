using Distributions: Bernoulli
∑(x) = sum(x)

function generate_user_response(w::Union{Tuple,AbstractVector}; μ = [0.8, 0.2], σ = [0.1, 0.1], β = 25.0, scale = 2.0, binary = true)

    @assert length(w) == 2 "The user preferences are currently only defined for 2-dimensional gains."

    # Negative squared distance
    f(x) = -∑(((x .- μ) .^ 2.0 ./ σ))

    # P is proportional to distance from goal, multiplied by a scaling factor
    # Scaling factor ensures we can get higher than 0.5 when w = μ
    p(y) = scale / (1 + exp(-β * f(y)))

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

# #use this for visualisation
#using Plots
#z = zeros(100,100);
#for i in 1:100
#    for j in 1:100
#	z[i,j] = generate_user_response([i/100,j/100],binary=false)
#    end
#end
#heatmap(z);
#savefig("user_prefs.png")
