using Distributions: Bernoulli

function generate_user_response(w::AbstractVector; μ=[0.8, 0.2], a=2, b=1, c=25, d=-0.4)

    @assert length(w) == 2 "The user preferences are currently only defined for 2-dimensional gains."

    f(x) = -sqrt((x[1]-μ[1])^(2)/a + (x[2]-μ[2])^(2)/b)

    z = zeros(10,10)
    for kx = 1:10
        for ky = 1:10
            z[kx, ky] = f([kx/10, ky/10])
        end
    end

    maxz, minz = maximum(z), minimum(z)

    p = 1 ./ (1 + exp.(-c*((f(w)-minz)/(maxz-minz)-0.5+d)))

    return rand(Bernoulli(p))

end
