function instrumental(x1, xc, y1, σ, l)
    x2 = collect(xc)
    μ_pred, Σ_pred = predict(x1, x2, y1, σ, l)
    KL(ϕ(μ_pred[1], Σ_pred[1]), 1 - eps()) + h(ϕ(μ_pred[1], Σ_pred[1]))
end

function epistemic(x1, xc, y1, σ, l)
    x2 = collect(xc)
    μ_pred, Σ_pred = predict(x1, x2, y1, σ, l)
    -BALD(μ_pred[1], Σ_pred[1])
end

function get_new_decomp(grid, x1, y1, current, σ, l)
    # Compute the EFE grid
    #value_grid = choose_point.(Ref(x1),grid,Ref(y1),σ,l)
    epi_grid = epistemic.(Ref(x1), grid, Ref(y1), σ, l)
    inst_grid = instrumental.(Ref(x1), grid, Ref(y1), σ, l)

    value_grid = epi_grid + inst_grid
    # Ensure that we propose a new trial and not the same one twice in a row
    value_grid[collect(grid).==[(current[1], current[2])]] .= Inf

    # Find the minimum and try it out
    idx = argmin(value_grid)
    x2 = collect(grid)[idx]
    x2, epi_grid, inst_grid, value_grid, idx
end

function get_new_pointvalues(grid, x1, y1, current, σ, l)
    value_grid = choose_point.(Ref(x1), grid, Ref(y1), σ, l)
    # Ensure that we propose a new trial and not the same one twice in a row
    value_grid[collect(grid).==[(current[1], current[2])]] .= Inf

    # Find the minimum and try it out
    idx = argmin(value_grid)

    # Return epistemic/instrumental values at optimum
    epi = epistemic(x1, collect(grid)[idx], y1, σ, l)
    inst = instrumental(x1, collect(grid)[idx], y1, σ, l)

    x2 = collect(grid)[idx]
    x2, epi, inst
end

function load_grid(var, n, T)
    reshape(load(var * ".jld")[var], n, n, T)
end
