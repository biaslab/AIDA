# Point this to the right dir
using Pkg;Pkg.activate("../..");Pkg.instantiate()
using Plots,OhMyREPL, JLD
using Statistics:mean


T = 80
n_steps = 50

# Point to the file and variable
file = "archive/efe_grids.jld"
var = "efe_grids"

value_grids = reshape(JLD.load(file)[var],n_steps,n_steps,T)

# Make grid for plotting
gridman =LinRange(0,1,n_steps)

function make_heatmap(grids,t)
    # Pick out the right grid
    grid = value_grids[:,:,t]
    # Remove inhibition of return by setting equal to the max value
    grid[grid .== Inf] .= maximum(grid[grid .!= Inf])

    heatmap(gridman,gridman,grid)
end

# Sample
make_heatmap(value_grids,6)
