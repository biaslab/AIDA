using Pkg;
using AIDA
using JLD, Plots
using OhMyREPL
using Statistics: mean

# These should match the values used for the experiment
n_steps = 50;
T = 80;

experiment = JLD.load("jlds/experiment.jld")

efe_grids = reshape(experiment["efe_grids"], n_steps, n_steps, T)
epi_grids = reshape(experiment["epi_grids"], n_steps, n_steps, T)
inst_grids = reshape(experiment["inst_grids"], n_steps, n_steps, T)
idxs = experiment["idxs"]

function get_grid_vals(grid, idxs)
    [grid[:, :, t][idxs[t]] for t = 1:size(idxs)[1]]
end

function make_heatmap(grids, gridman, t)
    # Pick out the right grid
    grid = grids[:, :, t]
    # Remove inhibition of return by setting equal to the max value
    grid[grid.==Inf] .= maximum(grid[grid.!=Inf])

    heatmap(gridman, gridman, grid)
end

efe_vals = get_grid_vals(efe_grids, idxs)
epi_vals = get_grid_vals(epi_grids, idxs)
inst_vals = get_grid_vals(inst_grids, idxs)

p1 = plot(epi_vals, label = "Epistemic Value");
p2 = plot(inst_vals, label = "Instrumental Value");
plot(p1, p2, layout = (2, 1))


gridman = LinRange(0, 1, n_steps)

# heatmap
make_heatmap(efe_grids, gridman, 20)

# Get rid of infs for inhibition of return
efe_means = [mean(efe_grids[:, :, i][isfinite.(efe_grids[:, :, i])]) for i = 1:T]
epi_means = [mean(epi_grids[:, :, i][isfinite.(epi_grids[:, :, i])]) for i = 1:T]
inst_means = [mean(inst_grids[:, :, i][isfinite.(inst_grids[:, :, i])]) for i = 1:T]