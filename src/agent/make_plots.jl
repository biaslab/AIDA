using Pkg;Pkg.activate("../..");Pkg.instantiate()
using JLD,Plots
using OhMyREPL
using Statistics: mean
include("plot_utils.jl")

# These should match the values used for the experiment
n_steps = 50; T = 80

# Recreate the grid for plotting
gridman =LinRange(0,1,n_steps)
grid = Iterators.product(gridman,gridman)

# Load saved data
efe_grids = load_grid("efe_grids",n_steps,T);
epi_grids = load_grid("epi_grids",n_steps,T);
inst_grids = load_grid("inst_grids",n_steps,T);

# Reconstruct the choices made by the agent
idxs = [argmin(efe_grids[:,:,i]) for i in 1:T]

# Get the values of queried points for all time steps
efe_mins = [efe_grids[:,:,i][idxs[i]] for i in 1:T]
epi_mins = [epi_grids[:,:,i][idxs[i]] for i in 1:T]
inst_mins = [inst_grids[:,:,i][idxs[i]] for i in 1:T]

# Get rid of infs for inhibition of return
efe_means = [mean(efe_grids[:,:,i][isfinite.(efe_grids[:,:,i])]) for i in 1:T]
epi_means = [mean(epi_grids[:,:,i][isfinite.(epi_grids[:,:,i])]) for i in 1:T]
inst_means = [mean(inst_grids[:,:,i][isfinite.(inst_grids[:,:,i])]) for i in 1:T]


# Plot trajectories, demeaned
epi_plot = plot(epi_mins - epi_means, label="Epistemic Value");
inst_plot = plot(inst_mins - inst_means,label="Instrumental Value");
efe_plot = plot(efe_mins - efe_means,label="EFE")
plot(epi_plot,inst_plot,efe_plot,layout = (3,1))

