using Pkg;Pkg.activate("../..");Pkg.instantiate()
using JLD,Plots
using OhMyREPL
include("plot_utils.jl")

n_steps = 50; T = 80
gridman =LinRange(0,1,n_steps)
grid = Iterators.product(gridman,gridman)

# Load saved data
efe_grids = load_grid("efe_grids",50,80);
epi_grids = load_grid("epi_grids",50,80);
inst_grids = load_grid("inst_grids",50,80);


idxs = [argmin(efe_grids[:,:,i]) for i in 1:T]

efe_mins = [efe_grids[:,:,i][idxs[i]] for i in 1:T]
epi_mins = [epi_grids[:,:,i][idxs[i]] for i in 1:T]
inst_mins = [inst_grids[:,:,i][idxs[i]] for i in 1:T]

plot(epi_mins)
plot(inst_mins)
plot(efe_mins)

