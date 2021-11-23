using Pkg;Pkg.activate("../..");Pkg.instantiate()
using Plots, JLD
#using OhMyREPL
using SpecialFunctions: erf

include("plot_utils.jl")
include("../environment/user.jl")

target =[0.8, 0.2]

# Save figures?
savefig = false

T = 80
ndims = 2
npoints = 1


# There's probably a better way than gridsearch
n_steps = 50
gridman =LinRange(0,1,n_steps)
σ = .5
l = 0.5

# Keep track of last point visited
current = (0.5,0.5)
grid = Iterators.product(gridman,gridman)
# Initialise at a random point
x1 = rand(ndims,npoints) #.* 5
y1 = generate_user_response.(eachcol(x1))

# bookkeeping
efe_grids = []
epi_grids = []
inst_grids = []
idxs = []

let x1 = x1, y1 = y1,current = current, σ = σ, l = l,idxs=idxs;
    for t ∈ 1:T;
	print(t, "\n")
	x2,epi_grid,inst_grid,value_grid,idx = get_new_decomp(grid,x1,y1,current,σ,l);

	# Update current point after testing
	current = (x2[1],x2[2]);

	# Get some user feedback
	y1 = vcat(y1,generate_user_response(collect(x2)));
	x1 = hcat(x1,collect(x2));

	# Optimize hyperparams every 5th iteration
	if t % 5 == 0
	    σ,l = optimize_hyperparams(x1,y1,[σ,l]);
	end

	append!(efe_grids,value_grid)
	append!(epi_grids,epi_grid)
	append!(inst_grids,inst_grid)
	idxs = vcat(idxs,idx)

	if savefig
	    heatmap(gridman,gridman,value_grid);
	    scatter!([x2[2]],[x2[1]],markersize=10,label="Current"); # This gets flipped for some reason???
	    scatter!([target[2]],[target[1]],markersize=10,label="Target");
	    savefig("timepoint_" * (t<10 ? "0" * repr(t) : repr(t)))
	end
    end
    println("All done, saving results")
    save("experiment.jld","efe_grids",efe_grids, "epi_grids",epi_grids ,"inst_grids",inst_grids, "responses",y1, "points",x1,"idxs",idxs)
end;




