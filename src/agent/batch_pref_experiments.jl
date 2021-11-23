using Pkg;Pkg.activate("../..");Pkg.instantiate()
#using Plots, JLD;
using JLD;
#using OhMyREPL
using SpecialFunctions: erf;

include("plot_utils.jl");
include("../environment/user.jl");

target =[0.8, 0.2]

ndims = 2
npoints = 1
n_steps = 50
T = 80
n_runs = 100
gridman =LinRange(0,1,n_steps)

efe_vals = zeros(n_runs,T)
epi_vals = zeros(n_runs,T)
inst_vals = zeros(n_runs,T)
responses = ones(n_runs,T+1)
traj = zeros(n_runs,2,T)

# Runs experiments in a loop
for run ∈ 1:n_runs
    println("starting run: ",run)
    x1 = rand(ndims,npoints) #.* 5
    y1 = generate_user_response.(eachcol(x1))

    σ = .5
    l = 0.5

    # Keep track of last point visited
    current = (0.5,0.5)
    grid = Iterators.product(gridman,gridman)

    let x1 = x1, y1 = y1,current = current, σ = σ, l = l;
	for t ∈ 1:T;
	    x2,epi,inst = get_new_pointvalues(grid,x1,y1,current,σ,l);

#	    println(x2,t)
	    # Update current point after testing
	    current = (x2[1],x2[2]);

	    # Get some user feedback
	    y1 = vcat(y1,generate_user_response(collect(x2)));
	    x1 = hcat(x1,collect(x2));

	    # Optimize hyperparams every 5th iteration
	    if t % 5 == 0
		σ,l = optimize_hyperparams(x1,y1,[σ,l]);
		#println("new σ: ",σ," new l: ",l)
	    end

	    efe_vals[run,t] = epi + inst
	    epi_vals[run,t] = epi
	    inst_vals[run,t] = inst
	    traj[run,:,t] .= x2
	end
	responses[run,:] .= y1
    end
end
#println("All done, saving data")
save("batch_experiment_3.jld", "efe_vals",efe_vals, "epi_vals",epi_vals, "inst_vals",inst_vals, "traj",traj,"responses",responses)




