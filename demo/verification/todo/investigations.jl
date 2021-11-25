using AIDA
using Plots, OhMyREPL, JLD
using Statistics: mean, cov

experiment_1 = JLD.load("jlds/batch_experiment.jld")
experiment_2 = JLD.load("jlds/batch_experiment_2.jld")
# experiment_3 = JLD.load("batch_experiment_3.jld") This round got unlucky....

# Heatmap of positive user appraisals
responses_1 = experiment_1["responses"]
responses_2 = experiment_2["responses"]
#responses_3 = experiment_3["responses"]
resp = vcat(responses_1, responses_2)#,responses_3)
heatmap(resp, legend = false, xlabel = "Time step", ylabel = "Agent number", title = "Simulated user responses")

savefig("tmp/response_heatmap.png")
# Histogram of first index of positive response
idxs = [findfirst(isequal(1), resp[i, :]) for i = 1:size(resp)[1]]
idxs[isnothing.(idxs)] .= 82 # If the agent didn't get a thumbsup, set it to after the last trial
mean(idxs[isnothing.(idxs).==0])
histogram(idxs, bins = 1:maximum(idxs)+1)
savefig("tmp/response_hishist.png")
;


efe_vals_1 = experiment_1["efe_vals"]
efe_vals_2 = experiment_2["efe_vals"]
efe_vals_3 = experiment_3["efe_vals"]
efes = vcat(efe_vals_1, efe_vals_2)#,efe_vals_3)
plot(mean(efes, dims = 1)[:], legend = false);
pyplot()
savefig("tmp/avg_efe.png")

mean(efes, dims = 1);


#epi_vals = JLD.load("epi_vals_bak.jld")["epi_vals"];
#plot(epi_vals[1,:]);
#for t in 2:99
#    plot!(epi_vals[t,:]);
#end
#p1 = plot!(epi_vals[100,:],title="Epistemic Value",legend=false);
#
#efe_vals = JLD.load("efe_vals_bak.jld")["efe_vals"];
#plot(efe_vals[1,:]);
#for t in 2:99
#    plot!(efe_vals[t,:]);
#end
#p2 = plot!(efe_vals[100,:],title="EFE",legend=false);
#
#inst_vals = JLD.load("inst_vals_bak.jld")["inst_vals"];
#plot(inst_vals[1,:]);
#for t in 2:99
#    plot!(inst_vals[t,:]);
#end
#p3 = plot!(inst_vals[100,:],title="Instrumental Value",legend=false);
#
#plot(p1,p2,p3,layout=(3,1))
#
##heatmat
##plot(efe_vals[100,:],title="EFE",legend=false)
