# if frame is voiced then 1 else 0
function model_selection(frame, inf_ms, priors; vmp_iter=10, verbose=false)
    fes = map(zip(inf_ms, priors)) do (inf_m, prior)
        return inf_m(frame, vmp_iter, priors=prior)[end]
    end
    return verbose ? fes : findmin(last.(fes))[2]
end