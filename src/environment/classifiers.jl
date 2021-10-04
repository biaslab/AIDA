# if frame is voiced then 1 else 0
function model_selection(frame, inf_m1, inf_m2; kwargs...)
    @unpack vmp_iter, priors_m1, priors_m2 = kwargs

    res_m1 = inf_m1(frame, vmp_iter, priors=priors_m1)
    res_m2 = inf_m2(frame, vmp_iter, priors=priors_m2)
    res_m1[end][end] < res_m2[end][end]
end