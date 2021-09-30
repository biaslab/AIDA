# if frame is voiced then 1 else 0
function VAD(frame, inf_ar, inf_gaus; kwargs...)
    @unpack ar_order, vmp_iter, priors_m1, priors_m2 = kwargs

    res_ar = inf_ar(data, vmp_iter, priors=priors_m1)
    res_gaus = inf_gaus(data, vmp_iter, priors=priors_m2)

    res_ar[end][end] < res_gaus[end][end]
end

# TODO: Extend to multiple models
function context_detection(frame, inf_ar1, inf_ar2; kwargs...)
    @unpack ar_order1, ar_order2, vmp_iter, priors_ar1, priors_ar2 = kwargs

    res_ar1 = inf_ar1(data, vmp_iter, priors=priors_m1)
    res_ar2 = inf_ar2(data, vmp_iter, priors=priors_m1)

    res_ar1[end][end] < res_ar2[end][end]
end