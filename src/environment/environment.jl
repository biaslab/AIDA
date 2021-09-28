# models
export ar_model, coupled_model, gaussian_model, lar_model, lar_model_ex
export ar_inference, coupled_inference, inference_gaussian, lar_inference, lar_inference_ex
export lar_batch_learning, batch_coupled_learning

include("ar.jl")
include("coupled_ar.jl")
include("gaussian.jl")
include("lar.jl")