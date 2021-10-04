# models
export ar_model, coupled_model, gaussian_model, lar_model
export ar_inference, coupled_inference, inference_gaussian, lar_inference
export lar_batch_learning, batch_coupled_learning
export model_selection

include("ar.jl")
include("coupled_ar.jl")
include("gaussian.jl")
include("lar.jl")
include("classifiers.jl")