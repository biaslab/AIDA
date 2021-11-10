# models
export ar_model, coupled_model, gaussian_model, lar_model, coupled_model_tvar
export ar_inference, coupled_inference, gaussian_inference, lar_inference, coupled_inference_tvar
export lar_batch_learning, batch_coupled_learning
export model_selection
export generate_user_response

include("ar.jl")
include("coupled_ar.jl")
include("gaussian.jl")
include("lar.jl")
include("classifiers.jl")
include("user.jl")