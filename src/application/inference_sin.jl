using Rocket
using GraphPPL
using ReactiveMP
using AIDA
using Plots
using WAV
using JLD

function HA_sin_output(clean_path, sin_mag=0.05)

    s, fs = WAV.wavread(clean_path)
    n = sin_mag*sin.(collect(1:length(s)))
    x = s .+ n
    println("Obtaining prior for noise component")
    inputs, outputs = ar_ssm(n, 2)
    γ, θ, fe = ar_inference(inputs, outputs, 2, 10, priors=Dict(:μθ => zeros(2), :Λθ => diageye(2), :aγ => 1.0, :bγ => 1.0))

    speech_seg = get_frames(x, fs)
    totseg = size(speech_seg)[1]
    priors_mη, priors_vη, priors_τ = prior_to_priors(mean(θ), precision(θ), (11263, 1.0), totseg, 2)
    priors_η = (priors_mη, priors_vη)

    println("Obtaining HA output")
    rmz, rvz, rmθ, rvθ, rγ, rmx, rvx, rmη, rvη, rτ, fe = batch_coupled_learning(speech_seg, priors_η, priors_τ, 10, 2, 10);

    s_ = get_signal(rmz, fs)
    n_ = get_signal(rmx, fs)

    WAV.wavwrite(x, fs, "sound/speech/sin/"*clean_path[findlast("/", clean_path)[1]+1:end-3]*"wav")

    JLD.save("sound/separated_jld/speech/sin/"*clean_path[findlast("/", clean_path)[1]+1:end-3]*"jld",
            "rmz", rmz, "rvz", rvz, "rmθ", rmθ, "rvθ", rvθ, "rγ", rγ, 
            "rmx", rmx, "rvx", rvx, "rmη", rmη, "rvη", rvη, "rτ", rτ,
            "fe", fe, "filename", clean_path)

    return x, s_, n_
end

x, s, n = HA_sin_output("sound/speech/clean/sp01.wav")