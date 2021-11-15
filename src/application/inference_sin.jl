using Rocket
using GraphPPL
using ReactiveMP
using AIDA
using Plots
using WAV

s, fs = WAV.wavread("sound/speech/clean/sp01.wav")
n = .01*sin.(collect(1:length(s)))
Plots.plot(x, xlims=(1, 100))
x = s .+ n
WAV.wavwrite(x, fs, "/Users/apodusenko/Desktop/test.wav")


inputs, outputs = ar_ssm(n, 2)
γ, θ, fe = ar_inference(inputs, outputs, 2, 10, priors=Dict(:μθ => zeros(2), :Λθ => diageye(2), :aγ => 1.0, :bγ => 1.0))

speech_seg = get_frames(x, fs)
totseg = size(speech_seg)[1]
priors_mη, priors_vη, priors_τ = prior_to_priors(mean(θ), precision(θ), (11263, 1.0), totseg, 2)
priors_η = (priors_mη, priors_vη)
rmz, rvz, rmθ, rvθ, rγ, rmx, rvx, rmη, rvη, rτ, fe = batch_coupled_learning(speech_seg, priors_η, priors_τ, 10, 2, 10);

s_ = get_signal(rmz, fs)
n_ = get_signal(rmx, fs)
Plots.plot(x)
Plots.plot!(s_)
Plots.plot!(n_)

WAV.wavwrite(s_, fs, "/Users/apodusenko/Desktop/speech.wav")
WAV.wavwrite(n_, fs, "/Users/apodusenko/Desktop/noise.wav")
