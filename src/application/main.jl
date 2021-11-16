using Stipple, StippleUI, StipplePlotly
using WAV
using JLD
using AIDA
using Colors
using Images

include("agent.jl")

agent = EFEAgent(CONTEXTS, 20, 2, 1)

INPUT_PATH = "/Users/apodusenko/Documents/Julia/AIDA/sound/speech/"
OUTPUT_PATH = "/Users/apodusenko/Documents/Julia/AIDA/sound/separated_jld/speech/"
# LOGO_PATH = "/Users/apodusenko/Documents/Julia/AIDA/src/application/logo.png"
# OUTPUT_PATH = "/Users/apodusenko/Documents/Julia/AIDA/sound/AIDA/train/"
SNR = "5dB"
CONTEXTS = ["train", "babble"]
FS = 8000

function get_ha_intputs()
    files = map(x -> readdir(INPUT_PATH * x * "/" * SNR * "/", join = true), CONTEXTS)
    files = collect(Iterators.flatten(files))
    filter!(x -> contains(x, ".wav"), files)
end

function get_ha_outputs()
    files = map(x -> readdir(OUTPUT_PATH * x * "/" * SNR * "/", join = true), CONTEXTS)
    files = collect(Iterators.flatten(files))
    filter!(x -> contains(x, ".jld"), files)
end

# helper
function map_input_output(inputs, outputs)
    pairs = []
    for input in inputs
        for output in outputs
            if split(split(input, "/")[end], ".")[1] == split(split(output, "/")[end], ".")[1]
                out_dict = JLD.load(output)
                push!(pairs, Dict("input" => signal_alignment(wavread(input)[1], FS),
                    "speech" => get_signal(out_dict["rmz"], FS),
                    "noise" => get_signal(out_dict["rmx"], FS), "output" => get_signal(out_dict["rmz"], FS) + get_signal(out_dict["rmx"], FS)))
            end
        end
    end
    return pairs
end

function plots_ha_io(pair)
    plt_in = plot(pair["input"], title = "HA input", label = false, size = (400, 300), color = "red")
    plt_sp = plot(pair["speech"], title = "Speech", label = false, size = (200, 150), color = "magenta")
    plt_ns = plot(pair["noise"], title = "Noise", label = false, size = (200, 150), color = "pink")
    plt_out = plot(pair["output"], title = "HA output", label = false, size = (400, 300), color = "green")
    return plt_in, plt_sp, plt_ns, plt_out
end

inputs, outputs = get_ha_intputs(), get_ha_outputs()
ha_pairs = map_input_output(inputs, outputs)


#== plot HA ==#
pl_input(index) = PlotData(y = ha_pairs[index]["input"], plot = StipplePlotly.Charts.PLOT_TYPE_SCATTER, name = "input")
pl_speech(index) = PlotData(y = ha_pairs[index]["speech"], plot = StipplePlotly.Charts.PLOT_TYPE_SCATTER, name = "speech")
pl_noise(index) = PlotData(y = ha_pairs[index]["noise"], plot = StipplePlotly.Charts.PLOT_TYPE_SCATTER, name = "noise")
pl_output(index) = PlotData(y = ha_pairs[index]["output"], plot = StipplePlotly.Charts.PLOT_TYPE_SCATTER, name = "output")

HA_layout = PlotLayout(plot_bgcolor = "white", yaxis = [PlotLayoutAxis(xy = "y", index = 1, ticks = "outside", showline = true, zeroline = false, title = "amplitude")])
# HA_out_layout = PlotLayout(plot_bgcolor = "white", title = PlotLayoutTitle(text="HA output", font=Font(24)), yaxis=[PlotLayoutAxis(xy = "y", index = 1, ticks = "outside", showline = true, zeroline = false, title="response A")])

#== reactive model ==#
Base.@kwdef mutable struct Model <: ReactiveModel

    index::R{Integer} = 1
    play_in::R{Bool} = false
    play_speech::R{Bool} = false
    play_noise::R{Bool} = false
    play_output::R{Bool} = false

    like::R{Bool} = false
    dislike::R{Bool} = false

    logourl::R{String} = "img/logo.png"
    dislikeurl::R{String} = "img/dislike.png"
    likeurl::R{String} = "img/like.png"
    ha_data::R{Vector{PlotData}} = [pl_input(index), pl_speech(index), pl_noise(index), pl_output(index)]
    ha_layout::R{PlotLayout} = HA_layout

    config::R{PlotConfig} = PlotConfig()

end

const stipple_model = Stipple.init(Model())

function mod_index(index, pairs)
    mod1(index, length(pairs))
end

function update_plots(i)
    stipple_model.ha_data[] = [pl_input(i), pl_speech(i), pl_noise(i), pl_output(i)]
end

function playsound(type)
    wavplay(ha_pairs[mod_index(stipple_model.index[], ha_pairs)][type], FS)
end

on(i -> update_plots(mod_index(i, ha_pairs)), stipple_model.index)

on(_ -> playsound("input"), stipple_model.play_in)
on(_ -> playsound("speech"), stipple_model.play_speech)
on(_ -> playsound("noise"), stipple_model.play_noise)
on(_ -> playsound("output"), stipple_model.play_output)

on(_ -> optimize_hyperparams(agent.cmems[1].dataset["X"], agent.cmems[1].dataset["Y"], agent.cmems[1].params), stipple_model.like)
on(_ -> optimize_hyperparams(agent.cmems[1].dataset["X"], agent.cmems[1].dataset["Y"], agent.cmems[1].params), stipple_model.dislike)

#== ui ==#

function ui(stipple_model)
    dashboard(
        vm(stipple_model), class = "container", [
            heading("Active Inference Design Agent")
            Stipple.center([img(src = stipple_model.logourl[], style = "height: 500px; max-width: 700px")]) Stipple.center([col(btn("Next ", @click("index += 1"), color = "pink", type = "submit", wrap = StippleUI.NO_WRAPPER))]) row([
            cell(class = "st-module",
            [
            h5("Listen to HA")
            cell(
            class = "st-module",
            [
            btn("input ", @click("play_in = !play_in"), color = "blue", type = "submit", wrap = StippleUI.NO_WRAPPER)
            btn("speech ", @click("play_speech = !play_speech"), color = "orange", type = "submit", wrap = StippleUI.NO_WRAPPER)
            btn("noise ", @click("play_noise = !play_noise"), color = "green", type = "submit", wrap = StippleUI.NO_WRAPPER)
            btn("output ", @click("play_output = !play_output"), color = "red", type = "submit", wrap = StippleUI.NO_WRAPPER)
        ]
        )
        ]
        )
        ]) 
        ])
end

#== server ==#

route("/") do
    ui(stipple_model) |> html
end

up()
