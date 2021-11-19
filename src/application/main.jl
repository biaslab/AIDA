using Stipple, StippleUI, StipplePlotly
using WAV
using JLD
using AIDA
using Colors
using Images
## Initialize agent

include("agent.jl")
ndims = 2
agent = EFEAgent(CONTEXTS, 20, ndims, 1)

## helper functions
function get_ha_files(type)

    if type == "wav"
        PATH = INPUT_PATH
    elseif type == "jld"
        PATH = OUTPUT_PATH
    end

    files = map(x -> readdir(PATH * x * "/" * SNR * "/", join = true), CONTEXTS)
    files = collect(Iterators.flatten(files))
    filter!(x -> contains(x, "."*type), files)
end

function get_ha_files_sin(PATH)

    files = readdir(PATH, join = true)
    type = files[1][end-2:end]
    filter!(x -> contains(x, "."*type), files)
end

function map_input_output(inputs, outputs)
    pairs = []
    for input in inputs
        for output in outputs
            if split(split(input, "/")[end], ".")[1] == split(split(output, "/")[end], ".")[1]
                out_dict = JLD.load(output)
                push!(pairs, Dict("input" => signal_alignment(wavread(input)[1], FS),
                    "speech" => get_signal(out_dict["rmz"], FS),
                    "noise" => get_signal(out_dict["rmx"], FS), 
                    "output" => get_signal(out_dict["rmz"], FS) + get_signal(out_dict["rmx"], FS)))
            end
        end
    end
    return pairs
end

##

#== plot HA ==#
pl_input(index) = PlotData(y = ha_pairs[index]["input"], plot = StipplePlotly.Charts.PLOT_TYPE_SCATTER, name = "input")
pl_speech(index) = PlotData(y = ha_pairs[index]["speech"], plot = StipplePlotly.Charts.PLOT_TYPE_SCATTER, name = "speech")
pl_noise(index) = PlotData(y = ha_pairs[index]["noise"], plot = StipplePlotly.Charts.PLOT_TYPE_SCATTER, name = "noise")
function pl_output(index)
    gains = agent.current_gain
    speech = ha_pairs[index]["speech"]
    noise = ha_pairs[index]["noise"]
    PlotData(y = gains[1]*speech + gains[2]*noise, plot = StipplePlotly.Charts.PLOT_TYPE_SCATTER, name = "output")
end
##

function mod_index(index, pairs)
    mod1(index, length(pairs))
end

function update_plots(i)
    stipple_model.ha_data[] = [pl_input(i), pl_speech(i), pl_noise(i), pl_output(i)]
end

function update_gains(i, context, response)
    update_dataset!(agent, context, response)
    # if response == 1.0
    #     return
    # end
    new_X, _ = get_new_proposal(agent, context)
    @show new_X
    agent.current_gain = reshape(collect(new_X), size(agent.current_gain))
    @show new_X
    update_plots(i)
end

function playsound(type)
    if type == "output"
        gains  = agent.current_gain
        speech = ha_pairs[mod_index(stipple_model.index[], ha_pairs)]["speech"]
        noise  = ha_pairs[mod_index(stipple_model.index[], ha_pairs)]["noise"]
        wavplay(gains[1]*speech + gains[2]*noise, FS)
    else
        wavplay(ha_pairs[mod_index(stipple_model.index[], ha_pairs)][type], FS)
    end
end

##

# Meta for audio processing
INPUT_PATH = "sound/speech/"
OUTPUT_PATH = "sound/separated_jld/speech/"

INPUT_PATH_SIN = "sound/speech/sin"
OUTPUT_PATH_SIN = "sound/separated_jld/speech/sin"

SNR = "5dB"
CONTEXTS = ["train", "babble"]
FS = 8000

inputs, outputs = map(x -> get_ha_files(x), ["wav", "jld"])
ha_pairs = map_input_output(inputs, outputs)

inputs, outputs = map(x -> get_ha_files_sin(x), [INPUT_PATH_SIN, OUTPUT_PATH_SIN])
ha_pairs = map_input_output(inputs, outputs)

# Create layout for plots
HA_layout = PlotLayout(plot_bgcolor = "white", yaxis = [PlotLayoutAxis(xy = "y", index = 1, ticks = "outside", showline = true, zeroline = false, title = "amplitude")])

#== reactive model ==#
Base.@kwdef mutable struct Model <: ReactiveModel

    index::R{Integer} = 1

    context::R{String} = "train"

    play_in::R{Bool} = false
    play_speech::R{Bool} = false
    play_noise::R{Bool} = false
    play_output::R{Bool} = false

    like::R{Bool} = false
    dislike::R{Bool} = false

    optimize::R{Bool} = false

    logourl::R{String}    = "img/logo.png"
    dislikeurl::R{String} = "img/dislike.png"
    likeurl::R{String}    = "img/like.png"

    

    ha_data::R{Vector{PlotData}} = [pl_input(index), pl_speech(index), pl_noise(index), pl_output(index)]
    ha_layout::R{PlotLayout} = HA_layout

    config::R{PlotConfig} = PlotConfig()

end

const stipple_model = Stipple.init(Model())


on(i -> update_plots(mod_index(i, ha_pairs)), stipple_model.index)

on(_ -> optimize_hyperparams!(agent, "train"), stipple_model.optimize)

on(_ -> playsound("input"), stipple_model.play_in)
on(_ -> playsound("speech"), stipple_model.play_speech)
on(_ -> playsound("noise"), stipple_model.play_noise)
on(_ -> playsound("output"), stipple_model.play_output)

on(_ -> update_gains(mod_index(stipple_model.index[], ha_pairs), "train", 1.0), stipple_model.like)
on(_ -> update_gains(mod_index(stipple_model.index[], ha_pairs), "train", 0.0), stipple_model.dislike)

#== ui ==#

function ui(stipple_model)
    dashboard(
        vm(stipple_model), class = "container", [
            heading("Active Inference Design Agent")
            Stipple.center([img(src = stipple_model.logourl[], style = "height: 500px; max-width: 700px")
            ]) 
            toggle("Hello")
            Stipple.center([btn("Optim", @click("optimize = !optimize"), color = "pink", type = "submit", wrap = StippleUI.NO_WRAPPER)
                            btn("Next", @click("index += 1"), color = "pink", type = "submit", wrap = StippleUI.NO_WRAPPER)
            ]) 
            row([cell(class = "st-module", [
                h5("Listen to HA")
                cell(class = "st-module", [
                    btn("input ", @click("play_in = !play_in"), color = "blue", type = "submit", wrap = StippleUI.NO_WRAPPER)
                    btn("speech ", @click("play_speech = !play_speech"), color = "orange", type = "submit", wrap = StippleUI.NO_WRAPPER)
                    btn("noise ", @click("play_noise = !play_noise"), color = "green", type = "submit", wrap = StippleUI.NO_WRAPPER)
                    btn("output ", @click("play_output = !play_output"), color = "red", type = "submit", wrap = StippleUI.NO_WRAPPER)
                    ])
                ])
            ]) 
            row([cell(class = "st-module", [
                h5("Evaluate") 
                cell(class = "st-module", [
                    btn("", @click("like = !like"), content = img(src = stipple_model.likeurl[], style = "height: 30; max-width: 30"), type = "submit", wrap = StippleUI.NO_WRAPPER)
                    btn("", @click("dislike = !dislike"), content = img(src = stipple_model.dislikeurl[], style = "height: 30; max-width: 30"), type = "submit", wrap = StippleUI.NO_WRAPPER)])
                ])
            ])
            row([cell(class = "st-module",[ 
                h5("Hearing Aid") 
                StipplePlotly.plot(:ha_data, layout = :ha_layout, config = :config)
                ])
            ])
        ])
end

#== server ==#
route("/") do
    ui(stipple_model) |> html
end

up()
