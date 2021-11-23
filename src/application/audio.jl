# Meta for audio processing
INPUT_PATH = "sound/speech/"
OUTPUT_PATH = "sound/separated_jld/speech/"

INPUT_PATH_SIN = "sound/speech/sin"
OUTPUT_PATH_SIN = "sound/separated_jld/speech/sin"

SEGLEN = 80

SNR = "5dB"
CONTEXTS = ["babble", "train", "synthetic"]
FS = 8000


## helper functions
function get_ha_files(PATH)
    files = map(x -> readdir(PATH * x * "/" * SNR * "/", join = true), filter(x -> x != "synthetic", CONTEXTS))
    files = collect(Iterators.flatten(files))
    type = files[1][end-2:end]
    filter!(x -> contains(x, "."*type), files)
end

function get_ha_files_sin(PATH)
    files = readdir(PATH, join = true)
    type = files[1][end-2:end]
    filter!(x -> contains(x, "."*type), files)
end

function map_input_output(inputs, outputs, contexts=CONTEXTS)
    pairs = []
    for input in inputs
        for output in outputs
            if split(split(input, "/")[end], ".")[1] == split(split(output, "/")[end], ".")[1]
                out_dict = JLD.load(output)
                push!(pairs, Dict("input" => signal_alignment(wavread(input)[1], FS),
                                  "speech" => get_signal(out_dict["rmz"], FS),
                                  "noise" => get_signal(out_dict["rmx"], FS), 
                                  "output" => get_signal(out_dict["rmz"], FS) + get_signal(out_dict["rmx"], FS),
                                  "context" => occursin(CONTEXTS[1], input) ? CONTEXTS[1] : (occursin(CONTEXTS[2], input) ? CONTEXTS[2] : CONTEXTS[3])))
                                  
            end
        end
    end
    return pairs
end