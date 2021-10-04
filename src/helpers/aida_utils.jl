export prior_to_priors
export get_frames, get_signal, signal_alignment
export ar_ssm
export get_learning_data

# coupled AR model is deisgned to work with time-varying priors for both speech and environmental noise
# prior_to_priors map "static" priors to the corresponding matrices with equal elements
function prior_to_priors(mη, vη, τ, totseg, ar_order)
    # ar_order = size(mη, 2)
    rmη = zeros(totseg, ar_order)
    rvη = zeros(totseg, ar_order, ar_order)
    for segnum in 1:totseg
        rmη[segnum, :], rvη[segnum, :, :] = reshape(mη, (ar_order,)), vη
    end
    priors_eta = rmη, rvη
    priors_tau = [τ for _ in 1:totseg]
    priors_eta[1], priors_eta[2], priors_tau
end

# splitting signal into frames
function get_frames(signal, fs; len_sec=0.01, overlap_sec=0.0025)
    start = 1
    l = Int(round(len_sec*fs))
    overlap = Int(round(overlap_sec*fs))
    totseg = Int(ceil(length(signal)/(l-overlap)))
    segment = zeros(totseg, l)
    for i in 1:totseg - 1
        try
            segment[i,1:l]=signal[start:start+l-1]
            start = (l-overlap)*i+1
        catch
            totseg -= 1
            break
        end
    end
    segment[totseg, 1:length(signal)-start+1] = signal[start:length(signal)]
    return segment
end


# Reconstructing the signal
function get_signal(frames, fs; len_sec=0.01, overlap_sec=0.0025)
    l = Int(len_sec*fs)
    overlap = Int(round(overlap_sec*fs))
    totseg = size(frames, 1)
    signal_len = Int(round(totseg * (l - overlap)))
    signal = zeros(signal_len)
    signal[1:l] = frames[1, 1:l]
    start = l + 1
    for i in 2:totseg
        signal[start:start+l-1-overlap]  = frames[i,overlap+1:end]
        start = start + l - overlap - 1
    end
    signal
end

function signal_alignment(signal, fs; len_sec=0.01, overlap_sec=0.0025)
    frames = get_frames(signal, fs, len_sec=len_sec, overlap_sec=overlap_sec)
    get_signal(frames, fs, len_sec=len_sec, overlap_sec=overlap_sec)
end

function ar_ssm(series, order)
    inputs = [reverse!(series[1:order])]
    outputs = [series[order + 1]]
    for x in series[order+2:end]
        push!(inputs, vcat(outputs[end], inputs[end])[1:end-1])
        push!(outputs, x)
    end
    return inputs, outputs
end

function get_learning_data(preferences::Dict, context, jitter=1e-3)
    idx = findall(isequal(context), preferences["contexts"])
    gains = preferences["gains"]
    appraisals = preferences["appraisals"]
    tgains = vcat(hcat(gains...)', hcat(gains...)', hcat(gains...)')
    tappraisals = vcat(appraisals, appraisals, appraisals)
    
    # augmentation of the dataset with copies
    y = tappraisals
    X = tgains .+ sqrt(jitter)*randn(size(tgains))
    
    return X, y
end