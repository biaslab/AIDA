# splitting signal into frames
function get_frames(signal, fs; len_sec=0.01, overlap_sec=0.0025)
    start = 1
    l = Int(round(len_sec*fs))
    overlap = Int(round(overlap_sec*fs))
    totseg = Int(ceil(length(signal)/(l-overlap)))
    segment = zeros(totseg, l)
    for i in 1:totseg - 1
        segment[i,1:l]=signal[start:start+l-1]
        start = (l-overlap)*i+1
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