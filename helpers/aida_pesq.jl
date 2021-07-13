using PyCall
using Conda
using WAV

# Conda.rm("mkl")
# Conda.add("nomkl")

# install conda packages
# Conda.pip_interop(true)
# Conda.pip("install", "pesq")
# Conda.pip("install", "pystoi")

# https://pypi.org/project/pesq/
const PESQ = PyNULL()

# https://github.com/mpariente/pystoi
# const STOI = PyNULL()

copy!(PESQ, pyimport("pesq").pesq)
# copy!(STOI, pyimport("pystoi").stoi)

# real, fs = wavread("sound/NOIZEUS/clean/sp01.wav")
# noised, fs = wavread("sound/NOIZEUS/train_noise/0dB/sp01_train_sn0.wav")
# processed, fs = wavread("sound/processed/reconstructed_speech_train.wav")

# real = reshape(real, (size(real, 1), ))
# noised = reshape(noised, (size(noised, 1), ))
# processed = reshape(processed, (size(processed, 1), ))

# pesq_1 = PESQ(fs, real, noised, "nb")
# pesq_2 = PESQ(fs, real, processed, "nb")

function calc_PESQ(ref, noise, processed, fs)
    pesq_1 = PESQ(fs, ref, noise, "nb")
    pesq_2 = PESQ(fs, ref, processed, "nb")
    return pesq_1, pesq_2
end