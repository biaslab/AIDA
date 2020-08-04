import matplotlib.pyplot as plt
from scipy.io import wavfile

fs, y = wavfile.read("speech/clean.wav")
fs, x = wavfile.read("speech/noised.wav")
fs, z = wavfile.read("speech/filtered.wav")


fig, axs = plt.subplots(3)
axs[0].specgram(y, NFFT=80, Fs=fs, noverlap=10)
axs[0].set_title("Clean")
axs[1].specgram(x, NFFT=80, Fs=fs, noverlap=10)
axs[1].set_title("Noised")
axs[2].specgram(z, NFFT=80, Fs=fs, noverlap=10)
axs[2].set_title("Filtered")


axs[2].set(xlabel='Time s', ylabel='Hz')
plt.savefig("img/result.png")