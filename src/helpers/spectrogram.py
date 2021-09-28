import matplotlib.pyplot as plt
from scipy.io import wavfile
plt.style.use("ggplot")


fs, y = wavfile.read("sound/speech/sp02.wav")
fs, x = wavfile.read("sound/mixed/bar_speech.wav")
fs, z = wavfile.read("sound/processed/ha_output.wav")

y = y[0:len(y) - 100]
x = x[0:len(x) - 100]
z = z[0:len(z) - 100]

fig, axs = plt.subplots(3)
axs[0].specgram(y, Fs=fs)
axs[0].set_title("Clean")
axs[1].specgram(x, Fs=fs)
axs[1].set_title("Noised")
axs[2].specgram(z, Fs=fs)
axs[2].set_title("Filtered")


axs[2].set(xlabel='Time s', ylabel='Hz')
plt.savefig("img/result.pdf")


import tikzplotlib
tikzplotlib.save("img/spectrogram.tex")