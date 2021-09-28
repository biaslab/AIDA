export SNR, SNR_2

SNR(x, y) = 10*log10(sum(x.^2) / sum((x .- y).^2))

SNR_2(y_true, y_reconstructed) = 10*log10(mean(abs2.(y_true)) ./ mean(abs2.(y_true - y_reconstructed)))