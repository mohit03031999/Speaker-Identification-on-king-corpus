import librosa.util
import librosa.display
import numpy as np
import sys
import matplotlib.pyplot as plt

FIG_SIZE = (15, 10)


def get_spectogram(samples, len_N, adv_N, sample_rate, predict_frame):
    num_mels = 15                           # number of mels
    Spect = np.abs(librosa.stft(samples, n_fft=len_N, hop_length=adv_N, window="hamming", center=False, ))  #Computing STFT for the given signal

    # display spectrogram
    # plt.figure(figsize=FIG_SIZE)
    # librosa.display.specshow(Spect, sr=sample_rate, hop_length=adv_N)
    # plt.xlabel("Time")
    # plt.ylabel("Frequency")
    # plt.colorbar()
    # plt.title("Spectrogram")

    # Power Spectrum of the above computed STFT
    Power_avg = librosa.power_to_db(Spect ** 2)  # dB

    # plt.figure(figsize=FIG_SIZE)
    # librosa.display.specshow(Power_avg, sr=sample_rate, hop_length=adv_N)
    # plt.xlabel("Time")
    # plt.ylabel("Frequency")
    # plt.colorbar(format="%+2.0f dB")
    # plt.title("Spectrogram (dB)")

    # Mel-Scale Filter Bank
    mel_filterbank = librosa.filters.mel(sample_rate, len_N, n_mels=num_mels, fmin=0, fmax=sample_rate / 2)
    mel_power = np.dot(mel_filterbank, Spect)     #Applying filter bank on the Spectrogram
    np.seterr(divide='ignore')
    result = np.where(mel_power > 0.0000000001, np.log10(mel_power), -10)
    mel_specrtogram = 10 * result

    # librosa.display.specshow(mel_specrtogram, sr=sample_rate)
    # plt.colorbar(format="%+2.0f dB")

    delta = librosa.feature.delta(mel_specrtogram)           #Computing the delat featire on the mel spectrogram compted above
    combined = np.concatenate((mel_specrtogram, delta))      #Combining the delta feature with mel spectrogram to get the final feature vector for each frame

    # Taking only the feature vector for speech frames and discarding the feature vector for silence
    speech = []
    combined = np.transpose(combined)
    for i in range(0, len(predict_frame)):
        temp = []
        if predict_frame[i]:
            speech.append(combined[i])

    return speech                            # Return the feature vector for speech frames of an audio file
