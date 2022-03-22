import librosa.util
import librosa.display
import numpy as np
import os.path
import sys
import matplotlib.pyplot as plt

FIG_SIZE = (15, 10)

def main():
    filename = "speech.wav"
    adv_s = 0.010  # Frame advance (s)
    len_s = 0.020  # Frame advance (s)
    plot_spectrogram(filename,len_s, adv_s)


def plot_spectrogram(filename, len_s, adv_s):
    root = "C:\\Python Projects\Mohit_Project\SpeakerID"
    file_path = os.path.join(root, filename)

    #Loading the audio file using librosa library
    signal, sample_rate = librosa.load(file_path, sr=8000)

    #COmputing frame length and frame step
    frame_length = int(round((len_s) * sample_rate))
    frame_step = int(round((adv_s) * sample_rate))

    #Generating frames for the audio file
    frames = librosa.util.frame(signal, frame_length, frame_step)
    weigh = np.hamming(frame_length)[:, None]

    #Computing Fourier Transform on (frames * window)
    fft = np.fft.rfft(frames * weigh, axis=0)
    fft = np.absolute(fft)
    fft = fft ** 2

    # Prepare fft frequency list
    freqs = float(sample_rate) / frame_length * np.arange(fft.shape[0])

    # Compute spectrogram feature
    ind = np.where(freqs <= (sample_rate/2))[0][-1] + 1
    specgram = 20*np.log(fft[:ind, :])

    plt.figure(figsize=FIG_SIZE)
    librosa.display.specshow(specgram, sr=sample_rate, hop_length=frame_step,cmap = 'magma')
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.colorbar()
    plt.title("Spectrogram")
    plt.show()


if __name__ == "__main__":
    main()