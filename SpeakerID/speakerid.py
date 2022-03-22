# Python standard libraries
import os
# Python add-on libraries
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
import pandas as pd
import librosa.util
import numpy as np
from tensorflow.keras.layers import Input, Dense, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.regularizers import l1
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import seaborn as sns
from keras.models import Sequential
from sklearn.metrics import log_loss, accuracy_score

from sklearn.model_selection import KFold
import sys

# project modules
import lib.files
import lib.rms
import lib.Spectogram
import lib.buildmodels
from lib.endpointer import Endpointer


def main():
    plt.ion()

    root = "C:\\Python Projects\Speaker Identification on king corpus\king"

    adv_s = 0.010  # Frame advance (s)
    len_s = 0.020  # Frame advance (s)

    foldsN = 2  # cross validation folds

    # Create a dictionary for the different recording conditions
    # Each entry will have a subdictionary keyed by speaker identity
    corpora = dict()
    for dir in ('wb',):  # for both use:  ('nb', 'wb')
        # Get all files for this channel condition
        file_list = lib.files.get_files(os.path.join(root, dir))
        # Split up by speaker
        corpora[dir] = lib.files.partition_files(file_list)

        df = pd.DataFrame(data=((k, words) for k, v in corpora[dir].items() for words in v))
        train, test = train_test_split(df, test_size=0.3)
        train = train.reset_index(drop=True)
        test = test.reset_index(drop=True)
        i = 0
        signal = [None] * len(df)
        for x in df[1]:
            signal[i], samplerate = librosa.load(x, sr=8000)
            i = i + 1

        frame_length = int(round((len_s) * samplerate))
        frame_step = int(round((adv_s) * samplerate))

        dataset_train = pd.DataFrame(columns=['Features', 'SpeakerID', 'Filename'])
        dataset_test = pd.DataFrame(columns=['Features', 'SpeakerID', 'Filename'])
        for j in range(0, len(df)):
            rms = lib.rms.get_rms(signal[j], frame_length, frame_step)
            endpoint = lib.endpointer.Endpointer(np.array(rms).reshape(-1, 1))
            frame_predict = endpoint.predict(np.array(rms).reshape(-1, 1))
            spectr = lib.Spectogram.get_spectogram(signal[j], frame_length, frame_step, samplerate, frame_predict)

            df1 = pd.DataFrame({'Features': spectr})
            df1 = df1.assign(SpeakerID=df[0][j])
            df1 = df1.assign(Filename=df[1][j])
            filename = str(df[1][j])
            for k in range(0, len(train)):
                if train[1][k] == filename:
                    dataset_train = dataset_train.append(df1, ignore_index=True)
            for l in range(0, len(test)):
                if test[1][l] == filename:
                    dataset_test = dataset_test.append(df1, ignore_index=True)

        y_train = dataset_train['SpeakerID']
        y_test = dataset_test['SpeakerID']
        x_train = dataset_train['Features']
        x_test = dataset_test['Features']

        x_train = tf.stack(x_train)
        x_test = tf.stack(x_test)

        le = LabelEncoder()
        Y_Train = le.fit_transform(y_train)
        Y_Test = le.transform(y_test)

        Y_Train = to_categorical(Y_Train)  # converting the training labels into categories
        Y_Test = to_categorical(Y_Test)  # converting the test labels into categories

        arch_abstract = lambda indim, layer_width, penalty, outdim: [
            (Input, [], {'shape': indim}),
            (BatchNormalization, [], {}),
            (Dense, [layer_width], {'activation': 'relu', 'kernel_regularizer': l2(penalty)}),
            (BatchNormalization, [], {}),
            (Dense, [layer_width], {'activation': 'relu', 'kernel_regularizer': l2(penalty)}),
            (BatchNormalization, [], {}),
            (Dense, [layer_width], {'activation': 'relu', 'kernel_regularizer': l2(penalty)}),
            (BatchNormalization, [], {}),
            (Dense, [layer_width], {'activation': 'relu', 'kernel_regularizer': l2(penalty)}),
            (Dense, [outdim], {'activation': 'softmax'})
        ]

        arch_actual = arch_abstract((30,), 750, 0.01, 51)
        model = lib.buildmodels.build_model(arch_actual)
        model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"])
        model.fit(x_train, Y_Train, batch_size=8000, epochs=20,
                  shuffle=True)  # Training the above created model on train datset

        actual_id = []
        predicted_id = []
        for a in range(0, 75):
            print(a)
            actual_labels = []
            predicted_labels = []
            for i in range(0, len(dataset_test)):
                if dataset_test['Filename'][i] == test[1][a]:
                    pred = model.predict(tf.reshape(x_test[i], [1, -1]))
                    speaker_probs = np.sum(np.log(pred), axis=0)
                    predicted_speaker = np.argmax(np.sum(np.log(pred), axis=0))
                    predicted_labels.append(predicted_speaker)
                    actual_speaker = np.argmax(Y_Test[i], axis=-1)
                    actual_labels.append(actual_speaker)
            most_speaker = max(actual_labels, key=actual_labels.count)
            actual_id.append(most_speaker)
            pred_speak = max(predicted_labels, key=predicted_labels.count)
            predicted_id.append(pred_speak)

        accuracy = accuracy_score(actual_id, predicted_id)
        loss = 1-accuracy
        print("Accuracy on test data: %.2f%%" % (accuracy * 100))
        print("Loss on test data: %.2f%%" % (loss * 100))


        # build confusion matrix and normalized confusion matrix
        conf_matrix = confusion_matrix(actual_labels, predicted_speaker)
        conf_matrix_norm = confusion_matrix(actual_labels, predicted_speaker, normalize='true')


        # plot confusion matrices
        plt.figure(figsize=(16, 6))
        sns.set(font_scale=1.8)  # emotion label and title size
        plt.subplot(1, 2, 1)
        plt.title('Confusion Matrix')
        sns.heatmap(conf_matrix, annot=True, annot_kws={"size": 18})  # annot_kws is value font
        plt.subplot(1, 2, 2)
        plt.title('Normalized Confusion Matrix')
        sns.heatmap(conf_matrix_norm, annot=True, annot_kws={"size": 13})  # annot_kws is value font
        plt.show()

    # print('experiment complete')  # breakpoint line for plots


if __name__ == "__main__":
    main()