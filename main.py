import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import os
import sys
import pandas as pd
import cv2
import time
import argparse
import matplotlib.font_manager as fm
import random

from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from keras.layers import *
from keras.models import Model
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
from tensorflow import keras
import tensorflow_datasets.public_api as tfds
from keras.datasets import mnist
from keras.utils import np_utils
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from matplotlib.collections import QuadMesh
from pandas import DataFrame
from tensorflow.keras import regularizers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
(X_train_1, Y_train_1), (X_test_1, Y_test_1) = keras.datasets.mnist.load_data()
data = pd.read_csv("A_Z Handwritten Data.csv").astype('float32')


# final try

number_of_classifiers = 36

def load_az_dataset():
    # initialize the list of data and labels
    dataaz = []
    labelsaz = []
    data_az = pd.read_csv("A_Z Handwritten Data.csv").astype('float32')
    data_az.rename(columns={'0': 'label'}, inplace=True)

    # Splite data the X - Our data , and y - the predict label
    x = data_az.drop('label', axis=1)
    y = data_az['label']
    data_az_tr, data_az_ts, labels_az_tr, labels_az_ts = train_test_split(x, y)

    standard_scaler = MinMaxScaler()
    standard_scaler.fit(data_az_tr)

    data_az_tr = standard_scaler.transform(data_az_tr)
    data_az_ts = standard_scaler.transform(data_az_ts)

    data_az_tr = data_az_tr.reshape(data_az_tr.shape[0], 28, 28, 1).astype('float32')  # zmiana kształtu na 3 wymiary
    data_az_ts = data_az_ts.reshape(data_az_ts.shape[0], 28, 28, 1).astype('float32')  # zmiana kształtu na 3 wymiary

    labels_az_tr = np.array(labels_az_tr).astype(int)
    labels_az_ts = np.array(labels_az_ts).astype(int)

    # return a 2-tuple of the A-Z data and labels
    return data_az_tr, labels_az_tr, data_az_ts, labels_az_ts


def load_zero_nine_dataset():
    # load the MNIST dataset and stack the training data and testing
    # data together (we'll create our own training and testing splits
    # later in the project)
    ((train_data, train_labels), (test_data, test_labels)) = mnist.load_data()
    data = np.vstack([train_data, test_data])
    labels = np.hstack([train_labels, test_labels])
    train_data = train_data.reshape(train_data.shape[0], 28, 28, 1).astype('float32')
    test_data = test_data.reshape(test_data.shape[0], 28, 28, 1).astype('float32')

    return train_data, train_labels, test_data, test_labels


# load all datasets
(az_data_tr, az_labels_tr, az_data_ts, az_labels_ts) = load_az_dataset()
(digits_data_tr, digits_labels_tr, digits_data_ts, digits_labels_ts) = load_zero_nine_dataset()

# the MNIST dataset occupies the labels 0-9,
# so let's add 10 to every A-Z label to ensure the A-Z characters are not incorrectly labeled as digits
az_labels_tr += 10
az_labels_ts += 10

data_tr = np.concatenate((az_data_tr, digits_data_tr), axis=0)
data_ts = np.concatenate((az_data_ts, digits_data_ts), axis=0)
labels_tr = np.hstack([az_labels_tr, digits_labels_tr])
labels_ts = np.hstack([az_labels_ts, digits_labels_ts])
# print(data_tr.shape)
# print(labels_tr.shape)

names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
         'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']


def list_to_np_array_and_reshape(data_tr, labels_tr):
    data_tr_cut = np.asarray(data_tr, dtype=np.float32)
    labels_tr_cut = np.asarray(labels_tr, dtype=np.int)
    data_tr_cut = data_tr_cut.reshape(data_tr_cut.shape[0], 28, 28, 1).astype('float32')
    return data_tr_cut, labels_tr_cut


def cut_contents_of_each_element(labels_tr, data_tr, range_of_contents):
    data_tr_cut = []
    labels_tr_cut = []
    count = np.zeros(number_of_classifiers, dtype='int')
    for i, j in zip(labels_tr, data_tr):
        if count[i] < range_of_contents:
            data_tr_cut.append(j)
            labels_tr_cut.append(i)
        count[i] += 1
    return data_tr_cut, labels_tr_cut


def filling_gapes(labels_tr, data_tr, data_tr_cut, labels_tr_cut, range_of_contents):
    data_tr_cut_new = []
    labels_tr_cut_new = []
    for i, j in zip(labels_tr_cut, data_tr_cut):
        data_tr_cut_new.append(j)
        labels_tr_cut_new.append(i)
    count = np.zeros(number_of_classifiers, dtype='int')
    for i in labels_tr:
        count[i] += 1
    missing = np.zeros(number_of_classifiers, dtype='int')
    for i in range(number_of_classifiers):
        missing[i] = range_of_contents - count[i]
    index = 0
    for i in missing:
        if i > 0:
            for j in range(i):
                c = list(zip(labels_tr, data_tr))
                random.shuffle(c)
                labels_tr, data_tr = zip(*c)
                for m, n in zip(labels_tr, data_tr):
                    if m == index:
                        data_tr_cut_new.append(n)
                        labels_tr_cut_new.append(m)
                        break
        index += 1
    return data_tr_cut_new, labels_tr_cut_new


def show_number_of_each_element(labels):
    count = np.zeros(number_of_classifiers, dtype='int')
    for i in labels:
        count[i] += 1
    alphabets = []
    for i in names:
        alphabets.append(i)
    fig, ax = plt.subplots(1, 1, figsize=(10, 15))
    ax.barh(alphabets, count)
    plt.title("Contents of training input per character: ")
    plt.xlabel("Number of elements ")
    plt.ylabel("Characters")
    plt.grid()
    plt.savefig('Contents of training input per character after cutting and filling - prounning' + str(count[0]) + '.png')


def train_my_model(optm, num, epochs, data_tr, labels_tr , lsr='all'):  # train_my_model(optimizer, number for image saving, number of epochs,
    # learning set range)
    '''
    model_ld = Sequential()
    model_ld.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
    model_ld.add(MaxPool2D(pool_size=(2, 2), strides=2))
    model_ld.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model_ld.add(MaxPool2D(pool_size=(2, 2), strides=2))
    model_ld.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
    model_ld.add(MaxPool2D(pool_size=(2, 2), strides=2))
    model_ld.add(Flatten())
    model_ld.add(Dense(64, activation="relu"))
    model_ld.add(Dense(128, activation="relu"))
    model_ld.add(Dense(number_of_classifiers, activation="softmax"))
    '''

    model_ld = Sequential()  # LeNet-5 upgraded
    # Layer 1
    model_ld.add(Conv2D(filters=32, kernel_size=5, strides=1, activation="relu", input_shape=(28, 28, 1),
                        kernel_regularizer=regularizers.l2(0.0005)))
    # Layer 2
    model_ld.add(Conv2D(filters=32, kernel_size=5, strides=1, use_bias=False, activation="relu"))
    # Layer 3
    model_ld.add(BatchNormalization())
    model_ld.add(MaxPooling2D(pool_size=2, strides=2))
    model_ld.add(Dropout(0.25))
    # Layer 3
    model_ld.add(Conv2D(filters=64, kernel_size=3, strides=1, activation="relu",
                        kernel_regularizer=regularizers.l2(0.0005)))
    # Layer 4
    model_ld.add(Conv2D(filters=64, kernel_size=3, strides=1, use_bias=False, activation="relu"))
    # Layer 5
    model_ld.add(BatchNormalization())
    model_ld.add(MaxPooling2D(pool_size=2, strides=2))
    model_ld.add(Dropout(0.25))
    model_ld.add(Flatten())
    # Layer 6
    model_ld.add(Dense(units=256, use_bias=False, activation="relu"))
    # Layer 7
    model_ld.add(BatchNormalization())
    # Layer 8
    model_ld.add(Dense(units=128, use_bias=False, activation="relu"))
    # Layer 9
    model_ld.add(BatchNormalization())
    # Layer 10
    model_ld.add(Dense(units=84, use_bias=False, activation="relu"))
    # Layer 11
    model_ld.add(BatchNormalization())
    model_ld.add(Dropout(0.25))
    # Output
    model_ld.add(Dense(units=number_of_classifiers, activation="softmax"))

    # optimizer: adam , sgd , rmsprop, SGD(lr=1e-4, momentum=0.9)
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    model_ld.compile(optimizer=optm, loss='sparse_categorical_crossentropy', metrics=['accuracy'])  # run_eagerly=True)

    history = model_ld.fit(data_tr, labels_tr, epochs=epochs, validation_data=(data_ts, labels_ts), callbacks=[callback]
                           , use_multiprocessing=True)

    model_ld.summary()

    model_ld.evaluate(data_ts, labels_ts, use_multiprocessing=True)

    # model_ld.save('CNN_ld_2.model')

    print("The validation accuracy is :", history.history['val_accuracy'])
    print("The training accuracy is :", history.history['accuracy'])
    print("The validation loss is :", history.history['val_loss'])
    print("The training loss is :", history.history['loss'])

    # model_lc = tf.keras.models.load_model('CNN_lc_1.model')

    y_predicted = model_ld.predict(data_ts, use_multiprocessing=True)
    y_predicted_labels = [np.argmax(i) for i in y_predicted]
    cm = tf.math.confusion_matrix(labels=labels_ts, predictions=y_predicted_labels)
    c_m = confusion_matrix(labels_ts, y_predicted_labels, labels=None, sample_weight=None, normalize='true')

    # heatmap with values
    plt.figure(figsize=(24, 16))
    heat_map = sn.heatmap(c_m, annot=True, fmt='.3f', xticklabels=names, yticklabels=names, cbar=False)
    heat_map.set_yticklabels(heat_map.get_yticklabels(), rotation=35)
    sn.set(font_scale=0.8)
    plt.title('CNN_ld_' + str(num) + '.model training ' + optm + ' epochs ' + str(epochs) +
              ' test of LeNet-5 based neural network v2 and set range ' + str(lsr) + ' without filling', fontsize=18)
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.savefig('pres - CNN_ld_' + str(num) + '.model training heatmap ' + optm + ' epochs ' + str(epochs) +
                ' test of LeNet-5 based neural network v2 and set range ' + str(lsr) + ' without filling.png')

    # heatmap with values and lines
    plt.figure(figsize=(24, 16))
    heat_map = sn.heatmap(c_m, annot=True, fmt='.3f', xticklabels=names, yticklabels=names, cbar=False, linewidths=0.01,
                          linecolor='white')
    heat_map.set_yticklabels(heat_map.get_yticklabels(), rotation=35)
    sn.set(font_scale=0.8)
    plt.title('CNN_ld_' + str(num) + '.model training ' + optm + ' epochs ' + str(epochs) +
              ' test of LeNet-5 based neural network v2 and set range ' + str(lsr) + ' without filling', fontsize=18)
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.savefig('pres_l - CNN_ld_' + str(num) + '.model training heatmap ' + optm + ' epochs ' + str(epochs) +
                ' test of LeNet-5 based neural network v2 and set range ' + str(lsr) + ' without filling.png')

    # heatmap without values
    plt.figure(figsize=(24, 16))
    heat_map = sn.heatmap(c_m, fmt='', xticklabels=names, yticklabels=names, cbar=False, linewidths=0.01,
                          linecolor='white')
    heat_map.set_yticklabels(heat_map.get_yticklabels(), rotation=35)
    sn.set(font_scale=0.8)
    plt.title('CNN_ld_' + str(num) + '.model training ' + optm + ' epochs ' + str(epochs) +
              ' test of LeNet-5 based neural network v2 and set range ' + str(lsr) + ' without filling', fontsize=18)
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.savefig('pres - CNN_ld_' + str(num) + '.model training heatmap without values ' + optm + ' epochs ' + str(epochs) +
                ' test of LeNet-5 based neural network v2 and set range ' + str(lsr) + ' without filling.png')

    # double plot showing learning process
    plt.figure(figsize=(20, 10))
    plt.suptitle('CNN_ld_' + str(num) + '.model training ' + optm + ' epochs ' + str(epochs) +
                 'test of LeNet-5 based neural network v2 and set range ' + str(lsr) + ' without filling', fontsize=18)

    plt.subplot(1, 2, 1)
    plt.xlabel('Epochs', fontsize=8)
    plt.ylabel('Training Loss', fontsize=8)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend(loc='upper right')

    plt.subplot(1, 2, 2)
    plt.ylabel('Training Accuracy', fontsize=8)
    plt.xlabel('Epochs', fontsize=8)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend(loc='lower right')
    # plt.show()
    plt.savefig('pres - CNN_ld_' + str(num) + '.model training evaluation ' + optm + ' epochs ' + str(epochs) +
                'test of LeNet-5 based neural network v2 and set range ' + str(lsr) + 'without filling.png')


'''
train_my_model('adam', 1, 20)
train_my_model('adam', 1, 25)
train_my_model('adam', 1, 30)
train_my_model('sgd', 2, 20)
train_my_model('sgd', 2, 25)
train_my_model('sgd', 2, 30)
train_my_model('adadelta', 3, 20)
train_my_model('adadelta', 3, 30)
train_my_model('adadelta', 3, 40)
train_my_model('adadelta', 3, 50)
train_my_model('adadelta', 3, 60)
train_my_model('adadelta', 3, 100)
train_my_model('adagrad', 4, 20)
train_my_model('adagrad', 4, 25)
train_my_model('adagrad', 4, 30)
train_my_model('adamax', 5, 10)
train_my_model('adamax', 5, 20)
train_my_model('adamax', 5, 25)
train_my_model('adamax', 5, 30)
train_my_model('nadam', 6, 10)
train_my_model('nadam', 6, 20)
train_my_model('nadam', 6, 25)
train_my_model('nadam', 6, 30)
train_my_model('ftrl', 7, 10)
train_my_model('ftrl', 7, 20)
train_my_model('ftrl', 7, 25)
train_my_model('ftrl', 7, 30)
'''
# train_my_model('adamax', 5, 35)
# train_my_model('adamax', 5, 40)
# train_my_model('adamax', 5, 45)
# train_my_model('adamax', 5, 50)
# train_my_model('adamax', 5, 55)
# train_my_model('adamax', 5, 60)

'''
for i in range (50):
    train_my_model('adamax',5,i+1)
'''
# sets = [900, 1000, 2000, 5000, 10000]
# for i in sets:
    #(data_tr_cut, labels_tr_cut) = cut_contents_of_each_element(labels_tr, data_tr, i)
    #(data_tr_cut, labels_tr_cut) = list_to_np_array_and_reshape(data_tr_cut, labels_tr_cut)
    #(data_tr_cut, labels_tr_cut) = filling_gapes(labels_tr, data_tr, data_tr_cut, labels_tr_cut, i)
    #(data_tr_cut, labels_tr_cut) = list_to_np_array_and_reshape(data_tr_cut, labels_tr_cut)
    #show_number_of_each_element(labels_tr_cut)
start = time.time()
train_my_model('adamax', 5, 50, data_tr, labels_tr)  # i)
end = time.time()
print('Time of model learning: ' + str(end - start))
