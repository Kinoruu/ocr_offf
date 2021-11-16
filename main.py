import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import os
import sys
import pandas as pd
import cv2

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
import matplotlib.font_manager as fm
from matplotlib.collections import QuadMesh
from pandas import DataFrame
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
(X_train_1, Y_train_1), (X_test_1, Y_test_1) = keras.datasets.mnist.load_data()
data = pd.read_csv("A_Z Handwritten Data.csv").astype('float32')

# print(y_train.shape)

# print(len(X_train))
# print(len(X_test))

# print(X_train[0].shape)

# print(X_train[0])
# plt.show(X_train[0])
'''
y_train[0]
X_train = X_train / 255
X_test = X_test / 255
X_train[0]
X_train_flattened = X_train.reshape(len(X_train), 28*28)
X_test_flattened = X_test.reshape(len(X_test), 28*28)
X_train_flattened.shape
X_train_flattened[0]
'''
'''
# simple neural network with no hidden layers
model_nhl = keras.Sequential([keras.layers.Dense(10, input_shape=(784,), activation='sigmoid')])

model_nhl.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model_nhl.fit(X_train_flattened, y_train, epochs=5)

model_nhl.evaluate(X_test_flattened, y_test)

model_nhl.save('CNN_nhl_1.model')

y_predicted = model_nhl.predict(X_test_flattened)
y_predicted[0]
#plt.matshow(X_test[0])
np.argmax(y_predicted[0])
y_predicted_labels = [np.argmax(i) for i in y_predicted]
y_predicted_labels[:5]
cm = tf.math.confusion_matrix(labels=y_test, predictions=y_predicted_labels)
cm

plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()

# Using hidden layer
model_hl = keras.Sequential([keras.layers.Dense(1000, input_shape=(784,), activation='relu'),
                            keras.layers.Dense(100, activation='sigmoid'),
                            keras.layers.Dense(10, activation='sigmoid')])

model_hl.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model_hl.fit(X_train_flattened, y_train, epochs=5)

model_hl.evaluate(X_test_flattened, y_test)

model_hl.save('CNN_hl_1.model')

y_predicted = model_hl.predict(X_test_flattened)
y_predicted_labels = [np.argmax(i) for i in y_predicted]
cm = tf.math.confusion_matrix(labels=y_test, predictions=y_predicted_labels)

plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()

# Using Flatten layer
model_fl = keras.Sequential([keras.layers.Flatten(input_shape=(28, 28)),
                          keras.layers.Dense(100, activation='relu'),
                          keras.layers.Dense(10, activation='sigmoid')])

model_fl.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model_fl.fit(X_train, y_train, epochs=10)
model_fl.evaluate(X_test,y_test)

model_fl.save('CNN_fl_1.model')

y_predicted = model_fl.predict(X_test)
y_predicted_labels = [np.argmax(i) for i in y_predicted]
cm = tf.math.confusion_matrix(labels=y_test,predictions=y_predicted_labels)

plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()

'''
# letters
'''
X = data.drop('0',axis = 1)
y = data['0']
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size = 0.2)
X_train_2, X_test_2, Y_train_2, Y_test_2 = train_test_split(X, y, test_size = 0.2)
train_x = np.reshape(train_x.values, (train_x.shape[0], 28,28))
test_x = np.reshape(test_x.values, (test_x.shape[0], 28,28))
print("Train data shape: ", train_x.shape)
print("Test data shape: ", test_x.shape)
word_dict = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',
             16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X', 24:'Y',25:'Z'}
y_int = np.int0(y)
count = np.zeros(26, dtype='int')
for i in y_int:
    count[i] +=1
alphabets = []
for i in word_dict.values():
    alphabets.append(i)
fig, ax = plt.subplots(1,1, figsize=(10,10))
ax.barh(alphabets, count)
plt.xlabel("Number of elements ")
plt.ylabel("Alphabets")
plt.grid()
#plt.show()
shuff = shuffle(train_x[:100])
fig, ax = plt.subplots(3,3, figsize = (10,10))
axes = ax.flatten()
for i in range(9):
    _, shu = cv2.threshold(shuff[i], 30, 200, cv2.THRESH_BINARY)
    axes[i].imshow(np.reshape(shuff[i], (28,28)), cmap="Greys")
#plt.show()

train_X = train_x.reshape(train_x.shape[0],train_x.shape[1], train_x.shape[2],1)
print("New shape of train data: ", train_x.shape)
test_X = test_x.reshape(test_x.shape[0], test_x.shape[1], test_x.shape[2],1)
print("New shape of train data: ", test_x.shape)
train_yOHE = tf.keras.utils.to_categorical(train_y, num_classes = 26, dtype='int')
print("New shape of train labels: ", train_yOHE.shape)
test_yOHE = tf.keras.utils.to_categorical(test_y, num_classes = 26, dtype='int')
print("New shape of test labels: ", test_yOHE.shape)
'''
'''
model_l = Sequential()
model_l.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28,28, 1)))
model_l.add(MaxPool2D(pool_size=(2, 2), strides=2))
model_l.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding = 'same'))
model_l.add(MaxPool2D(pool_size=(2, 2), strides=2))
model_l.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding = 'valid'))
model_l.add(MaxPool2D(pool_size=(2, 2), strides=2))
model_l.add(Flatten())
model_l.add(Dense(64, activation ="relu"))
model_l.add(Dense(128, activation ="relu"))
model_l.add(Dense(26, activation ="softmax"))
model_l.compile(optimizer = 'Adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(train_x[0].shape)
history = model_l.fit(train_X, train_yOHE, epochs=1,  validation_data = (test_X, test_yOHE))
model_l.summary()

model_l.save('CNN_l_1.model')

print("The validation accuracy is :", history.history['val_accuracy'])
print("The training accuracy is :", history.history['accuracy'])
print("The validation loss is :", history.history['val_loss'])
print("The training loss is :", history.history['loss'])

y_predicted = model_l.predict(test_X)
y_predicted_labels = [np.argmax(i) for i in y_predicted]
cm = tf.math.confusion_matrix(labels=test_y,predictions=y_predicted_labels)

plt.figure(figsize = (10,7))
#df_col=(cm-cm.mean())/cm.std()
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()
'''

# merging models

#modelA = tf.keras.load_model(model_l)
#modelB = tf.keras.load_model(model_hl)

#modelA = model_l
#modelB = model_hl

#new_input = Input((dim1,dim2,2000)) # I want a fixed timestep now, say 2000

# reset layer names to avoid name conflicts
'''
modelA.name = "A"
modelB.name = "B"

out1 = modelA(new_input)
out2 = modelB(new_input)

new_model = Model([new_input], [out1,out2])

res = new_model.predict(feat) # will print correct result
new_model.save("new_model.h5") #save to local
'''
'''
mergedOut = Add()([modelA.output,modelB.output])
mergedOut = Flatten()(mergedOut)
mergedOut = Dense(256, activation='relu')(mergedOut)
mergedOut = Dropout(.5)(mergedOut)
mergedOut = Dense(128, activation='relu')(mergedOut)
mergedOut = Dropout(.35)(mergedOut)

# output layer
mergedOut = Dense(5, activation='softmax')(mergedOut)
newModel = Model([modelA.input,modelB.input], mergedOut)

newModel.fit([X_train_1, X_train_2], Y_train_1, Y_train_2, epochs = 5, validation_data = (X_test_1, X_test_2, Y_test_1, Y_test_2))
input_shape = (28,28)
commonInput = Input(input_shape)

out1 = modelA(commonInput)
out2 = modelB(commonInput)

mergedOut = Add()([out1,out2])
oneInputModel = Model(commonInput,mergedOut)
'''

'''
iter = 0
train2_y = []
target = 1
for iter1 in range(297960):
    for iter2 in range(26):
        iter += 1
        if train_yOHE[iter1, iter2] == target:
            train2_y.append(chr(65 + iter%26))

train1_y = []
for iter1 in range(60000):
        train1_y.append(str(y_train[iter1]))

print(len(train2_y))
print(type(train2_y[0]))
print(len(train1_y))
print(type(train1_y[0]))


#train_yOHE = np.reshape(train_yOHE, (-1,1))
#train_y = np.reshape(train_y, (-1,26))
labels = train1_y + train2_y

inputA = Input(shape=(28,28))
inputB = Input(shape=(28,28))
print(len(labels))
print(type(labels[0]))
# the first branch operates on the first input
x = Dense(1000, activation="relu")(inputA)
x = Dense(100, activation="relu")(x)
x = Dense(10, activation="relu")(x)
x = Model(inputs=inputA, outputs=x)

# the second branch opreates on the second input
y = Dense(1000, activation="relu")(inputB)
y = Dense(100, activation="sigmoid")(y)
y = Dense(10, activation="sigmoid")(y)
y = Model(inputs=inputB, outputs=y)

# combine the output of the two branches
combined = concatenate([x.output, y.output])

# apply a FC layer and then a regression prediction on the
# combined outputs
z = Dense(2, activation="relu")(combined)
z = Dense(1, activation="linear")(z)

# our model will accept the inputs of the two branches and
# then output a single value

model_c = Model(inputs=[x.input, y.input], outputs=z)

model_c.compile(optimizer = 'Adam', loss='categorical_crossentropy', metrics=['accuracy'])


print(X_train.shape)
print(type(X_train[0]))
print(train_X.shape)
print(type(train_X[0]))
'''
'''
model_c.fit([X_train, train_X], labels, epochs=1,
            validation_split=0.15, callbacks=[EarlyStopping(monitor='val_loss', patience=5)])
'''


'''
(img_train, img_test, img_validation), metadata = tfds.as_numpy(tfds.load(
    'binary_alpha_digits',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    as_supervised=True,
))
'''
#print(img_train.shape)
#print(label_train.shape)
#print(img_train[0].shape)
#print(label_train.shape)
#print(img_train[0])
#print(label_train[0])

'''
model_lc = keras.Sequential([keras.layers.Dense(1000, input_shape=(28, 28, 1), activation='relu'),
                            keras.layers.Dense(100, activation='sigmoid'),
                            keras.layers.Dense(10, activation='sigmoid')])

model_lc.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model_lc.fit(img_train, label_train, epochs=1)

model_lc.evaluate(img_test, label_test)

model_lc.save('CNN_lc_1.model')

y_predicted = model_lc.predict(label_test)
y_predicted_labels = [np.argmax(i) for i in y_predicted]
cm = tf.math.confusion_matrix(labels=y_test, predictions=y_predicted_labels)

plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()


model_lc = Sequential()
model_lc.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(20, 16, 1)))
model_lc.add(MaxPool2D(pool_size=(2, 2), strides=2))
model_lc.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding = 'same'))
model_lc.add(MaxPool2D(pool_size=(2, 2), strides=2))
model_lc.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding = 'valid'))
model_lc.add(MaxPool2D(pool_size=(2, 2), strides=2))
model_lc.add(Flatten())
model_lc.add(Dense(64, activation ="relu"))
model_lc.add(Dense(128, activation ="relu"))
model_lc.add(Dense(26, activation ="softmax"))
model_lc.compile(optimizer = 'Adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model_lc.fit(img_train, epochs=1,  validation_data = img_test)
model_lc.summary()

model_lc.save('CNN_lc_1.model')

print("The validation accuracy is :", history.history['val_accuracy'])
print("The training accuracy is :", history.history['accuracy'])
print("The validation loss is :", history.history['val_loss'])
print("The training loss is :", history.history['loss'])

y_predicted = model_lc.predict(img_test)
y_predicted_labels = [np.argmax(i) for i in y_predicted]
cm = tf.math.confusion_matrix(labels=img_test,predictions=y_predicted_labels)

plt.figure(figsize = (10,7))
#df_col=(cm-cm.mean())/cm.std()
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()
'''

# final try

def load_az_dataset():
    # initialize the list of data and labels
    dataaz = []
    labelsaz = []
    data_az = pd.read_csv("A_Z Handwritten Data.csv").astype('float32')
    data_az.rename(columns={'0': 'label'}, inplace=True)

    # Splite data the X - Our data , and y - the prdict label
    x = data_az.drop('label', axis=1)
    y = data_az['label']
    data_az_tr, data_az_ts, labels_az_tr, labels_az_ts = train_test_split(x, y)

    standard_scaler = MinMaxScaler()
    standard_scaler.fit(data_az_tr)

    data_az_tr = standard_scaler.transform(data_az_tr)
    data_az_ts = standard_scaler.transform(data_az_ts)

    data_az_tr = data_az_tr.reshape(data_az_tr.shape[0], 28, 28, 1).astype('float32')  # zmiana kształtu na 3 wymiary
    data_az_ts = data_az_ts.reshape(data_az_ts.shape[0], 28, 28, 1).astype('float32')  # zmiana kształtu na 3 wymiary
    print(labels_az_tr[0])
    # labels_az_tr = np_utils.to_categorical(labels_az_tr)
    # labels_az_ts = np_utils.to_categorical(labels_az_ts)
    print(labels_az_tr[0])
    labels_az_tr = np.array(labels_az_tr).astype(int)
    labels_az_ts = np.array(labels_az_ts).astype(int)
    print(labels_az_tr[0])
    print(labels_az_tr[1])
    print(labels_az_tr[2])
    print(labels_az_tr[3])
    print(labels_az_tr[0])

    # return a 2-tuple of the A-Z data and labels
    return data_az_tr, labels_az_tr, data_az_ts, labels_az_ts


def load_zero_nine_dataset():
    # load the MNIST dataset and stack the training data and testing
    # data together (we'll create our own training and testing splits
    # later in the project)
    ((train_data, train_labels), (test_data, test_labels)) = mnist.load_data()
    data = np.vstack([train_data, test_data])
    labels = np.hstack([train_labels, test_labels])
    print(train_labels[0])
    train_data = train_data.reshape(train_data.shape[0], 28, 28, 1).astype('float32')
    test_data = test_data.reshape(test_data.shape[0], 28, 28, 1).astype('float32')

    return train_data, train_labels, test_data, test_labels


# load all datasets
(az_data_tr, az_labels_tr, az_data_ts, az_labels_ts) = load_az_dataset()
(digits_data_tr, digits_labels_tr, digits_data_ts, digits_labels_ts) = load_zero_nine_dataset()


# the MNIST dataset occupies the labels 0-9, so let's add 10 to every A-Z label to ensure the A-Z characters are not incorrectly labeled as digits
az_labels_tr += 10
az_labels_ts += 10

data_tr = np.concatenate((az_data_tr, digits_data_tr), axis=0)
data_ts = np.concatenate((az_data_ts, digits_data_ts), axis=0)
labels_tr = np.hstack([az_labels_tr, digits_labels_tr])
labels_ts = np.hstack([az_labels_ts, digits_labels_ts])

model_lc = Sequential()
model_lc.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model_lc.add(MaxPool2D(pool_size=(2, 2), strides=2))
model_lc.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model_lc.add(MaxPool2D(pool_size=(2, 2), strides=2))
model_lc.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
model_lc.add(MaxPool2D(pool_size=(2, 2), strides=2))
model_lc.add(Flatten())
model_lc.add(Dense(64, activation="relu"))
model_lc.add(Dense(128, activation="relu"))
model_lc.add(Dense(36, activation="softmax"))
# optimizer: adam , sgd , rmsprop, SGD(lr=1e-4, momentum=0.9)
model_lc.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])  # run_eagerly=True)

history = model_lc.fit(data_tr, labels_tr, epochs=25, validation_data=(data_ts, labels_ts), use_multiprocessing=True)

model_lc.summary()

model_lc.evaluate(data_ts, labels_ts, use_multiprocessing=True)

model_lc.save('CNN_l_d_2.model')

print("The validation accuracy is :", history.history['val_accuracy'])
print("The training accuracy is :", history.history['accuracy'])
print("The validation loss is :", history.history['val_loss'])
print("The training loss is :", history.history['loss'])

# model_lc = tf.keras.models.load_model('CNN_lc_1.model')

y_predicted = model_lc.predict(data_ts, use_multiprocessing=True)
y_predicted_labels = [np.argmax(i) for i in y_predicted]
cm = tf.math.confusion_matrix(labels=labels_ts, predictions=y_predicted_labels)
c_m = confusion_matrix(labels_ts, y_predicted_labels,  labels=None, sample_weight=None, normalize='true')

names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
         'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

plt.figure(figsize=(10, 7))
heat_map = sn.heatmap(c_m, annot=True, fmt='.3f', xticklabels=names, yticklabels=names, cbar=False)
heat_map.set_yticklabels(heat_map.get_yticklabels(), rotation=35)
sn.set(font_scale=0.8)

plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()
cv2.waitKey(1)

plt.figure(figsize=(10, 5))
plt.suptitle('CNN_l_d_1.model training ', fontsize=20)

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
plt.show()
