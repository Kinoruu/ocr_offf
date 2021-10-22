import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import os
import pandas as pd
import cv2

from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from keras.layers import *
from keras.models import Model
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow import keras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
(X_train_1, Y_train_1), (X_test_1, Y_test_1) = keras.datasets.mnist.load_data()
data = pd.read_csv("A_Z Handwritten Data.csv").astype('float32')

print(y_train.shape)

print(len(X_train))
print(len(X_test))

print(X_train[0].shape)

print(X_train[0])
#plt.show(X_train[0])
y_train[0]
X_train = X_train / 255
X_test = X_test / 255
X_train[0]
X_train_flattened = X_train.reshape(len(X_train), 28*28)
X_test_flattened = X_test.reshape(len(X_test), 28*28)
X_train_flattened.shape
X_train_flattened[0]
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

iter = 0
train2_y = []
target = 1
for iter1 in range(297960):
    for iter2 in range(26):
        iter += 1
        if train_yOHE[iter1, iter2] == target:
            train2_y.append(10 + iter%26)

train1_y = []
for iter1 in range(60000):
        train1_y.append(y_train[iter1])


#train_yOHE = np.reshape(train_yOHE, (-1,1))
#train_y = np.reshape(train_y, (-1,26))
labels = train1_y + train2_y

inputA = Input(shape=(28,28))
inputB = Input(shape=(28,28))

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
model_c.fit([X_train_flattened, train_X], labels, epochs=10,
            validation_split=0.15, callbacks=[EarlyStopping(monitor='val_loss', patience=5)])

print("x")