#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import librosa
import librosa.display
import librosa.feature
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout
import tensorflow.keras.backend as K
from tensorflow.keras import optimizers

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

K.clear_session()

tf_config = tf.ConfigProto( allow_soft_placement=True )
tf_config.gpu_options.allow_growth = True
sess = tf.Session(config=tf_config) 
K.set_session(sess)
opt = optimizers.Adam(lr=0.0001)

X, y = unison_shuffled_copies(np.load("X.npy"), np.load("y.npy"))
X = X.reshape((X.shape[0], X.shape[1], X.shape[2], 1))

mean = np.mean(np.mean(X, axis=2), axis=1)
std = np.std(np.std(X, axis=2), axis=1)
for i in range(X.shape[0]):
    X[i] -= mean[i]
    X[i] /= std[i]
    
val_num = 400
N = X.shape[0]
Xtr = X[0:N - val_num]
ytr = y[0:N - val_num]
Xts = X[N - val_num:]
yts = y[N - val_num:]

input_shape = (128, 128, 1)

model = Sequential()
model.add(Conv2D(64, kernel_size=(8, 8), strides=(1, 1),
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer=opt,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

hist = model.fit(Xtr, ytr, epochs=30, batch_size=32, validation_data=(Xts,yts))
print("Hello world")
