import time

import matplotlib.pyplot as plt
# import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
from tensorflow import keras


def plot_time_per_epoch(model, epochs, times):
    plt.plot(epochs, times, label='Time per epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Time (seconds)')
    plt.legend()
    plt.show()


model = Sequential()
model.add(Dense(100, activation='relu', input_shape=(784,)))
model.add(Dense(10, activation='softmax'))

model.compile(loss="sparse_categorical_crossentropy", optimizer='adam', metrics=['accuracy'])

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32').reshape(60000,784)
x_test = x_test.astype('float32').reshape(10000,784)
x_train /= 255
x_test /= 255

# y_train = keras.utils.to_categorical(y_train, 10)
# y_test = keras.utils.to_categorical(y_test, 10)

epochs = []
times = []

for epoch in range(10):
    start = time.time()
    model.fit(x_train, y_train, epochs=4, batch_size=32, verbose=2)
    end = time.time()

    times.append(end - start)
    epochs.append(epoch)

plot_time_per_epoch(model, epochs, times)
