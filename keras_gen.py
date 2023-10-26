import time

import matplotlib.pyplot as plt
from EfficientNet import build_model
import tensorflow as tf
import scipy
import os
import time


class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

def plot_time_per_epoch(model, epochs, times):
    plt.plot(epochs, times, label='Time per epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Time (seconds)')
    plt.legend()
    plt.show()


path = 'WasteImagesDataset'
IMG_SIZE = 224
epochs = 5

datagen = tf.keras.preprocessing.image.ImageDataGenerator()
dataset = datagen.flow_from_directory(path, (IMG_SIZE, IMG_SIZE), batch_size=32, class_mode='sparse')

num_classes = len(os.listdir(path))
model = build_model(num_classes)


# for epoch in range(10):
#     start = time.time()
#     model.fit(dataset, epochs=1, batch_size=32, verbose=2)
#     end = time.time()
#
#     times.append(end - start)
#     epochs.append(epoch)
#
# plot_time_per_epoch(model, epochs, times)

time_callback = TimeHistory()
model.fit(dataset, epochs=epochs, batch_size=32, verbose=2, callbacks=[time_callback])

plot_time_per_epoch(model, range(epochs), time_callback.times)


