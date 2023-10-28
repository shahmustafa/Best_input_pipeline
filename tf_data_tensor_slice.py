import time

import matplotlib.pyplot as plt
from EfficientNet import build_model
import tensorflow as tf
import scipy
import os
from glob import glob
import random
import mlflow


class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


def make_dataset(path, batch_size):
    def parse_image(filename):
        image = tf.io.read_file(filename)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
        return image

    def configure_for_performance(ds):
        ds = ds.shuffle(buffer_size=1000)
        ds = ds.batch(batch_size)
        # ds = ds.repeat()
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return ds

    classes = os.listdir(path)
    filenames = glob(path + '/*/*')
    random.shuffle(filenames)
    labels = [classes.index(name.split('/')[-2]) for name in filenames]

    filenames_ds = tf.data.Dataset.from_tensor_slices(filenames)
    images_ds = filenames_ds.map(parse_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    labels_ds = tf.data.Dataset.from_tensor_slices(labels)
    ds = tf.data.Dataset.zip((images_ds, labels_ds))
    ds = configure_for_performance(ds)

    return ds


path = 'WasteImagesDataset'
IMG_SIZE = 224
batch_size = 32
epochs = 5

dataset = make_dataset(path, batch_size=batch_size)
num_classes = len(os.listdir(path))
model = build_model(num_classes)

time_callback = TimeHistory()
model.fit(dataset, epochs=epochs, batch_size=32, verbose=2, callbacks=[time_callback])
_, acc = model.evaluate(dataset, verbose=0)

exp_name = "input_pipeline"
mlflow.set_experiment(exp_name)
# Log the epoch times in MLflow
with mlflow.start_run(run_name='tensor_slice'):
    for epoch, epoch_time in enumerate(time_callback.times):
        mlflow.log_param('epoch', epochs)
        mlflow.log_metric(f'epoch_{epoch}_time', epoch_time)
        mlflow.log_metric('accuracy', acc)
