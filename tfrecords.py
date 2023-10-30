from EfficientNet import build_model
import tensorflow as tf
import os
import time
import mlflow
from glob import glob
import random
import math

class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


path = 'WasteImagesDataset'
IMG_SIZE = 224
batch_size = 32
epochs = 10


# with mlflow.start_run():
#     mlflow.tensorflow.autolog()

# Make TFrecords
def serialize_example(image, label):
    feature = {
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def make_tfrecords(path, record_file='images.tfrecords'):
    classes = os.listdir(path)
    with tf.io.TFRecordWriter(record_file) as writer:
        files_list = glob(path + '/*/*')
        random.shuffle(files_list)
        for filename in files_list:
            image_string = open(filename, 'rb').read()
            category = filename.split('/')[-2]
            label = classes.index(category)
            tf_example = serialize_example(image_string, label)
            writer.write(tf_example)


# make_tfrecords(path)


# Train with TFrecords
def _parse_image_function(example):
    image_feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }

    features = tf.io.parse_single_example(example, image_feature_description)
    image = tf.image.decode_jpeg(features['image'], channels=3)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])

    label = tf.cast(features['label'], tf.int32)

    return image, label


def read_dataset(filename, batch_size):
    dataset = tf.data.TFRecordDataset(filename)
    dataset = dataset.map(_parse_image_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.shuffle(500)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    # dataset = dataset.repeat()
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset


dataset = read_dataset('images.tfrecords', batch_size)

num_classes = len(os.listdir(path))
num_images = len(glob(path + '/*/*'))
model = build_model(num_classes)
# steps_per_epoch=math.ceil(num_images/batch_size)
time_callback = TimeHistory()
model.fit(dataset, epochs=epochs, batch_size=32, verbose=2, callbacks=[time_callback])
_, acc = model.evaluate(dataset, verbose=0)

exp_name = "input_pipeline_final"
mlflow.set_experiment(exp_name)
# Log the epoch times in MLflow
with mlflow.start_run(run_name='tfrecords'):
    for epoch, epoch_time in enumerate(time_callback.times):
        mlflow.log_param('epoch', epochs)
        mlflow.log_metric(f'epoch_{epoch}_time', epoch_time)
        mlflow.log_metric('accuracy', acc)
