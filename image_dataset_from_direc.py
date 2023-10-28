from EfficientNet import build_model
import tensorflow as tf
import os
import time
import mlflow


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
epochs = 5

# with mlflow.start_run():
#     mlflow.tensorflow.autolog()

keras_ds = tf.keras.preprocessing.image_dataset_from_directory(path, batch_size=batch_size,
                                                               image_size=(IMG_SIZE, IMG_SIZE))
keras_ds = keras_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

num_classes = len(os.listdir(path))
model = build_model(num_classes)

time_callback = TimeHistory()
model.fit(keras_ds, epochs=epochs, batch_size=32, verbose=2, callbacks=[time_callback])
_, acc = model.evaluate(keras_ds, verbose=0)

exp_name = "input_pipeline"
mlflow.set_experiment(exp_name)
# Log the epoch times in MLflow
with mlflow.start_run(run_name='img_ds_dir'):
    for epoch, epoch_time in enumerate(time_callback.times):
        mlflow.log_param('epoch', epochs)
        mlflow.log_metric(f'epoch_{epoch}_time', epoch_time)
        mlflow.log_metric('accuracy', acc)
