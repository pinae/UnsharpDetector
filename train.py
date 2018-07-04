#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import division, unicode_literals, print_function
from sacred import Experiment
from sacred.observers import MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds
from keras.optimizers import Adam
import keras.backend as K
from keras.losses import categorical_crossentropy
from keras.callbacks import Callback, ModelCheckpoint
from model import create_model
from TrainingDataGenerator import UnsharpTrainingDataGenerator
from ValidationDataProvider import UnsharpValidationDataProvider
from secret_settings import mongo_url, db_name
import json
import os

ex = Experiment("UnsharpDetector")
ex.observers.append(MongoObserver.create(url=mongo_url, db_name=db_name))
ex.captured_out_filter = apply_backspaces_and_linefeeds
last_result = None


@ex.capture
def log_training_performance_batch(_run, loss, accuracy):
    _run.log_scalar("batch_loss", float(loss))
    _run.log_scalar("batch_accuracy", float(accuracy))


@ex.capture
def log_training_performance_epoch(_run, loss, accuracy):
    _run.log_scalar("loss", float(loss))
    _run.log_scalar("accuracy", float(accuracy))


@ex.capture
def log_validation_performance(_run, val_loss, val_accuracy):
    _run.log_scalar("validation_loss", float(val_loss))
    _run.log_scalar("validation_accuracy", float(val_accuracy))
    _run.result = float(val_accuracy)
    global last_result
    last_result = float(val_accuracy)


@ex.capture
def log_lr(_run, lr):
    _run.log_scalar("lr", float(lr))


class LogPerformance(Callback):
    def __init__(self, model, gui_callback, data_generator, bs):
        super().__init__()
        self.model = model
        self.data_generator = data_generator
        self.gui_callback = gui_callback
        self.bs = bs
        self.epoch = 0

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch = epoch

    def on_batch_begin(self, batch, logs=None):
        if self.gui_callback and batch % 10 == 0:
            x, y = self.data_generator.__getitem__(batch)
            prediction = self.model.predict(x, batch_size=self.bs)
            self.gui_callback(x, y, prediction, self.epoch)

    def on_batch_end(self, batch, logs={}):
        log_training_performance_batch(loss=logs.get("loss"), accuracy=logs.get("acc"))

    def on_epoch_end(self, epoch, logs={}):
        lr = self.model.optimizer.lr
        decay = self.model.optimizer.decay
        iterations = self.model.optimizer.iterations
        lr_with_decay = lr / (1. + decay * K.cast(iterations, K.dtype(decay)))
        log_lr(lr=K.eval(lr_with_decay))
        log_training_performance_epoch(loss=logs.get("loss"), accuracy=logs.get("acc"))
        log_validation_performance(val_loss=logs.get("val_loss"), val_accuracy=logs.get("val_acc"))


@ex.config
def config():
    input_size = (256, 256)
    bs = 12
    lr = 0.002
    lr_decay = 0.005
    blur_rate = 0.5
    mask_rate = 0.2
    noise_rate = 0.2
    min_blur = 0.5
    min_shake = 2.5
    l1fc = 32
    l1fs = (9, 9)
    l1st = 2
    l2fc = 32
    l2fs = (3, 3)
    l2st = 2
    l3fc = 32
    l3fs = (3, 3)
    res_c = 0
    res_fc = 32
    res_fs = (3, 3)
    eac_size = 16
    image_folders = [
        "../../Bilder/gesammelte Landschaftsbilder/",
        "../../Bilder/kleine Landschaftsbilder/",
        "../../Bilder/Hintergrundbilder - schöne Landschaften/",
        "../../Bilder/Bilder der Woche/",
        "../../Bilder/Texturen/",
        "../../Bilder/Famous Photos/",
        "../../Bilder/Korea/",
        "../../Bilder/Urlaubsbilder/",
        "../../Bilder/Stephanie Waetjen/",
        "../../Bilder/Sharp Photos/"
    ]
    epochs = 50
    use_gui = True
    load_weights = False


@ex.capture
def validate(model, x, y, bs):
    prediction = model.predict(x, batch_size=bs)
    validation_loss = K.eval(K.mean(categorical_crossentropy(K.constant(y), K.constant(prediction))))
    log_validation_performance(val_loss=validation_loss)
    return validation_loss


@ex.capture
def get_model(input_size, l1fc, l1fs, l1st, l2fc, l2fs, l2st, l3fc, l3fs, eac_size, res_c, res_fc, res_fs):
    return create_model(input_size, l1fc, l1fs, l1st, l2fc, l2fs, l2st, l3fc, l3fs, eac_size, res_c, res_fc, res_fs)


@ex.capture
def get_model_config_settings(l1fc, l1fs, l1st, l2fc, l2fs, l2st, l3fc, l3fs, eac_size, res_c, res_fc, res_fs):
    return {
        "l1fc": l1fc, "l1fs": l1fs, "l1st": l1st,
        "l2fc": l2fc, "l2fs": l2fs, "l2st": l2st,
        "l3fc": l3fc, "l3fs": l3fs,
        "eac_size": eac_size,
        "res_c": res_c, "res_fc": res_fc, "res_fs": res_fs
    }


@ex.capture
def train(gui_callback, input_size, bs, lr, lr_decay, image_folders, epochs, load_weights,
          blur_rate, mask_rate, noise_rate, min_blur, min_shake):
    optimizer = Adam(lr, decay=lr_decay)
    model = get_model()
    model.compile(optimizer, loss=categorical_crossentropy, metrics=["accuracy"])
    print(model.summary())
    data_generator = UnsharpTrainingDataGenerator(image_folders, batch_size=bs, target_size=input_size,
                                                  blur_rate=blur_rate, mask_rate=mask_rate, noise_rate=noise_rate,
                                                  min_blur=min_blur, min_shake=min_shake)
    data_generator.on_epoch_end()
    validation_data_provider = UnsharpValidationDataProvider("validation_data", batch_size=bs, target_size=input_size)
    with open('unsharpDetectorSettings.json', 'w') as json_file:
        json_file.write(json.dumps(get_model_config_settings()))
    if load_weights and os.path.exists("unsharpDetectorWeights.hdf5"):
        model.load_weights("unsharpDetectorWeights.hdf5")
    else:
        model.save("unsharpDetectorWeights.hdf5", include_optimizer=True)
    model.fit_generator(generator=data_generator,
                        validation_data=validation_data_provider,
                        callbacks=[ModelCheckpoint("unsharpDetectorWeights.hdf5", monitor='val_loss',
                                                   save_best_only=False, mode='auto', period=1),
                                   LogPerformance(model, gui_callback, data_generator, bs)],
                        epochs=epochs,
                        use_multiprocessing=True,
                        workers=8, max_queue_size=30)


@ex.automain
def run(use_gui):
    gui_thread = None
    gui_callback = None
    if use_gui:
        from training_gui import init_gui
        gui_callback, feedback_queue, gui_thread = init_gui()
    train(gui_callback)
    if gui_thread:
        gui_thread.join()
    return last_result
