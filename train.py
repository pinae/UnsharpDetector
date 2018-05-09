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

ex = Experiment("UnsharpDetector")
ex.observers.append(MongoObserver.create(url=mongo_url, db_name=db_name))
ex.captured_out_filter = apply_backspaces_and_linefeeds


@ex.capture
def log_training_performance(_run, loss, accuracy):
    _run.log_scalar("loss", float(loss))
    _run.log_scalar("accuracy", float(accuracy))


@ex.capture
def log_validation_performance(_run, val_loss, val_accuracy):
    _run.log_scalar("validation_loss", float(val_loss))
    _run.log_scalar("validation_accuracy", float(val_accuracy))
    _run.result = float(val_accuracy)


class LogPerformance(Callback):
    def __init__(self, model, gui_queue, data_generator, bs):
        super().__init__()
        self.model = model
        self.data_generator = data_generator
        self.gui_queue = gui_queue
        self.bs = bs

    def on_batch_end(self, batch, logs={}):
        log_training_performance(loss=logs.get("loss"), accuracy=logs.get("acc"))

    def on_epoch_end(self, epoch, logs={}):
        log_validation_performance(val_loss=logs.get("val_loss"), val_accuracy=logs.get("val_acc"))
        if self.gui_queue:
            x, y = self.data_generator.__getitem__(0)
            prediction = self.model.predict(x, batch_size=self.bs)
            self.gui_queue.put({'x': x, 'y': y, 'prediction': prediction, 'epoch': epoch})


@ex.config
def config():
    input_size = (256, 256)
    bs = 10
    lr = 0.0001
    lr_decay = 0.0
    l1fc = 32
    l1fs = (9, 9)
    l2fc = 32
    l2fs = (3, 3)
    l3fc = 32
    l3fs = (3, 3)
    image_folders = [
        "../../Bilder/gesammelte Landschaftsbilder/",
        "../../Bilder/kleine Landschaftsbilder/",
        "../../Bilder/Hintergrundbilder - schöne Landschaften/"
    ]
    epochs = 100
    use_gui = True


@ex.capture
def validate(model, x, y, bs):
    prediction = model.predict(x, batch_size=bs)
    validation_loss = K.eval(K.mean(categorical_crossentropy(K.constant(y), K.constant(prediction))))
    log_validation_performance(val_loss=validation_loss)
    return validation_loss


@ex.capture
def train(gui_queue, input_size, bs, lr, lr_decay, l1fc, l1fs, l2fc, l2fs, l3fc, l3fs, image_folders, epochs):
    optimizer = Adam(lr, decay=lr_decay)
    model = create_model(input_size, l1fc, l1fs, l2fc, l2fs, l3fc, l3fs)
    model.compile(optimizer, loss=categorical_crossentropy, metrics=["accuracy"])
    data_generator = UnsharpTrainingDataGenerator(image_folders, batch_size=bs, target_size=input_size)
    validation_data_provider = UnsharpValidationDataProvider("validation_data", batch_size=bs, target_size=input_size)
    for epoch in range(epochs):
        model.fit_generator(generator=data_generator,
                            validation_data=validation_data_provider,
                            callbacks=[ModelCheckpoint("unsharpDetectorWeights.hdf5", monitor='val_loss',
                                                       save_best_only=True, mode='auto', period=1),
                                       LogPerformance(model, gui_queue, data_generator, bs)],
                            epochs=epochs,
                            use_multiprocessing=True,
                            workers=8)


@ex.automain
def run(use_gui):
    gui_process = None
    gui_queue = None
    if use_gui:
        from gui import init_gui
        gui_queue, gui_process = init_gui()
    train(gui_queue)
    if gui_process:
        gui_process.join()
