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
import sys
from PyQt5.QtWidgets import QApplication, QWidget
from multiprocessing import Process, Queue

ex = Experiment("UnsharpDetector")
ex.observers.append(MongoObserver.create(url=mongo_url, db_name=db_name))
ex.captured_out_filter = apply_backspaces_and_linefeeds


'''class TrainingPreview(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Training preview")
        self.show()
        self.result_queue = Queue()
        self.train_process = Process(target=train, args=(self.result_queue,))
        self.train_process.start()
        self.training_is_running = True
        while self.training_is_running:
            print(self.result_queue.get())
        self.train_process.join()

    def __del__(self):
        if self.train_process.is_alive():
            self.train_process.join()'''


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
    def on_batch_end(self, batch, logs={}):
        log_training_performance(loss=logs.get("loss"), accuracy=logs.get("acc"))

    def on_epoch_end(self, batch, logs={}):
        log_validation_performance(val_loss=logs.get("val_loss"), val_accuracy=logs.get("val_acc"))


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
    use_gui = True


@ex.capture
def validate(model, x, y, bs):
    prediction = model.predict(x, batch_size=bs)
    validation_loss = K.eval(K.mean(categorical_crossentropy(K.constant(y), K.constant(prediction))))
    log_validation_performance(val_loss=validation_loss)
    return validation_loss


@ex.capture
def train(input_size, bs, lr, lr_decay, l1fc, l1fs, l2fc, l2fs, l3fc, l3fs, image_folders):
    optimizer = Adam(lr, decay=lr_decay)
    model = create_model(input_size, l1fc, l1fs, l2fc, l2fs, l3fc, l3fs)
    model.compile(optimizer, loss=categorical_crossentropy, metrics=["accuracy"])
    data_generator = UnsharpTrainingDataGenerator(image_folders, batch_size=bs, target_size=input_size)
    validation_data_provider = UnsharpValidationDataProvider("validation_data", batch_size=bs, target_size=input_size)
    while True:
        model.fit_generator(generator=data_generator,
                            validation_data=validation_data_provider,
                            callbacks=[ModelCheckpoint("unsharpDetectorWeights.hdf5", monitor='val_loss',
                                                       save_best_only=True, mode='auto', period=1),
                                       LogPerformance()],
                            use_multiprocessing=True,
                            workers=8)
    #for x, y in data_generator:
    #    train_loss, train_acc = model.train_on_batch(x, y)
    #    log_training_performance(loss=train_loss, lr=lr)
        #val_loss, val_acc = model.test_on_batch(x, y)
        #queue.put(train_loss)
    #    print(validate(model, x, y))


@ex.automain
def run(use_gui):
    #if use_gui:
    #    app_object = QApplication(sys.argv)
    #    window = TrainingPreview()
    #    sys.exit(app_object.exec_())
    #else:
        train()
