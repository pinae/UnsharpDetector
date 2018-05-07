#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import division, unicode_literals, print_function
from sacred import Experiment
from sacred.observers import MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds
from keras.optimizers import Adam
import keras.backend as K
from keras.losses import categorical_crossentropy
from model import create_model
from TrainingDataGenerator import UnsharpTrainingDataGenerator
from secret_settings import mongo_url, db_name

ex = Experiment("UnsharpDetector")
ex.observers.append(MongoObserver.create(url=mongo_url, db_name=db_name))
ex.captured_out_filter = apply_backspaces_and_linefeeds


@ex.capture
def log_training_performance(_run, loss, lr):
    _run.log_scalar("loss", float(loss))
    _run.log_scalar("lr", float(lr))


@ex.capture
def log_validation_performance(_run, val_loss):
    _run.log_scalar("validation_loss", float(val_loss))
    _run.result = float(val_loss)


@ex.config
def config():
    input_size = (256, 256)
    bs = 20
    lr = 0.0004
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


@ex.capture
def validate(model, x, y, bs):
    prediction = model.predict(x, batch_size=bs)
    validation_loss = K.eval(K.mean(categorical_crossentropy(K.constant(y), K.constant(prediction))))
    return validation_loss


@ex.automain
def train(input_size, bs, lr, lr_decay, l1fc, l1fs, l2fc, l2fs, l3fc, l3fs, image_folders):
    optimizer = Adam(lr, decay=lr_decay)
    model = create_model(input_size, l1fc, l1fs, l2fc, l2fs, l3fc, l3fs)
    model.compile(optimizer, loss=categorical_crossentropy, metrics=["accuracy"])
    data_generator = UnsharpTrainingDataGenerator(image_folders, batch_size=bs, target_size=input_size)
    for x, y in data_generator:
        model.fit(x, y, batch_size=bs, epochs=10)
        print(validate(model, x, y))
