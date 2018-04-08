#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import division, unicode_literals, print_function
from sacred import Experiment
from keras.optimizers import Adam
from model import create_model
from TrainingDataGenerator import UnsharpTrainingDataGenerator

ex = Experiment("UnsharpDetector")


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


@ex.automain
def train(input_size, bs, lr, lr_decay, l1fc, l1fs, l2fc, l2fs, l3fc, l3fs):
    optimizer = Adam(lr, decay=lr_decay)
    model = create_model(input_size, l1fc, l1fs, l2fc, l2fs, l3fc, l3fs)
    model.compile(optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    data_generator = UnsharpTrainingDataGenerator([
        "../../Bilder/Backgrounds/"
    ], batch_size=bs, target_size=input_size)
    for x, y in data_generator:
        model.fit(x, y, batch_size=bs, epochs=1)
