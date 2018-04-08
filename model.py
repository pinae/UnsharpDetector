#!/usr/bin/python3
# -*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers import Conv2D, LeakyReLU, Dense, Flatten
from EdgeAndCenterExtractionLayer import get_edge_and_center_extraction_layer


def create_model(input_shape, l1fc, l1fs, l2fc, l2fs, l3fc, l3fs):
    model = Sequential()
    model.add(Conv2D(l1fc, kernel_size=l1fs, strides=2, use_bias=True, padding="same",
                     data_format="channels_last", input_shape=(input_shape[0], input_shape[1], 3)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(l2fc, kernel_size=l2fs, strides=2, use_bias=True, padding="same",
                     data_format="channels_last"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(l3fc, kernel_size=l3fs, strides=1, use_bias=True, padding="same",
                     data_format="channels_last"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(get_edge_and_center_extraction_layer())
    model.add(Flatten())
    model.add(Dense(2, activation="softmax", use_bias=True, name="output"))
    return model
