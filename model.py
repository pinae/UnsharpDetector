#!/usr/bin/python3
# -*- coding: utf-8 -*-
from keras.models import Model
from keras.layers import Conv2D, LeakyReLU, Dense, GlobalMaxPool2D, GlobalAveragePooling2D
from keras.layers import Input, Concatenate, MaxPool2D, AveragePooling2D, Flatten
from VarianceLayer import VarianceLayer
from EdgeAndCenterExtractionLayer import EdgeAndCenterExtractionLayer


def create_model(input_shape, l1fc, l1fs, l2fc, l2fs, l3fc, l3fs, eac_size):
    inp = Input(shape=(input_shape[0], input_shape[1], 3))
    c1 = Conv2D(l1fc, kernel_size=l1fs, strides=2, use_bias=True, padding="same",
                data_format="channels_last")(inp)
    l1 = LeakyReLU(alpha=0.2)(c1)
    c2 = Conv2D(l2fc, kernel_size=l2fs, strides=2, use_bias=True, padding="same",
                data_format="channels_last")(l1)
    l2 = LeakyReLU(alpha=0.2)(c2)
    c3 = Conv2D(l3fc, kernel_size=l3fs, strides=1, use_bias=True, padding="same",
                data_format="channels_last")(l2)
    eac = EdgeAndCenterExtractionLayer(width=eac_size)(c3)
    eac_max_grid = MaxPool2D((eac_size, eac_size), strides=eac_size,
                             padding="valid", data_format="channels_last")(eac)
    eac_avg_grid = AveragePooling2D((eac_size, eac_size), strides=eac_size,
                                    padding="valid", data_format="channels_last")(eac)
    features = [VarianceLayer()(c1),
                VarianceLayer()(c2),
                VarianceLayer()(c3),
                GlobalMaxPool2D(data_format="channels_last")(c1),
                GlobalMaxPool2D(data_format="channels_last")(c2),
                GlobalMaxPool2D(data_format="channels_last")(c3),
                GlobalAveragePooling2D(data_format="channels_last")(c1),
                GlobalAveragePooling2D(data_format="channels_last")(c2),
                GlobalAveragePooling2D(data_format="channels_last")(c3),
                GlobalMaxPool2D(data_format="channels_last")(eac),
                GlobalAveragePooling2D(data_format="channels_last")(eac),
                VarianceLayer()(eac),
                Flatten()(eac_max_grid),
                Flatten()(eac_avg_grid)
                ]
    feature_vector = Concatenate()(features)
    o = Dense(2, activation="softmax", use_bias=True, name="output")(feature_vector)
    return Model(inputs=inp, outputs=o)
