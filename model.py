#!/usr/bin/python3
# -*- coding: utf-8 -*-
from keras.models import Model
from keras.layers import Conv2D, LeakyReLU, Dense, GlobalMaxPool2D, GlobalAveragePooling2D
from keras.layers import Input, Concatenate, MaxPool2D, AveragePooling2D, Flatten
from GlobalVarianceLayer import GlobalVarianceLayer
from VarianceLayer import VarianceLayer
from EdgeAndCenterExtractionLayer import EdgeAndCenterExtractionLayer
import numpy as np


def laplacian_group_initializer(shape, dtype=None):
    kernel = np.zeros(shape, dtype=dtype)
    if np.random.random() < 0.5 and kernel.shape[0] >= 3 and len(kernel.shape) == 2:
        kernel[int(kernel.shape[0] // 2) - 1, int(kernel.shape[1] // 2)] = 1
        kernel[int(kernel.shape[0] // 2) + 1, int(kernel.shape[1] // 2)] = 1
    if np.random.random() < 0.5 and kernel.shape[1] >= 3 and len(kernel.shape) == 2:
        kernel[int(kernel.shape[0] // 2), int(kernel.shape[1] // 2) - 1] = 1
        kernel[int(kernel.shape[0] // 2), int(kernel.shape[1] // 2) + 1] = 1
    kernel[tuple(map(lambda x: int(np.floor(x / 2)), kernel.shape))] = -np.sum(kernel)
    return kernel + np.random.normal(0.0, 0.005, shape) * 1.0


def create_model(input_shape, l1fc, l1fs, l2fc, l2fs, l3fc, l3fs, eac_size):
    inp = Input(shape=(input_shape[0], input_shape[1], 3))
    c1 = Conv2D(l1fc, kernel_size=l1fs, strides=2, use_bias=True, padding="same",
                data_format="channels_last", kernel_initializer=laplacian_group_initializer)(inp)
    l1 = LeakyReLU(alpha=0.2)(c1)
    eac1 = EdgeAndCenterExtractionLayer(width=eac_size)(l1)
    c2 = Conv2D(l2fc, kernel_size=l2fs, strides=2, use_bias=True, padding="same",
                data_format="channels_last")(l1)
    l2 = LeakyReLU(alpha=0.2)(c2)
    eac2 = EdgeAndCenterExtractionLayer(width=eac_size)(l2)
    c3 = Conv2D(l3fc, kernel_size=l3fs, strides=1, use_bias=True, padding="same",
                data_format="channels_last")(l2)
    eac3 = EdgeAndCenterExtractionLayer(width=eac_size)(c3)
    eac3_max_grid = MaxPool2D((eac_size, eac_size), strides=eac_size,
                              padding="valid", data_format="channels_last")(eac3)
    eac3_avg_grid = AveragePooling2D((eac_size, eac_size), strides=eac_size,
                                     padding="valid", data_format="channels_last")(eac3)
    features = [GlobalVarianceLayer()(c1),
                GlobalVarianceLayer()(c2),
                GlobalVarianceLayer()(c3),
                GlobalMaxPool2D(data_format="channels_last")(c1),
                GlobalMaxPool2D(data_format="channels_last")(c2),
                GlobalMaxPool2D(data_format="channels_last")(c3),
                GlobalAveragePooling2D(data_format="channels_last")(c1),
                GlobalAveragePooling2D(data_format="channels_last")(c2),
                GlobalAveragePooling2D(data_format="channels_last")(c3),
                GlobalMaxPool2D(data_format="channels_last")(eac1),
                GlobalMaxPool2D(data_format="channels_last")(eac2),
                GlobalMaxPool2D(data_format="channels_last")(eac3),
                GlobalAveragePooling2D(data_format="channels_last")(eac1),
                GlobalAveragePooling2D(data_format="channels_last")(eac2),
                GlobalAveragePooling2D(data_format="channels_last")(eac3),
                GlobalVarianceLayer()(eac1),
                GlobalVarianceLayer()(eac2),
                GlobalVarianceLayer()(eac3),
                Flatten()(VarianceLayer((eac_size, eac_size))(eac1)),
                Flatten()(VarianceLayer((eac_size, eac_size))(eac2)),
                Flatten()(VarianceLayer((eac_size, eac_size))(eac3)),
                GlobalVarianceLayer()(eac3_max_grid),
                GlobalVarianceLayer()(eac3_avg_grid),
                Flatten()(eac3_max_grid),
                Flatten()(eac3_avg_grid)
                ]
    feature_vector = Concatenate()(features)
    o = Dense(2, activation="softmax", use_bias=True, name="output")(feature_vector)
    return Model(inputs=inp, outputs=o)
