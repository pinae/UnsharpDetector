#!/usr/bin/python3
# -*- coding: utf-8 -*-
from keras.models import Model
from keras.layers import Conv2D, LeakyReLU, Dense, Flatten, GlobalMaxPool2D, GlobalAveragePooling2D
from keras.layers import Input, Lambda, Concatenate
from VarianceLayer import VarianceLayer


def create_model(input_shape, l1fc, l1fs, l2fc, l2fs, l3fc, l3fs):
    inp = Input(shape=(input_shape[0], input_shape[1], 3))
    c1 = Conv2D(l1fc, kernel_size=l1fs, strides=2, use_bias=True, padding="same",
                data_format="channels_last")(inp)
    l1 = LeakyReLU(alpha=0.2)(c1)
    c2 = Conv2D(l2fc, kernel_size=l2fs, strides=2, use_bias=True, padding="same",
                data_format="channels_last")(l1)
    l2 = LeakyReLU(alpha=0.2)(c2)
    c3 = Conv2D(l3fc, kernel_size=l3fs, strides=1, use_bias=True, padding="same",
                data_format="channels_last")(l2)
    v1 = VarianceLayer()(c1)
    v2 = VarianceLayer()(c2)
    v3 = VarianceLayer()(c3)
    max1 = GlobalMaxPool2D(data_format="channels_last")(c1)
    max2 = GlobalMaxPool2D(data_format="channels_last")(c2)
    max3 = GlobalMaxPool2D(data_format="channels_last")(c3)
    avg1 = GlobalAveragePooling2D(data_format="channels_last")(c1)
    avg2 = GlobalAveragePooling2D(data_format="channels_last")(c2)
    avg3 = GlobalAveragePooling2D(data_format="channels_last")(c3)
    fv = Concatenate()([max1, max2, max3, avg1, avg2, avg3, v1, v2, v3])
    o = Dense(2, activation="softmax", use_bias=True, name="output")(fv)
    return Model(inputs=inp, outputs=o)
