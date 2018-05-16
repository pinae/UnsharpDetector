#!/usr/bin/python3
# -*- coding: utf-8 -*-
import keras.backend as K
from keras.engine.topology import Layer
from keras.models import Model
from keras.layers import Input
import numpy as np
import unittest


class EdgeAndCenterExtractionLayer(Layer):
    def __init__(self, width, **kwargs):
        self.w = width
        super(EdgeAndCenterExtractionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(EdgeAndCenterExtractionLayer, self).build(input_shape)

    def call(self, x, **kwargs):
        batch_size = K.shape(x)[0]
        half_y = K.cast(K.shape(x)[1] / 2, dtype="int32")
        half_x = K.cast(K.shape(x)[2] / 2, dtype="int32")
        channel_count = K.shape(x)[3]
        e0 = x[:, 0:self.w, 0:self.w]
        e1 = x[:, half_y - self.w:half_y + self.w, 0:self.w]
        e2 = x[:, -self.w:, 0:self.w]
        e7 = x[:, 0:self.w, half_x - self.w:half_x + self.w]
        cn = x[:, half_y - self.w:half_y + self.w, half_x - self.w:half_x + self.w]
        e3 = x[:, -self.w:, half_x - self.w:half_x + self.w]
        e6 = x[:, 0:self.w, -self.w:]
        e5 = x[:, half_y - self.w:half_y + self.w, -self.w:]
        e4 = x[:, -self.w:, -self.w:]
        l1 = K.concatenate([e0, e1, e2], axis=1)
        l2 = K.concatenate([e7, cn, e3], axis=1)
        l3 = K.concatenate([e6, e5, e4], axis=1)
        return K.reshape(K.concatenate([l1, l2, l3], axis=2), (batch_size, 4 * self.w, 4 * self.w, channel_count))

    def compute_output_shape(self, input_shape):
        print(input_shape)
        print((input_shape[0], self.w * 4, self.w * 4, input_shape[3]))
        return input_shape[0], self.w * 4, self.w * 4, input_shape[3]


class TestVarianceLayer(unittest.TestCase):
    def test_Extraction(self):
        data = np.zeros((1, 256, 256, 3), dtype=np.float32)
        data[0, 0, 0, 0] = 13
        data[0, 17, 17, 0] = 8
        data[0, 128, 128, 0] = -9
        data[0, 128, 2, 0] = -5
        data[0, 2, 128, 0] = 7
        data[0, 255, 255, 0] = 16
        data[0, 255, 128, 0] = 2
        inp = Input(shape=(256, 256, 3))
        x = EdgeAndCenterExtractionLayer(16)(inp)
        model = Model(inputs=inp, outputs=x)
        keras_values = model.predict(data, batch_size=1)
        self.assertAlmostEqual(keras_values[0, 0, 0, 0], 13, places=4)
        self.assertAlmostEqual(keras_values[0, 17, 17, 0], 0, places=4)
        self.assertAlmostEqual(keras_values[0, 32, 32, 0], -9, places=4)
        self.assertAlmostEqual(keras_values[0, 32, 2, 0], -5, places=4)
        self.assertAlmostEqual(keras_values[0, 2, 32, 0], 7, places=4)
        self.assertAlmostEqual(keras_values[0, 63, 63, 0], 16, places=4)
        self.assertAlmostEqual(keras_values[0, 63, 32, 0], 2, places=4)
