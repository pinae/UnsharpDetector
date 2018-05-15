#!/usr/bin/python3
# -*- coding: utf-8 -*-
import keras.backend as K
from keras.engine.topology import Layer
from keras.models import Model
from keras.layers import Input
import numpy as np
import unittest


class EdgeAndCenterExtractionLayer(Layer):
    def __init__(self, **kwargs):
        super(EdgeAndCenterExtractionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(EdgeAndCenterExtractionLayer, self).build(input_shape)

    def call(self, x, **kwargs):
        half_y = K.cast(K.shape(x)[1] / 2, dtype="int32")
        half_x = K.cast(K.shape(x)[2] / 2, dtype="int32")
        e0 = x[:, 0:16, 0:16]
        e1 = x[:, half_y - 16:half_y + 16, 0:16]
        e2 = x[:, -16:, 0:16]
        e7 = x[:, 0:16, half_x - 16:half_x + 16]
        cn = x[:, half_y - 16:half_y + 16, half_x - 16:half_x + 16]
        e3 = x[:, -16:, half_x - 16:half_x + 16]
        e6 = x[:, 0:16, -16:]
        e5 = x[:, half_y - 16:half_y + 16, -16:]
        e4 = x[:, -16:, -16:]
        l1 = K.concatenate([e0, e1, e2], axis=1)
        l2 = K.concatenate([e7, cn, e3], axis=1)
        l3 = K.concatenate([e6, e5, e4], axis=1)
        return K.concatenate([l1, l2, l3], axis=2)

    def compute_output_shape(self, input_shape):
        return input_shape[0], 64, 64, input_shape[3]


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
        x = EdgeAndCenterExtractionLayer()(inp)
        model = Model(inputs=inp, outputs=x)
        keras_values = model.predict(data, batch_size=1)
        self.assertAlmostEqual(keras_values[0, 0, 0, 0], 13, places=4)
        self.assertAlmostEqual(keras_values[0, 17, 17, 0], 0, places=4)
        self.assertAlmostEqual(keras_values[0, 32, 32, 0], -9, places=4)
        self.assertAlmostEqual(keras_values[0, 32, 2, 0], -5, places=4)
        self.assertAlmostEqual(keras_values[0, 2, 32, 0], 7, places=4)
        self.assertAlmostEqual(keras_values[0, 63, 63, 0], 16, places=4)
        self.assertAlmostEqual(keras_values[0, 63, 32, 0], 2, places=4)
