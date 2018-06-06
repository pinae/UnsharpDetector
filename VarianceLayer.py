#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals
from keras import backend as K
from keras.engine.topology import Layer
from keras.models import Model
from keras.layers import Input
import numpy as np
import unittest


class VarianceLayer(Layer):
    def __init__(self, tile_size, **kwargs):
        self.tile_size = tile_size
        super(VarianceLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(VarianceLayer, self).build(input_shape)

    def call(self, x, **kwargs):
        means = K.pool2d(x, self.tile_size, strides=self.tile_size, padding="same",
                         pool_mode="avg", data_format="channels_last")
        mean_matrix = K.resize_images(means, self.tile_size[0], self.tile_size[1],
                                      data_format="channels_last")[:,
                      0:K.shape(x)[1], 0:K.shape(x)[2], :]
        quad_diff = (x - mean_matrix) ** 2
        return K.pool2d(quad_diff, self.tile_size, strides=self.tile_size, padding="same", pool_mode="avg")

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1] // self.tile_size[0], input_shape[2] // self.tile_size[1], input_shape[3]


class TestVarianceLayer(unittest.TestCase):
    def test_pool_mean(self):
        data = np.array([[[[1, 0], [2, 1], [3, -1]],
                          [[0, 1], [1, -2], [2, 1]],
                          [[-2, -1], [-1, -1], [3, 2]],
                          [[-2, -1], [-1, -1], [3, 2]]]], dtype=np.float32)
        x = K.variable(data, dtype=K.floatx())
        means = K.eval(K.pool2d(x, (2, 2), strides=(2, 2), padding="valid", pool_mode="avg"))
        self.assertAlmostEqual(means[0, 0, 0, 0], 1.0)
        self.assertAlmostEqual(means[0, 0, 0, 1], 0.0)
        self.assertAlmostEqual(means[0, 1, 0, 0], -1.5)
        self.assertAlmostEqual(means[0, 1, 0, 1], -1.0)

    def test_variance(self):
        data = np.array([[[[1, 2], [2, 3], [-1, -2]],
                          [[-1, 3], [2, -5], [0, 1]],
                          [[-2, 2], [0.5, -2], [2, -1]],
                          [[2, -4], [-0.5, -1], [3, 2]]]], dtype=np.float32)
        inp = Input(shape=(4, 3, 2))
        x = VarianceLayer((2, 2))(inp)
        model = Model(inputs=inp, outputs=x)
        keras_values = model.predict(data, batch_size=1)
        self.assertAlmostEqual(keras_values[0, 0, 0, 0], 1.5, places=4)
        self.assertAlmostEqual(keras_values[0, 0, 1, 0], 0.25, places=4)
        self.assertAlmostEqual(keras_values[0, 1, 0, 0], 2.125, places=4)
        self.assertAlmostEqual(keras_values[0, 1, 1, 0], 0.25, places=4)
        self.assertAlmostEqual(keras_values[0, 0, 0, 1], 11.1875, places=4)
        self.assertAlmostEqual(keras_values[0, 0, 1, 1], 2.25, places=4)
        self.assertAlmostEqual(keras_values[0, 1, 0, 1], 4.6875, places=4)
        self.assertAlmostEqual(keras_values[0, 1, 1, 1], 2.25, places=4)
