#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals
from keras import backend as K
from keras.engine.topology import Layer
from keras.models import Model
from keras.layers import Input
import numpy as np
import unittest


class GlobalVarianceLayer(Layer):
    def __init__(self, **kwargs):
        super(GlobalVarianceLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(GlobalVarianceLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x, **kwargs):
        mean = K.mean(K.mean(x, axis=2), axis=1)
        mean_vector = K.repeat_elements(K.expand_dims(mean, axis=1), x.get_shape()[1], axis=1)
        mean_matrix = K.repeat_elements(K.expand_dims(mean_vector, axis=2), x.get_shape()[2], axis=2)
        quad_diff = (x - mean_matrix) ** 2
        return K.mean(K.mean(quad_diff, axis=2), axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[3]


class TestGlobalVarianceLayer(unittest.TestCase):
    def test_2d_mean(self):
        data = np.array([[[[1, 0], [2, 1], [3, -1]],
                          [[0, 1], [1, -2], [2, 1]],
                          [[-2, -1], [-1, -1], [3, 2]]]], dtype=np.float32)
        x = K.variable(data, dtype=K.floatx())
        mean = K.eval(K.mean(K.mean(x, axis=2), axis=1))
        self.assertAlmostEqual(mean[0, 0], 1.0)
        self.assertAlmostEqual(mean[0, 1], 0.0)

    def test_variance(self):
        data = np.array([[[[1, 2], [2, 3], [-1, -2]],
                          [[-1, 3], [2, -5], [0, 1]],
                          [[-2, 7], [0.5, -2], [2, -1]]]], dtype=np.float32)
        inp = Input(shape=(3, 3, 2))
        x = GlobalVarianceLayer()(inp)
        model = Model(inputs=inp, outputs=x)
        keras_values = model.predict(data, batch_size=1)
        self.assertAlmostEqual(keras_values[0, 0],
                               np.array([[[1, 2, -1],
                                          [-1, 2, 0],
                                          [-2, 0.5, 2]]], dtype=np.float32).var(), places=4)
        self.assertAlmostEqual(keras_values[0, 1],
                               np.array([[[2, 3, -2],
                                          [3, -5, 1],
                                          [7, -2, -1]]], dtype=np.float32).var(), places=4)
