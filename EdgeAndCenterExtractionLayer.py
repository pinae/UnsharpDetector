#!/usr/bin/python3
# -*- coding: utf-8 -*-
from keras.layers import Lambda
import keras.backend as K


def extract_edge_and_center_area(x):
    e0 = x[:, 0:64, 0:64]
    e1 = x[:, K.shape(x)[0]-64:K.shape(x)[0]+64, 0:64]
    e2 = x[:, -64:, 0:64]
    e7 = x[:, 0:64, K.shape(x)[1]-64:K.shape(x)[1]+64]
    cn = x[:, K.shape(x)[0]-64:K.shape(x)[0]+64, K.shape(x)[1]-64:K.shape(x)[1]+64]
    e3 = x[:, -64:, K.shape(x)[1]-64:K.shape(x)[1]+64]
    e6 = x[:, 0:64, -64:]
    e5 = x[:, K.shape(x)[0]-64:K.shape(x)[0]+64, -64:]
    e4 = x[:, -64:, -64:]
    return K.concatenate([
        K.concatenate([e0, e1, e2], axis=0),
        K.concatenate([e7, cn, e3], axis=0),
        K.concatenate([e6, e5, e4], axis=0)], axis=1)


def calculate_output_shape(input_shape):
    assert input_shape[0] >= 256
    assert input_shape[1] >= 256
    return 256, 256


def get_edge_and_center_extraction_layer():
    return Lambda(extract_edge_and_center_area, calculate_output_shape)
