#!/usr/bin/python3
# -*- coding: utf-8 -*-
from keras.layers import Lambda
import keras.backend as K


def extract_edge_and_center_area(x):
    e0 = x[:, 0:16, 0:16]
    e1 = x[:, K.shape(x)[0]-16:K.shape(x)[0]+16, 0:16]
    e2 = x[:, -16:, 0:16]
    e7 = x[:, 0:16, K.shape(x)[1]-16:K.shape(x)[1]+16]
    cn = x[:, K.shape(x)[0]-16:K.shape(x)[0]+16, K.shape(x)[1]-16:K.shape(x)[1]+16]
    e3 = x[:, -16:, K.shape(x)[1]-16:K.shape(x)[1]+16]
    e6 = x[:, 0:16, -16:]
    e5 = x[:, K.shape(x)[0]-16:K.shape(x)[0]+16, -16:]
    e4 = x[:, -16:, -16:]
    l1 = K.concatenate([e0, e1, e2], axis=1)
    l2 = K.concatenate([e7, cn, e3], axis=1)
    l3 = K.concatenate([e6, e5, e4], axis=1)
    return K.concatenate([l1, l2, l3], axis=2)


def calculate_output_shape(input_shape):
    #assert input_shape[0] >= 64
    #assert input_shape[1] >= 64
    return 64, 64


def get_edge_and_center_extraction_layer():
    return Lambda(extract_edge_and_center_area, calculate_output_shape)
