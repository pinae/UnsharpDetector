#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np


def convert_image(numpy_array):
    return np.left_shift(
        np.left_shift(
            np.left_shift(
                np.zeros((numpy_array.shape[0], numpy_array.shape[1]), dtype=np.uint32) + 0xff,
                8) + numpy_array[:, :, 0].astype(np.uint32),
            8) + numpy_array[:, :, 1].astype(np.uint32),
        8) + numpy_array[:, :, 2].astype(np.uint32)


def generate_y_image(batch_y, dtype=np.float):
    batch_size = batch_y.shape[0]
    batch_y_img_line = np.repeat(batch_y.astype(dtype).reshape(1, batch_size, 2), 256, axis=1)
    return np.repeat(
        np.concatenate([batch_y_img_line,
                        np.zeros((1, 256 * batch_size, 1), dtype=dtype)], axis=2),
        20, axis=0)
