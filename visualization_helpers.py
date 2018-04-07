#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np


def generate_y_image(batch_y, dtype=np.float):
    batch_size = batch_y.shape[0]
    batch_y_img_line = np.repeat(batch_y.astype(dtype).reshape(1, batch_size, 2), 256, axis=1)
    return np.repeat(
        np.concatenate([batch_y_img_line,
                        np.zeros((1, 256 * batch_size, 1), dtype=dtype)], axis=2),
        20, axis=0)
