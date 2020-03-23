# -*- coding: utf-8 -*-
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, LeakyReLU, Dense, GlobalMaxPool2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Input, Concatenate, MaxPool2D, AveragePooling2D, Flatten, Add
from GlobalVarianceLayer import GlobalVarianceLayer
from VarianceLayer import VarianceLayer
from EdgeAndCenterExtractionLayer import EdgeAndCenterExtractionLayer
import tensorflow as tf
import numpy as np


def laplacian_group_initializer(shape, dtype=None):
    kernel = np.zeros(shape, dtype=np.float)  # np.zeros(shape, dtype=dtype)
    if np.random.random() < 0.5 and kernel.shape[0] >= 3 and len(kernel.shape) == 2:
        kernel[int(kernel.shape[0] // 2) - 1, int(kernel.shape[1] // 2)] = 1
        kernel[int(kernel.shape[0] // 2) + 1, int(kernel.shape[1] // 2)] = 1
    if np.random.random() < 0.5 and kernel.shape[1] >= 3 and len(kernel.shape) == 2:
        kernel[int(kernel.shape[0] // 2), int(kernel.shape[1] // 2) - 1] = 1
        kernel[int(kernel.shape[0] // 2), int(kernel.shape[1] // 2) + 1] = 1
    kernel[tuple(map(lambda x: int(np.floor(x / 2)), kernel.shape))] = -np.sum(kernel)
    return kernel + np.random.normal(0.0, 0.005, shape) * 1.0


def create_model(input_shape, l1fc, l1fs, l1st, l2fc, l2fs, l2st, l3fc, l3fs, eac_size, res_c, res_fc, res_fs):
    inp = Input(shape=(input_shape[0], input_shape[1], 3))
    c1 = Conv2D(l1fc, kernel_size=l1fs, strides=l1st, use_bias=True, padding="same",
                data_format="channels_last", kernel_initializer=laplacian_group_initializer)(inp)
    l1 = LeakyReLU(alpha=0.2)(c1)
    eac1_obj = EdgeAndCenterExtractionLayer(width=eac_size)
    eac1 = eac1_obj(l1)
    eac1.set_shape(eac1_obj.compute_output_shape(l1.shape))
    c2 = Conv2D(l2fc, kernel_size=l2fs, strides=l2st, use_bias=True, padding="same",
                data_format="channels_last")(l1)
    l2 = LeakyReLU(alpha=0.2)(c2)
    eac2_obj = EdgeAndCenterExtractionLayer(width=eac_size)
    eac2 = eac2_obj(l2)
    eac2.set_shape(eac2_obj.compute_output_shape(l2.shape))
    c3 = Conv2D(l3fc, kernel_size=l3fs, strides=1, use_bias=True, padding="same",
                data_format="channels_last")(l2)
    last_layer = c3
    prev_layer = None
    for i in range(res_c):
        res_act = LeakyReLU(alpha=0.2)(last_layer)
        if prev_layer is not None:
            res_act = Add()([res_act, prev_layer])
        prev_layer = last_layer
        last_layer = Conv2D(res_fc, kernel_size=res_fs, strides=1, use_bias=True, padding="same",
                            data_format="channels_last")(res_act)
    eac3_obj = EdgeAndCenterExtractionLayer(width=eac_size)
    eac3 = eac3_obj(c3)
    eac3.set_shape(eac3_obj.compute_output_shape(c3.shape))
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
    if res_c > 0:
        res_eac = EdgeAndCenterExtractionLayer(width=eac_size)(last_layer)
        features.append(GlobalVarianceLayer()(last_layer))
        features.append(GlobalMaxPool2D()(last_layer))
        features.append(GlobalAveragePooling2D()(last_layer))
        features.append(GlobalVarianceLayer()(res_eac))
        features.append(GlobalMaxPool2D()(res_eac))
        features.append(GlobalAveragePooling2D()(res_eac))
        features.append(Flatten()(VarianceLayer((eac_size, eac_size))(res_eac)))
        res_eac_max_grid = MaxPool2D((eac_size, eac_size), strides=eac_size,
                                     padding="valid", data_format="channels_last")(res_eac)
        res_eac_avg_grid = AveragePooling2D((eac_size, eac_size), strides=eac_size,
                                            padding="valid", data_format="channels_last")(res_eac)
        features.append(GlobalVarianceLayer()(res_eac_max_grid))
        features.append(GlobalVarianceLayer()(res_eac_avg_grid))
        features.append(Flatten()(res_eac_max_grid))
        features.append(Flatten()(res_eac_avg_grid))
    feature_vector = Concatenate()(features)
    o = Dense(2, activation="softmax", use_bias=True, name="output")(feature_vector)
    return Model(inputs=inp, outputs=o)
