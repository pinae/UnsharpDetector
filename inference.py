#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals
from os import path, listdir
from skimage.io import imread
from model import create_model
import numpy as np
import json
import re


def load_model(input_size):
    with open('unsharpDetectorSettings.json', 'r') as json_file:
        settings = json.load(json_file)
        model = create_model(input_size,
                             settings["l1fc"], settings["l1fs"],
                             settings["l2fc"], settings["l2fs"],
                             settings["l3fc"], settings["l3fs"],
                             settings["eac_size"])
        model.load_weights("unsharpDetectorWeights.hdf5")
    return model


def inference(model, img_list):
    return model.predict(img_list, batch_size=len(img_list))


if __name__ == "__main__":
    filename_regex = re.compile(r".*\.(jpg|JPG|jpeg|JPEG|png|PNG|bmp|BMP)$")
    img_path = "validation_data/good/"
    filenames = listdir(path.abspath(img_path))
    for filename in filenames:
        if filename_regex.match(filename):
            print("reading " + str(path.join(path.abspath(img_path), filename)))
            data = np.array([
                imread(path.join(path.abspath(img_path), filename)) / 255
            ])
            trained_model = load_model(data.shape[1:])
            print(inference(trained_model, data))
