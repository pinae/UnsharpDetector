#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import division, print_function
from os import path, listdir
from keras.utils import Sequence
from random import random, choice, randrange
from skimage.io import imread, imshow, use_plugin, show
from skimage.transform import resize, rotate
from skimage.filters import gaussian
from scipy.ndimage.filters import convolve
from visualization_helpers import generate_y_image
import numpy as np
import re


class NoUsableTrainingData(Exception):
    pass


class UnsharpTrainingDataGenerator(Sequence):
    def __init__(self, image_folders=[], batch_size=10, target_size=(256, 256)):
        self.batch_size = batch_size
        self.target_size = target_size
        filename_regex = re.compile(r".*\.(jpg|JPG|jpeg|JPEG|png|PNG|bmp|BMP)$")
        self.image_filenames = []
        for folder in image_folders:
            filenames = listdir(path.abspath(folder))
            for filename in filenames:
                if filename_regex.match(filename):
                    self.image_filenames.append(path.join(path.abspath(folder), filename))
        if len(self.image_filenames) < 1:
            raise NoUsableTrainingData

    def __len__(self):
        return self.batch_size

    def __getitem__(self, idx):
        batch_x = []
        batch_y = []
        for i in range(len(self)):
            img = imread(choice(self.image_filenames))
            min_scale_factor = max(self.target_size[0]/img.shape[0], self.target_size[1]/img.shape[1])
            sf = random()*(1-min_scale_factor)+min_scale_factor
            img = resize(img, (int(img.shape[0]*sf), int(img.shape[1]*sf), img.shape[2]), mode='reflect')
            crop_start_x = randrange(0, img.shape[1]-self.target_size[1])
            crop_start_y = randrange(0, img.shape[0]-self.target_size[0])
            img = img[crop_start_y:crop_start_y+self.target_size[0], crop_start_x:crop_start_x+self.target_size[1], :]
            print(img.shape)
            if random() < 0.5:
                batch_x.append(img)
                batch_y.append(np.array([0, 1]))
            else:
                batch_x.append(self.blur_image(img))
                batch_y.append(np.array([1, 0]))
        return np.array(batch_x), np.array(batch_y)

    def blur_image(self, img):
        mode = choice([["blur"], ["shake"], ["blur", "shake"]])
        blurred_img = img
        if "blur" in mode:
            blurred_img = gaussian(img, sigma=0.5+4.5*random())
        if "shake" in mode:
            blurred_img = self.add_shake(blurred_img)
        if random() < 0.3:
            blurred_img = self.add_mask(blurred_img, img)
        if random() < 0.4:
            blurred_img = self.add_noise(blurred_img)
        return blurred_img

    @staticmethod
    def add_shake(img):
        filter_matrix = np.zeros((9, 9), dtype=img.dtype)
        shake_len = random()*6.5+2.5
        filter_matrix[4, 4] = 1.0
        for i in range(1, 5):
            x = (shake_len - i * 2 + 1) / 2
            filter_matrix[4+i, 4] = x
            filter_matrix[4-i, 4] = x
        filter_matrix = np.clip(filter_matrix, 0, 1)
        filter_matrix = np.repeat(
            filter_matrix.reshape(filter_matrix.shape[0], filter_matrix.shape[1], 1),
            3, axis=2)
        filter_matrix = rotate(filter_matrix, random() * 360, mode='constant', cval=0.0)
        filter_matrix = filter_matrix / filter_matrix.sum()
        img = convolve(img, filter_matrix, mode='reflect')
        return img

    @staticmethod
    def add_mask(blurred_img, clear_img):
        mask = np.array([[0, 0, 0, 1, 1, 1, 0, 0, 0],
                         [0, 0, 1, 1, 1, 1, 1, 0, 0],
                         [0, 0, 1, 1, 1, 1, 1, 0, 0],
                         [0, 1, 1, 1, 1, 1, 1, 1, 0],
                         [0, 1, 1, 1, 1, 1, 1, 1, 0],
                         [0, 1, 1, 1, 1, 1, 1, 1, 0],
                         [0, 1, 1, 1, 1, 1, 1, 1, 0],
                         [0, 0, 1, 1, 1, 1, 1, 0, 0],
                         [0, 0, 1, 1, 1, 1, 1, 0, 0],
                         [0, 0, 0, 1, 1, 1, 0, 0, 0]], dtype=blurred_img.dtype)
        mask = np.clip(mask + np.random.random(mask.shape)*0.5*(0.3+random()), 0, 1)
        mask = np.repeat(mask.reshape(mask.shape[0], mask.shape[1], 1), 3, axis=2)
        mask = resize(mask, (blurred_img.shape[0], blurred_img.shape[1], blurred_img.shape[2]), mode='reflect')
        return mask * blurred_img + (1 - mask) * clear_img

    @staticmethod
    def add_noise(img):
        noise = np.random.randn(*img.shape)*(0.05+0.1*random())
        noise = gaussian(noise, sigma=0.1+1.1*random())
        return np.clip(img+noise, 0, 1)


if __name__ == "__main__":
    generator = UnsharpTrainingDataGenerator(["../../Bilder/Backgrounds/"], batch_size=7)
    batch_x, batch_y = generator.__getitem__(0)
    print(batch_y)
    use_plugin("matplotlib")
    imshow(np.concatenate([np.concatenate(batch_x, axis=1), generate_y_image(batch_y, dtype=batch_x.dtype)], axis=0))
    show()
