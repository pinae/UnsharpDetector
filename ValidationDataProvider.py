# -*- coding: utf-8 -*-
from tensorflow.keras.utils import Sequence
from os import path, listdir
from skimage.io import imread
from skimage.transform import resize
import numpy as np
import re


class NoUsableData(Exception):
    pass


class UnsharpValidationDataProvider(Sequence):
    def __init__(self, image_folder="", batch_size=10, target_size=(256, 256)):
        self.batch_size = batch_size
        self.target_size = target_size
        filename_regex = re.compile(r".*\.(jpg|JPG|jpeg|JPEG|png|PNG|bmp|BMP)$")
        self.data = []
        good_filenames = listdir(path.join(path.abspath(image_folder), "good"))
        bad_filenames = listdir(path.join(path.abspath(image_folder), "bad"))
        for filename in good_filenames:
            if filename_regex.match(filename):
                self.data.append({"filename": path.join(path.abspath(image_folder), "good", filename), "label": 1})
        for filename in bad_filenames:
            if filename_regex.match(filename):
                self.data.append({"filename": path.join(path.abspath(image_folder), "bad", filename), "label": 0})
        if len(self.data) < 1:
            raise NoUsableData
        self.indexes = np.arange(len(self.data))

    def __len__(self):
        return int(np.floor(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        filename_selection = [self.data[k] for k in indexes]
        batch_x, batch_y = self.__data_generation(filename_selection)
        return batch_x, batch_y

    def __data_generation(self, selection):
        batch_x = []
        batch_y = []
        for d in selection:
            img = imread(d["filename"])
            if len(img.shape) != 3:
                raise NoUsableData
            img = resize(img, (max(self.target_size[0],
                                   int(np.floor(img.shape[0]*self.target_size[1]/img.shape[1]))),
                               max(self.target_size[1],
                                   int(np.floor(img.shape[1]*self.target_size[0]/img.shape[0]))),
                               img.shape[2]), mode='reflect')
            crop_start_y = int(np.floor((img.shape[0] - self.target_size[0]) / 2))
            crop_start_x = int(np.floor((img.shape[1] - self.target_size[1]) / 2))
            img = img[crop_start_y:crop_start_y + self.target_size[0],
                      crop_start_x:crop_start_x + self.target_size[1], :].astype(np.float32)
            batch_x.append(img)
            if d["label"] == 1:
                batch_y.append(np.array([0, 1], dtype=np.float32))
            else:
                batch_y.append(np.array([1, 0], dtype=np.float32))
        return np.array(batch_x), np.array(batch_y)

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.data))
        np.random.shuffle(self.indexes)


if __name__ == "__main__":
    generator = UnsharpValidationDataProvider("validation_data", batch_size=2)
    generator.on_epoch_end()
    bat_x, bat_y = generator.__getitem__(0)
    print(bat_y)
