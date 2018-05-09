#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals
import sys
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtGui import QImage, QPainter, QPixmap
from multiprocessing import Process, Queue
import numpy as np


class TrainingPreview(QWidget):
    def __init__(self, queue):
        super().__init__()
        self.setWindowTitle("Training preview")
        self.pixmap = QPixmap(QImage(np.zeros((256, 256, 3), dtype=np.int32), 256, 256, QImage.Format_RGB32))
        self.show()
        while True:
            try:
                batch = queue.get()
                self.show_data(batch['x'], batch['y'], batch['prediction'], batch['epoch'])
            except Queue.Empty:
                pass

    def __del__(self):
        if self.train_process.is_alive():
            self.train_process.join()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawPixmap(0, 0, 256, 256, self.pixmap)

    def show_data(self, images, labels, predictions, epoch):
        img = images[0].astype(np.int32)
        print(predictions)
        qimage = QImage(img, img.shape[0], img.shape[1], QImage.Format_RGB32)
        self.pixmap = QPixmap(qimage)
        self.update()


def start_gui(queue):
    app_object = QApplication(sys.argv)
    window = TrainingPreview(queue)
    sys.exit(app_object.exec_())


def init_gui():
    show_queue = Queue()
    gui_process = Process(target=start_gui, args=(show_queue,))
    gui_process.start()
    return show_queue, gui_process
