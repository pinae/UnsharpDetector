#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals
import sys
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtGui import QImage, QPainter, QPixmap, QFont, QColor
from PyQt5.QtCore import QRect
from threading import Thread
from queue import Queue
import numpy as np


class TrainingPreview(QWidget):
    def __init__(self, show_queue, feedback_queue):
        super().__init__()
        self.show_queue = show_queue
        self.feedback_queue = feedback_queue
        self.setWindowTitle("Training preview")
        self.pixmap = QPixmap(QImage(
            self.convert_image(np.zeros((256, 256, 3), dtype=np.float32)),
            256, 256, QImage.Format_RGB32))
        #self.pixmap = QPixmap(QImage("../../Bilder/Bilder der Woche/04_Farbspektakel-648c3ec3dfbb6952.jpg"))
        self.show()

    #def __del__(self):
    #    if self.train_process.is_alive():
    #        self.train_process.join()

    def paintEvent(self, e):
        qp = QPainter()
        qp.begin(self)
        qp.setRenderHint(QPainter.Antialiasing)
        qp.setRenderHint(QPainter.HighQualityAntialiasing)
        self.draw(qp)
        qp.end()

    def draw(self, qp):
        size = self.size()
        w = size.width()
        h = size.height()
        qp.drawPixmap(QRect(0, 0, 256, 256), self.pixmap, QRect(0, 0, 256, 256))
        font = QFont('Sans-Serif', 12, QFont.Normal)
        qp.setFont(font)
        qp.setPen(QColor(128, 128, 128))
        qp.setBrush(QColor(255, 255, 255))
        """qp.drawRoundedRect(0, 0, w, h, 5, 5)
        qp.setPen(QColor(0, 0, 0))
        font_metrics = qp.fontMetrics()
        c_start_position = 5
        cursor_pixel_position = c_start_position
        self.character_offsets = [cursor_pixel_position]
        for i, c in enumerate(self.text):
            start_of_parsed_block = False
            end_of_parsed_block = False
            inside_parsed_block = False
            for start, end in self.parsed_blocks:
                if start == i:
                    block_width = 4
                    for char in self.text[start:end]:
                        block_width += font_metrics.width(char["char"])
                    qp.setPen(QColor(0, 0, 0))
                    qp.setBrush(QColor(0, 0, 0))
                    qp.drawRoundedRect(c_start_position+2, 4, block_width, 20, 2, 2)"""

    def show_data(self, images, labels, predictions, epoch):
        img = images[0].astype(np.int32)
        print(predictions)
        qimage = QImage(img, img.shape[0], img.shape[1], QImage.Format_RGB32)
        self.pixmap = QPixmap(qimage)
        self.update()

    @staticmethod
    def convert_image(numpy_array):
        return np.left_shift(
            np.left_shift(
                np.left_shift(
                    np.zeros((numpy_array.shape[0], numpy_array.shape[1]), dtype=np.uint32) + 0xff,
                    8) + numpy_array[:, :, 0].astype(np.uint32),
                8) + numpy_array[:, :, 1].astype(np.uint32),
            8) + numpy_array[:, :, 2].astype(np.uint32)


def start_gui(show_queue, feedback_queue):
    app_object = QApplication(sys.argv)
    window = TrainingPreview(show_queue, feedback_queue)
    status = app_object.exec_()
    feedback_queue.put({"stop": status})
    running = False
    while running:
        #if not show_queue.empty():
        #    batch = show_queue.get()
        #    window.show_data(batch['x'], batch['y'], batch['prediction'], batch['epoch'])
        if not window:
            print("stop command")
            feedback_queue.put("stop")
            running = False
        #if not feedback_queue.empty():
        #    feedback = feedback_queue.get_nowait()
        #    if feedback == "stop":
        #        running = False


def init_gui():
    show_queue = Queue()
    feedback_queue = Queue()
    gui_thread = Thread(target=start_gui, args=(show_queue, feedback_queue))
    gui_thread.start()
    return show_queue, feedback_queue, gui_thread


if __name__ == "__main__":
    sq, fq, thread = init_gui()
    from TrainingDataGenerator import UnsharpTrainingDataGenerator
    g = UnsharpTrainingDataGenerator(["../../Bilder/Bilder der Woche/"], batch_size=2)
    g.on_epoch_end()
    x, y = g.__getitem__(0)
    sq.put({'x': x, 'y': y,
            'prediction': np.array([[0.2, 0.8], [0.9, 0.1]], dtype=np.float32),
            'epoch': 0})
    feedback = fq.get()
    if "stop" in feedback.keys():
        print("stopping")
        thread.join()
        print("join finished")
        sys.exit(feedback["stop"])
