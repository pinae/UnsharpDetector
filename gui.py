#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals
import sys
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtGui import QImage, QPainter, QPixmap, QFont, QColor
from PyQt5.QtCore import QRect, Qt
from threading import Thread
from queue import Queue
import numpy as np


class TrainingPreview(QWidget):
    def __init__(self, feedback_queue):
        super().__init__()
        self.feedback_queue = feedback_queue
        self.setWindowTitle("Training preview")
        self.resize(4 * 256, 3 * 276)
        self.setMinimumWidth(256)
        self.pixmaps = [QPixmap(QImage(
            self.convert_image(np.zeros((256, 256, 3), dtype=np.float32)),
            256, 256, QImage.Format_RGB32))]
        self.labels = [{"color": QColor(0, 255, 0)}]
        self.predictions = [{"color": QColor(128, 128, 0)}]
        self.white = QColor(255, 255, 255)
        self.font = QFont('Sans-Serif', 12, QFont.Normal)
        self.show()

    def paintEvent(self, e):
        qp = QPainter()
        qp.begin(self)
        qp.setRenderHint(QPainter.Antialiasing)
        qp.setRenderHint(QPainter.HighQualityAntialiasing)
        self.draw(qp)
        qp.end()

    def draw(self, qp):
        size = self.size()
        line_len = size.width()//256
        qp.setFont(self.font)
        qp.setPen(self.white)
        for i, pixmap in enumerate(self.pixmaps):
            qp.drawPixmap(QRect((i % line_len) * 256, (i // line_len) * 276, 256, 256),
                          pixmap, QRect(0, 0, 256, 256))
            qp.setBrush(self.labels[i]["color"])
            qp.drawRect((i % line_len) * 256, (i // line_len) * 276 + 256, 128, 20)
            qp.setBrush(self.predictions[i]["color"])
            qp.drawRect((i % line_len) * 256 + 128, (i // line_len) * 276 + 256, 128, 20)

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
        self.setWindowTitle("Training preview | Epoch: " + str(epoch))
        from skimage.io import imsave
        imsave("test_data.png", np.clip(np.concatenate(images, axis=0), 0, 1))
        self.pixmaps = []
        self.labels = []
        self.predictions = []
        for i, img in enumerate(images):
            qimage = QImage(self.convert_image(img * 255), img.shape[0], img.shape[1], QImage.Format_RGB32)
            self.pixmaps.append(QPixmap().fromImage(qimage, flags=(Qt.AutoColor | Qt.DiffuseDither)).copy())
            self.labels.append({
                "color": QColor(int(labels[i][0] * 255), int(labels[i][1] * 255), 0)
            })
            self.predictions.append({
                "color": QColor(int(predictions[i][0] * 255), int(predictions[i][1] * 255), 0)
            })
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


def run_gui(feedback_queue):
    app_object = QApplication(sys.argv)
    window = TrainingPreview(feedback_queue)
    feedback_queue.put({"callback": window.show_data})
    status = app_object.exec_()
    feedback_queue.put({"stop": status})


def init_gui():
    feedback_queue = Queue()
    gui_thread = Thread(target=run_gui, args=(feedback_queue,))
    gui_thread.start()
    initialization_answer = feedback_queue.get(True)
    if "callback" in initialization_answer:
        return initialization_answer["callback"], feedback_queue, gui_thread
    else:
        print("ERROR: No Callback in init answer!")
    return None, feedback_queue, gui_thread


if __name__ == "__main__":
    clb, fq, thread = init_gui()
    from TrainingDataGenerator import UnsharpTrainingDataGenerator
    g = UnsharpTrainingDataGenerator(["../../Bilder/Bilder der Woche/"], batch_size=7)
    g.on_epoch_end()
    x, y = g.__getitem__(0)
    print("x.shape: " + str(x.shape))
    print("y.shape: " + str(y.shape))
    clb(x, y, np.array([[0.2, 0.8], [0.9, 0.1],
                        [0.3, 0.7], [0.3, 0.7],
                        [0.3, 0.7], [0.3, 0.7],
                        [0.3, 0.7]], dtype=np.float32), 0)
    feedback = fq.get()
    if "stop" in feedback.keys():
        print("stopping")
        thread.join()
        print("join finished")
        sys.exit(feedback["stop"])
