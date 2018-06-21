#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals
from PyQt5.QtCore import Qt, QRect, pyqtSignal, QSize
from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout, QVBoxLayout, QSizePolicy
from PyQt5.QtWidgets import QPushButton, QLabel, QFileDialog, QSplitter, QCheckBox, QScrollArea
from PyQt5.QtGui import QImage, QPainter, QColor
from skimage.io import imread
from visualization_helpers import convert_image
import sys
import os
import re


class ImageWidget(QWidget):
    def __init__(self, img):
        super().__init__()
        self.img = img
        self.rect = QRect(0, 0, 128, 128)
        self.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))
        self.set_img(img)

    def set_img(self, img):
        self.img = img
        if self.img:
            self.setMaximumSize(self.img.size())
            self.setFixedSize(self.img.size())
            self.rect = QRect(0, 0, self.img.width(), self.img.height())
        else:
            self.rect = QRect(0, 0, 128, 128)
            self.setFixedSize(QSize(128, 128))
        self.updateGeometry()
        print("ImageWidget size: " + str(self.size()))
        print("ImageWidget pos: " + str(self.pos().x()) + ", " + str(self.pos().y()))

    def minimumSizeHint(self):
        if self.img:
            return QSize(self.img.width(), self.img.height())
        else:
            return QSize(20, 20)

    def sizeHint(self):
        return self.minimumSizeHint()

    def paintEvent(self, e):
        qp = QPainter()
        qp.begin(self)
        self.draw(qp)
        qp.end()

    def draw(self, qp):
        qp.setBrush(QColor(255, 0, 255))
        qp.setPen(QColor(255, 0, 0))
        qp.drawRect(self.rect)
        if self.img:
            qp.drawImage(0, 0, self.img)
        qp.drawLine(0, 0, 0, self.size().height())
        qp.setPen(QColor(0, 0, 255))
        qp.drawLine(0, 0, self.size().width(), self.size().height())


class ThumbnailList(QWidget):
    imgSelected = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.images = []
        self.selected = 0
        size_policy = QSizePolicy()
        size_policy.setVerticalPolicy(QSizePolicy.MinimumExpanding)
        self.setSizePolicy(size_policy)
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(4, 4, 4, 0)
        size_row = QHBoxLayout()
        slider_label = QLabel()
        slider_label.setText("Thumbnailgröße:")
        slider_label.setMinimumHeight(12)
        size_row.addWidget(slider_label, alignment=Qt.AlignLeading)
        self.layout.addLayout(size_row)
        self.t_list = QVBoxLayout()
        self.lines = []
        for _ in range(8):
            self.add_empty_line()
        self.layout.addLayout(self.t_list, stretch=1)
        self.setLayout(self.layout)

    def load_images(self, path):
        self.images = []
        self.selected = 0
        filename_regex = re.compile(r".*\.(jpg|JPG|jpeg|JPEG|png|PNG|bmp|BMP)$")
        for filename in os.listdir(path):
            if filename_regex.match(filename):
                np_img = imread(os.path.join(path, filename))
                qimage = QImage(convert_image(np_img), np_img.shape[1], np_img.shape[0], QImage.Format_RGB32)
                self.images.append({
                    'thumb': qimage.scaledToWidth(128),
                    'np_img': np_img,
                    'qimage': qimage,
                    'filename': os.path.join(path, filename)})
        self.update_list()
        if len(self.images) > 0:
            self.imgSelected.emit(self.images[0])

    def update_list(self):
        while len(self.lines) < len(self.images):
            self.add_empty_line()
        for i, img_data in enumerate(self.images):
            self.lines[i]['checkbox'].setVisible(True)
            self.lines[i]['image_widget'].set_img(img_data['thumb'])
            self.lines[i]['image_widget'].setMinimumSize(img_data['thumb'].size())
            self.lines[i]['image_widget'].setFixedSize(img_data['thumb'].size())
            self.lines[i]['image_widget'].setVisible(True)
            self.lines[i]['image_widget'].updateGeometry()
        for i in range(len(self.lines), len(self.images)):
            self.lines[i]['checkbox'].setVisible(False)
            self.lines[i]['image_widget'].setVisible(False)
        self.t_list.update()
        self.updateGeometry()

    def add_empty_line(self):
        img_line = QHBoxLayout()
        checkbox = QCheckBox()
        img_line.addWidget(checkbox)
        img_wid = ImageWidget(None)
        img_wid.setFixedSize(QSize(128, 128))
        img_line.addWidget(img_wid)
        self.lines.append({'checkbox': checkbox, 'image_widget': img_wid})
        self.t_list.addLayout(img_line)


class PreviewArea(QWidget):
    def __init__(self):
        super().__init__()
        size_policy = QSizePolicy()
        size_policy.setHorizontalPolicy(QSizePolicy.Expanding)
        size_policy.setVerticalPolicy(QSizePolicy.Expanding)
        self.setSizePolicy(size_policy)
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 4, 0, 0)
        this_row = QHBoxLayout()
        this_row.addSpacing(4)
        selection_label = QLabel()
        selection_label.setText("Dieses Bild")
        this_row.addWidget(selection_label)
        layout.addLayout(this_row)
        img_scroll_area = QScrollArea()
        img_scroll_area.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding))
        self.img_widget = ImageWidget(None)
        img_scroll_area.setWidget(self.img_widget)
        layout.addWidget(img_scroll_area, stretch=1)
        layout.addStretch()
        self.setLayout(layout)

    def set_image(self, img_d):
        self.img_widget.set_img(img_d['qimage'])
        self.update()


class InferenceInterface(QWidget):
    def __init__(self):
        super().__init__(flags=Qt.WindowTitleHint | Qt.WindowCloseButtonHint |
                               Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint)
        self.path = None
        self.setGeometry(200, 100, 1280, 720)
        self.setWindowTitle("Unsharp Detector")
        main_layout = QVBoxLayout()
        path_row = QHBoxLayout()
        open_button = QPushButton()
        open_button.setText("Pfad auswählen")
        open_button.clicked.connect(self.open_path_select_dialog)
        path_row.addWidget(open_button, alignment=Qt.AlignLeading)
        self.path_label = QLabel()
        path_row.addWidget(self.path_label, alignment=Qt.AlignLeading)
        path_row.addStretch()
        main_layout.addLayout(path_row, stretch=0)
        image_splitter = QSplitter()
        image_splitter.setOrientation(Qt.Horizontal)
        self.thumbnail_list = ThumbnailList()
        self.thumbnail_list.imgSelected.connect(self.img_selected)
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.thumbnail_list)
        scroll_area.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding))
        image_splitter.addWidget(scroll_area)
        self.preview_area = PreviewArea()
        image_splitter.addWidget(self.preview_area)
        image_splitter.setSizes([176, self.width()-176])
        image_splitter.setSizePolicy(QSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding))
        main_layout.addWidget(image_splitter)
        self.setLayout(main_layout)
        self.show()
        self.thumbnail_list.load_images("./validation_data/good/")

    def open_path_select_dialog(self):
        dialog = QFileDialog()
        dialog.setWindowTitle("Pfad der Bilder auswählen")
        dialog.setModal(False)
        dialog.setFileMode(QFileDialog.Directory)
        if dialog.exec():
            self.path = dialog.selectedFiles()[0]
            self.thumbnail_list.load_images(self.path)
            self.path_label.setText("Path: " + self.path)
        else:
            self.path = None
            self.path_label.setText("Kein Pfad ausgewählt.")

    def img_selected(self, img_d):
        self.preview_area.set_image(img_d)


if __name__ == "__main__":
    app_object = QApplication(sys.argv)
    window = InferenceInterface()
    status = app_object.exec_()
