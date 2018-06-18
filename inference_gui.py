#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals
from PyQt5.QtCore import Qt, QRect, pyqtSignal, QSize
from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout, QVBoxLayout, QSizePolicy
from PyQt5.QtWidgets import QPushButton, QLabel, QFileDialog, QSplitter, QWidgetItem, QCheckBox, QScrollArea
from PyQt5.QtGui import QImage, QPainter, QPixmap, QColor
from skimage.io import imread
from visualization_helpers import convert_image
import sys
import os
import re


class ImageWidget(QWidget):
    def __init__(self, img_pixmap):
        super().__init__()
        self.img = img_pixmap
        if self.img:
            print(self.img.size())
            self.setMinimumWidth(self.img.width())
            self.setMinimumHeight(self.img.height())
            self.rect = QRect(0, 0, self.img.width(), self.img.height())
            print(self.rect.size())
        else:
            self.rect = QRect(0, 0, 20, 20)
        print(self.size())

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
        #qp.setRenderHint(QPainter.Antialiasing)
        #qp.setRenderHint(QPainter.HighQualityAntialiasing)
        self.draw(qp)
        qp.end()

    def draw(self, qp):
        #size = self.size()
        #w = size.width()
        #h = size.height()
        qp.setBrush(QColor(255, 0, 255))
        qp.setPen(QColor(255, 128, 0))
        qp.drawRect(self.rect)
        if self.img:
            qp.drawPixmap(self.rect,
                          self.img, self.rect)


class ThumbnailList(QWidget):
    imgSelected = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.images = []
        self.selected = 0
        size_policy = QSizePolicy()
        size_policy.setVerticalPolicy(QSizePolicy.MinimumExpanding)
        self.setSizePolicy(size_policy)
        layout = QVBoxLayout()
        size_row = QHBoxLayout()
        slider_label = QLabel()
        slider_label.setText("Thumbnailgröße:")
        slider_label.setMinimumHeight(12)
        size_row.addWidget(slider_label, alignment=Qt.AlignLeading)
        layout.addLayout(size_row)
        self.t_list = QVBoxLayout()
        layout.addLayout(self.t_list)
        layout.addStretch()
        self.setLayout(layout)

    def load_images(self, path):
        self.images = []
        self.selected = 0
        filename_regex = re.compile(r".*\.(jpg|JPG|jpeg|JPEG|png|PNG|bmp|BMP)$")
        for filename in os.listdir(path):
            if filename_regex.match(filename):
                np_img = imread(os.path.join(path, filename))
                qimage = QImage(convert_image(np_img), np_img.shape[0], np_img.shape[1], QImage.Format_RGB32)
                self.images.append({
                    'thumb': qimage.scaledToWidth(128),
                    'np_img': np_img,
                    'qimage': qimage,
                    'filename': os.path.join(path, filename)})
        self.update_list()
        if len(self.images) > 0:
            self.imgSelected.emit(self.images[0])

    def update_list(self):
        while self.t_list.count() > 0:
            item = self.t_list.takeAt(self.t_list.count()-1)
            if type(item) == QWidgetItem:
                item.widget().deleteLater()
            else:
                item.deleteLater()
        for img_data in self.images:
            img_line = QHBoxLayout()
            checkbox = QCheckBox()
            checkbox.setText(".")
            img_line.addWidget(checkbox, alignment=Qt.AlignLeading)
            image = ImageWidget(QPixmap.fromImage(img_data['thumb'], flags=(Qt.AutoColor | Qt.DiffuseDither)))
            img_line.addWidget(image, alignment=Qt.AlignLeading)
            self.t_list.addLayout(img_line)
        self.update()


class PreviewArea(QWidget):
    def __init__(self):
        super().__init__()
        size_policy = QSizePolicy()
        size_policy.setHorizontalPolicy(QSizePolicy.Expanding)
        size_policy.setVerticalPolicy(QSizePolicy.Expanding)
        self.setSizePolicy(size_policy)
        layout = QVBoxLayout()
        selection_label = QLabel()
        selection_label.setText("Dieses Bild")
        layout.addWidget(selection_label, alignment=Qt.AlignLeading)
        self.img_widget = ImageWidget(None)
        layout.addWidget(self.img_widget, alignment=Qt.AlignLeading)
        layout.addStretch()
        self.setLayout(layout)

    def set_image(self, img_d):
        self.img_widget = ImageWidget(QPixmap.fromImage(img_d['qimage'], flags=(Qt.AutoColor | Qt.DiffuseDither)))
        self.update()


class InferenceInterface(QWidget):
    def __init__(self):
        super().__init__(flags=Qt.WindowTitleHint | Qt.WindowCloseButtonHint |
                               Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint)
        self.path = None
        self.setGeometry(300, 150, 800, 600)
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
        image_splitter.setSizes([158, self.width()-158])
        image_splitter.setSizePolicy(QSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding))
        main_layout.addWidget(image_splitter)
        self.setLayout(main_layout)
        self.show()

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
