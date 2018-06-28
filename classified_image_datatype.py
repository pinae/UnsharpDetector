#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals
from PyQt5.QtCore import Qt, QObject, QPropertyAnimation, QSequentialAnimationGroup, QEasingCurve
from PyQt5.QtCore import pyqtSignal, pyqtProperty
from PyQt5.QtGui import QImage, QColor
from visualization_helpers import convert_image


class ClassifiedImageBundle(QObject):
    def get_animation_progress(self):
        return self._animation_progress

    def set_animation_progress(self, val):
        self._animation_progress = val
        self.data_changed.emit(self)

    UNDECIDED, CLASSIFIED, MANUAL, PROGRESS = range(4)
    data_changed = pyqtSignal(QObject)
    animation_progress = pyqtProperty(float, get_animation_progress, set_animation_progress)

    def __init__(self, *args):
        super().__init__(*args)
        self.img = None
        self.thumb = None
        self.filename = None
        self.np_array = None
        self.status = ClassifiedImageBundle.UNDECIDED
        self.color = None
        self.keep = None
        self.show_buttons = False
        self._animation_progress = 1.0
        self.ani = QSequentialAnimationGroup()
        self.init_animation()

    def init_animation(self):
        ani1 = QPropertyAnimation(self, b"animation_progress")
        ani1.setDuration(3700)
        ani1.setEasingCurve(QEasingCurve.InOutQuad)
        ani1.setStartValue(-0.001)
        ani1.setEndValue(-1.0)
        self.ani.addAnimation(ani1)
        ani2 = QPropertyAnimation(self, b"animation_progress")
        ani2.setDuration(2300)
        ani2.setEasingCurve(QEasingCurve.InOutQuad)
        ani2.setStartValue(0.0)
        ani2.setEndValue(1.0)
        self.ani.addAnimation(ani2)
        self.ani.setLoopCount(-1)

    def set_np_image(self, np_array, thumb_width=128):
        self.np_array = np_array
        self.img = QImage(convert_image(np_array), np_array.shape[1], np_array.shape[0], QImage.Format_RGB32)
        self.thumb = self.img.scaledToWidth(thumb_width, mode=Qt.SmoothTransformation)

    def set_filename(self, filename):
        self.filename = filename

    def set_image_from_filename(self, filename, thumb_width=128):
        self.filename = filename
        self.img = QImage(filename)
        self.thumb = self.img.scaledToWidth(thumb_width, mode=Qt.SmoothTransformation)

    def set_progress(self):
        self.status = ClassifiedImageBundle.PROGRESS
        self.color = QColor(148, 148, 255)
        self.ani.start()

    def set_manual(self, decision):
        self.keep = decision
        self.status = ClassifiedImageBundle.MANUAL
        if self.keep:
            self.color = QColor(0, 255, 0)
        else:
            self.color = QColor(255, 0, 0)
        self.ani.stop()
        self._animation_progress = 1.0

    def set_classification(self, result):
        if self.status != ClassifiedImageBundle.MANUAL:
            self.keep = result[0] > result[1]
            self.status = ClassifiedImageBundle.CLASSIFIED
            self.color = QColor(int(result[1] * 255), int(result[0] * 255), 0)
            self.ani.stop()
            self._animation_progress = 1.0

    def set_show_buttons(self, button_state=False):
        self.show_buttons = button_state
        self.data_changed.emit(self)

    def get_show_buttons(self):
        return self.show_buttons

    def get_thumb(self):
        return self.thumb

    def get_image(self):
        return self.img

    def has_color(self):
        return self.color is not None and \
               self.status in [ClassifiedImageBundle.CLASSIFIED,
                               ClassifiedImageBundle.MANUAL,
                               ClassifiedImageBundle.PROGRESS]

    def get_color(self):
        return self.color

    def is_classified(self):
        return self.status in [ClassifiedImageBundle.CLASSIFIED,
                               ClassifiedImageBundle.PROGRESS]

    def reset(self):
        self.set_show_buttons(False)
