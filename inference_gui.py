#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout, QVBoxLayout
from PyQt5.QtWidgets import QPushButton, QLabel, QFileDialog
from PyQt5.QtGui import QImage
import sys


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
        main_layout.addLayout(path_row)
        self.setLayout(main_layout)
        self.show()

    def open_path_select_dialog(self):
        dialog = QFileDialog()
        dialog.setWindowTitle("Pfad der Bilder auswählen")
        dialog.setFileMode(QFileDialog.Directory)
        if dialog.exec():
            self.path = dialog.selectedFiles()[0]
            self.path_label.setText("Path: " + self.path)
        else:
            self.path = None
            self.path_label.setText("Kein Pfad ausgewählt.")


if __name__ == "__main__":
    app_object = QApplication(sys.argv)
    window = InferenceInterface()
    status = app_object.exec_()
