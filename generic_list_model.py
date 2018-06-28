#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals
from PyQt5.QtCore import Qt, QAbstractListModel


class GenericListModel(QAbstractListModel):
    def __init__(self, *args):
        super().__init__(*args)
        self.list = []

    def rowCount(self, parent=None, *args, **kwargs):
        if parent:
            return len(self.list)

    def data(self, index, role=None):
        return self.list[index.row()]

    def data_by_int_index(self, index):
        return self.list[index]

    def append(self, item):
        item.data_changed.connect(self.data_changed)
        self.list.append(item)

    def data_changed(self, item):
        model_index = self.createIndex(self.list.index(item), 0, item)
        self.setData(model_index, item)

    def setData(self, model_index, data, role=Qt.EditRole):
        super().setData(model_index, data, role=role)
        self.dataChanged.emit(model_index, model_index, [role])

    def reset_whole_list(self):
        for item in self.list:
            item.reset()
