#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals
from PyQt5.QtCore import Qt, QSize, QLineF
from PyQt5.QtWidgets import QStyledItemDelegate
from PyQt5.QtGui import QPen, QBrush, QPainter
from classified_image_datatype import ClassifiedImageBundle


class ImageableStyledItemDelegate(QStyledItemDelegate):
    def __init__(self, parent=None, *args):
        super().__init__(*args)
        self.setParent(parent)

    def paint(self, qp, style_option_view_item, model_index):
        mid = model_index.data()
        if type(mid) is ClassifiedImageBundle:
            qp.save()
            qp.drawImage(style_option_view_item.rect.left() + 4, style_option_view_item.rect.top() + 4, mid.get_thumb())
            qp.setRenderHint(QPainter.Antialiasing)
            qp.setRenderHint(QPainter.HighQualityAntialiasing)
            if mid.has_color():
                qp.setPen(QPen(QBrush(mid.get_color()), 4.0,
                               Qt.DotLine if mid.is_classified() else Qt.SolidLine,
                               Qt.SquareCap, Qt.RoundJoin))
                lines_to_draw = []
                len_of_all_lines = 2 * (mid.get_thumb().height() + mid.get_thumb().width() + 12)
                line_start_pos = -1 * min(0, mid.get_animation_progress()) * len_of_all_lines
                if mid.get_animation_progress() >= 0:
                    line_end_pos = mid.get_animation_progress() * len_of_all_lines
                else:
                    line_end_pos = 1 * len_of_all_lines
                tx = style_option_view_item.rect.left() + 2
                ty = style_option_view_item.rect.top() + 2
                h = mid.get_thumb().height() + 4
                w = mid.get_thumb().width() + 4
                if line_start_pos <= h and 0 < line_end_pos:
                    lines_to_draw.append(QLineF(tx,
                                                ty + line_start_pos,
                                                tx,
                                                ty + min(h, line_end_pos)))
                if line_start_pos <= h + w and h < line_end_pos:
                    lines_to_draw.append(QLineF(tx + max(line_start_pos, h) - h,
                                                ty + h,
                                                tx + min(h + w, line_end_pos) - h,
                                                ty + h))
                if line_start_pos <= 2 * h + w and h + w < line_end_pos:
                    lines_to_draw.append(QLineF(tx + w,
                                                ty + h - (max(line_start_pos - h - w, 0)),
                                                tx + w,
                                                ty + h - (min(line_end_pos - h - w, h))))
                if line_start_pos <= 2 * h + 2 * w and 2 * h + w < line_end_pos:
                    lines_to_draw.append(QLineF(tx + w - (max(line_start_pos - 2 * h - w, 0)),
                                                ty,
                                                tx + w - (min(line_end_pos - 2 * h - w, w)),
                                                ty))
                qp.drawLines(lines_to_draw)
            qp.restore()
        else:
            super().paint(qp, style_option_view_item, model_index)

    def sizeHint(self, style_option_view_item, model_index):
        mid = model_index.data()
        if type(mid) is ClassifiedImageBundle:
            return QSize(mid.get_thumb().width() + 8, mid.get_thumb().height() + 8)
        else:
            return super().sizeHint(style_option_view_item, model_index)

    def editorEvent(self, event, model, style_option_view_item, model_index):
        print("called editorEvent")
        print(event)
        print(model)
        return super().editorEvent(event, model, style_option_view_item, model_index)

    def createEditor(self, parent, style_option_view_item, model_index):
        print("called createEditor")
        print(parent)
        return super().createEditor(parent, style_option_view_item, model_index)

    def commitData(self, editor):
        print("called commitData")
        print(editor)
        return super().commitData(editor)

    def setEditorData(self, editor, model_index):
        print("called setEditorData")
        print(editor)
        return super().setEditorData(editor, model_index)
