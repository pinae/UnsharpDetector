# -*- coding: utf-8 -*-
from PyQt5.QtCore import Qt, QSize, QLineF, QEvent
from PyQt5.QtWidgets import QStyledItemDelegate
from PyQt5.QtGui import QPen, QBrush, QPainter, QColor, QMouseEvent
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
                line_start_pos = -1 * min(0, mid.animation_progress) * len_of_all_lines
                if mid.animation_progress >= 0:
                    line_end_pos = mid.animation_progress * len_of_all_lines
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
            if mid.keep or mid.keep is None or mid.get_show_buttons():
                qp.setBrush(QColor(0, 255, 0))
                qp.setPen(QPen(QBrush(QColor(0, 255, 0)), 1.0, Qt.SolidLine, Qt.SquareCap, Qt.RoundJoin))
                qp.drawEllipse(style_option_view_item.rect.left() + mid.get_thumb().width() - 30,
                               style_option_view_item.rect.top() + mid.get_thumb().height() - 30,
                               30, 30)
                qp.setPen(QPen(QBrush(QColor(255, 255, 255)), 6.0, Qt.SolidLine, Qt.SquareCap, Qt.RoundJoin))
                qp.drawLines([
                    QLineF(style_option_view_item.rect.left() + mid.get_thumb().width() - 24,
                           style_option_view_item.rect.top() + mid.get_thumb().height() - 13,
                           style_option_view_item.rect.left() + mid.get_thumb().width() - 19,
                           style_option_view_item.rect.top() + mid.get_thumb().height() - 8),
                    QLineF(style_option_view_item.rect.left() + mid.get_thumb().width() - 19,
                           style_option_view_item.rect.top() + mid.get_thumb().height() - 8,
                           style_option_view_item.rect.left() + mid.get_thumb().width() - 7,
                           style_option_view_item.rect.top() + mid.get_thumb().height() - 20)
                ])
            if (mid.keep is not None and not mid.keep) or mid.get_show_buttons():
                qp.setBrush(QColor(255, 0, 0))
                qp.setPen(QPen(QBrush(QColor(255, 0, 0)), 1.0, Qt.SolidLine, Qt.SquareCap, Qt.RoundJoin))
                qp.drawEllipse(style_option_view_item.rect.left() + 8,
                               style_option_view_item.rect.top() + mid.get_thumb().height() - 30,
                               30, 30)
                qp.setPen(QPen(QBrush(QColor(255, 255, 255)), 6.0, Qt.SolidLine, Qt.SquareCap, Qt.RoundJoin))
                qp.drawLine(style_option_view_item.rect.left() + 16,
                            style_option_view_item.rect.top() + mid.get_thumb().height() - 22,
                            style_option_view_item.rect.left() + 30,
                            style_option_view_item.rect.top() + mid.get_thumb().height() - 8)
                qp.drawLine(style_option_view_item.rect.left() + 16,
                            style_option_view_item.rect.top() + mid.get_thumb().height() - 8,
                            style_option_view_item.rect.left() + 30,
                            style_option_view_item.rect.top() + mid.get_thumb().height() - 22)
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
        if type(event) != QMouseEvent:
            return super().editorEvent(event, model, style_option_view_item, model_index)
        mid = model_index.data()
        if type(mid) is ClassifiedImageBundle:
            x_in_delegate = event.pos().x() - style_option_view_item.rect.left()
            y_in_delegate = event.pos().y() - style_option_view_item.rect.top()
            thumb_w = model_index.data().get_thumb().width()
            thumb_h = model_index.data().get_thumb().height()
            if event.type() == QEvent.MouseMove:
                model.reset_whole_list()
                if 4 < x_in_delegate < 4 + thumb_w and 4 < y_in_delegate < 4 + thumb_h:
                    model_index.data().set_show_buttons(True)
            if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                if 9 <= x_in_delegate <= 39 and thumb_h - 30 <= y_in_delegate <= thumb_h:
                    model_index.data().set_manual(False)
                elif thumb_w - 30 <= x_in_delegate <= thumb_w and thumb_h - 30 <= y_in_delegate <= thumb_h:
                    model_index.data().set_manual(True)
                elif 4 < x_in_delegate < 4 + thumb_w and 4 < y_in_delegate < 4 + thumb_h:
                    model_index.data().select()
        return super().editorEvent(event, model, style_option_view_item, model_index)
