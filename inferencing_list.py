#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals
from generic_list_model import GenericListModel
from classified_image_datatype import ClassifiedImageBundle
from threading import Thread
from queue import Queue
from inference import load_model
import numpy as np


def inferencer(work_queue):
    running = True
    data = work_queue.get()
    if type(data) == bool:
        running = data
    elif type(data) == ClassifiedImageBundle:
        data.set_progress()
    while running:
        model = load_model(data.get_np_array().shape)
        prediction = model.predict(np.array([data.get_np_array() / 255]), batch_size=1)
        print(prediction[0])
        data.set_classification(prediction[0])
        work_queue.task_done()
        data = work_queue.get()
        if type(data) == bool:
            running = data
        elif type(data) == ClassifiedImageBundle:
            data.set_progress()


class InferencingList(GenericListModel):
    def __init__(self, *args):
        super().__init__(*args)
        self.work_queue = Queue()
        self.inferencer_thread = Thread(
            target=inferencer,
            args=(self.work_queue,))
        self.inferencer_thread.start()

    def stop_worker_thread(self):
        self.clear_queue()
        self.work_queue.put(False)
        self.inferencer_thread.join()

    def update_queue(self):
        self.clear_queue()
        for item in self.list:
            if item.is_undecided():
                self.work_queue.put(item)
                item.ani.start()

    def clear_queue(self):
        while not self.work_queue.empty():
            try:
                self.work_queue.get(False)
            except Queue.Empty:
                continue
            self.work_queue.task_done()

    def append(self, item):
        super().append(item)
        self.update_queue()

    def data_changed(self, item):
        super().data_changed(item)
        self.update_queue()

    def clear(self):
        super().clear()
        self.update_queue()
