# -*- coding: utf-8 -*-
# ******************************************************
# aitk.networks: Keras model wrapper with visualizations
#
# Copyright (c) 2021 Douglas S. Blank
#
# https://github.com/ArtificialIntelligenceToolkit/aitk.networks
#
# ******************************************************

import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import Callback, EarlyStopping


class PlotCallback(Callback):
    def __init__(self, network, report_rate):
        super().__init__()
        self._network = network
        self._report_rate = report_rate
        self._figure = None

    def on_train_begin(self, logs=None):
        print("Training %s..." % self._network.name)

    def on_epoch_end(self, epoch, logs=None):
        self._network.plot_results(self, logs, self._report_rate)

    def on_train_end(self, logs=None):
        if self._figure is not None:
            plt.close()

def make_early_stop(monitor, patience):
    return EarlyStopping(monitor=monitor, patience=patience)

def make_stop(metric, goal, patience, use_validation):
    return StopWhen(metric, goal, patience, use_validation)

class StopWhen(Callback):
    def __init__(self, metric="acc", goal=1.0, patience=0, use_validation=False, verbose=True):
        super().__init__()
        self.metric = metric
        self.goal = goal
        self.patience = patience
        self.use_validation = use_validation
        self.wait = 0
        self.verbose = verbose

    def get_metric_names(self):
        if self.metric.startswith("acc"):
            if self.use_validation:
                return ["val_accuracy", "val_acc"]
            else:
                return ["accuracy", "acc"]
        else:
            if self.use_validation:
                return ["val_loss", "val_loss"]
            else:
                return ["loss"]

    def on_train_begin(self, logs=None):
        self.wait = 0
        self.i_stopped_it = False

    def on_train_end(self, logs=None):
        if self.i_stopped_it and self.verbose > 0:
            prefix = "val_" if self.use_validation else ""
            item = "%s%s" % (prefix, self.metric)
            print("Stopped because %s beat goal of %s" % (item, self.goal))

    def compare(self, value, goal):
        if self.metric.startswith("acc"):
            return value >= goal
        else:
            return value <= goal

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            metric_value = None
            for key in self.get_metric_names():
                if key in logs:
                    metric_value = logs[key]
                    break
            if metric_value is not None:
                if self.compare(metric_value, self.goal):
                    if self.wait >= self.patience:
                        self.model.stop_training = True
                        self.i_stopped_it = True
                    else:
                        self.wait += 1
                else:
                    # else, go back to zero and start over
                    self.wait = 0
