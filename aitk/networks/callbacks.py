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
from tensorflow.keras.callbacks import Callback


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
