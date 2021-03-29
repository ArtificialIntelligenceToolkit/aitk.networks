# -*- coding: utf-8 -*-
# ******************************************************
# aitk.networks: Keras model wrapper with visualizations
#
# Copyright (c) 2021 Douglas S. Blank
#
# https://github.com/ArtificialIntelligenceToolkit/aitk.networks
#
# ******************************************************

import numpy as np
import re

from .utils import (
    image_to_uri,
)

class WeightWatcher():
    def __init__(self, network, from_name, to_name):
        self.network = network
        self.from_name = from_name
        self.to_name = to_name
        self.name = "WeightWatcher: from %s to %s" % (from_name, to_name)

    def update(self, **kwargs):
        weights = self.network._model.get_weights()
        image_uri = image_to_uri(image)
        width, height = image.size
        div = """<div style="outline: 5px solid #1976D2FF; width: %spx; height: %spx;"><image src="%s"></image></div>""" % (width, height, image_uri)
        self._widget.value = div

    def get_widget(self):
        from ipywidgets import HTML

        if self._widget is None:
            self._widget = HTML()

        self.update()

        return self._widget

class LayerWatcher():
    def __init__(self, network, layer_name):
        self.name = "LayerWatcher: %s" % (layer_name)
        self.network = network
        self.layer_name = layer_name
        self._widget = None
        self.get_widget()

    def update(self, inputs=None, targets=None):
        if inputs is None and targets is None:
            return

        if len(self.network.input_bank_order) == 1:
            inputs = [np.array([inputs])]
        else:
            inputs = [np.array([bank]) for bank in inputs]

        image = self.network.make_image(
            self.layer_name, self.network.predict_to(inputs, self.layer_name)[0]
        )
        image_uri = image_to_uri(image)
        width, height = image.size
        div = """<div style="outline: 5px solid #1976D2FF; width: %spx; height: %spx;"><image src="%s"></image></div>""" % (width, height, image_uri)
        self._widget.value = div

    def get_widget(self):
        from ipywidgets import HTML

        if self._widget is None:
            self._widget = HTML()
            image = self.network.make_image(
                self.layer_name, np.array(self.network.make_dummy_vector(self.layer_name)),
            )
            image_uri = image_to_uri(image)
            width, height = image.size
            div = """<div style="outline: 5px solid #1976D2FF; width: %spx; height: %spx;"><image src="%s"></image></div>""" % (width, height, image_uri)
            self._widget.value = div

        return self._widget


class NetworkWatcher():
    def __init__(self,
                 network,
                 show_error=None,
                 show_targets=None,
                 rotate=None,
                 scale=None,
    ):
        self.name = "NetworkWatcher"
        self.network = network
        self._widget_kwargs = {}
        self._widget = None
        # Update the defaults:
        if show_error is not None:
            self._widget_kwargs["show_error"] = show_error
        if show_targets is not None:
            self._widget_kwargs["show_targets"] = show_targets
        if rotate is not None:
            self._widget_kwargs["rotate"] = rotate
        if scale is not None:
            self._widget_kwargs["scale"] = scale
        self.get_widget(**self._widget_kwargs)

    def update(self, inputs=None, targets=None):
        if inputs is None and targets is None:
            return

        svg = self.network.get_image(inputs, targets, format="svg", **self._widget_kwargs)

        # Watched items get a border
        # Need width and height; we get it out of svg:
        header = svg.split("\n")[0]
        width = int(re.match('.*width="(\d*)px"', header).groups()[0])
        height = int(re.match('.*height="(\d*)px"', header).groups()[0])
        div = """<div style="outline: 5px solid #1976D2FF; width: %spx; height: %spx;">%s</div>""" % (width, height, svg)
        self._widget.value = div

    def get_widget(self,
        show_error=None,
        show_targets=None,
        rotate=None,
        scale=None,
    ):
        """
        """
        from ipywidgets import HTML

        # Update the defaults:
        if show_error is not None:
            self._widget_kwargs["show_error"] = show_error
        if show_targets is not None:
            self._widget_kwargs["show_targets"] = show_targets
        if rotate is not None:
            self._widget_kwargs["rotate"] = rotate
        if scale is not None:
            self._widget_kwargs["scale"] = scale

        svg = self.network.get_image(format="svg", **self._widget_kwargs)

        # Watched items get a border
        # Need width and height; we get it out of svg:
        header = svg.split("\n")[0]
        width = int(re.match('.*width="(\d*)px"', header).groups()[0])
        height = int(re.match('.*height="(\d*)px"', header).groups()[0])
        div = """<div style="outline: 5px solid #1976D2FF; width: %spx; height: %spx;">%s</div>""" % (width, height, svg)

        if self._widget is None:
            # Singleton:
            self._widget = HTML(value=div)
        else:
            self._widget.value = div

        return self._widget
