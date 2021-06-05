# -*- coding: utf-8 -*-
# ******************************************************
# aitk.networks: Keras model wrapper with visualizations
#
# Copyright (c) 2021 Douglas S. Blank
#
# https://github.com/ArtificialIntelligenceToolkit/aitk.networks
#
# ******************************************************

import functools
import html
import io
import itertools
import math
import numbers
import operator
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from PIL import Image, ImageDraw

from .utils import (
    find_path,
    get_argument_bindings,
    get_error_colormap,
    get_layer_name,
    get_templates,
    image_to_uri,
    make_input_from_shape,
    render_curve,
    scale_output_for_image,
    svg_to_image,
    topological_sort,
)

try:
    from IPython.display import HTML, clear_output, display
except ImportError:
    HTML = None

class Network:
    """
    Wrapper around a keras.Model.
    """
    def __init__(self, model=None, layers=None, **config):
        self._initialized = False
        self._watchers = []
        self._fit_inputs = None
        self._fit_targets = None
        self._init_state()
        self._model = model
        # {name: (layer, [incoming], [outgoing])...}
        if layers is not None:
            self._pre_layers = {get_layer_name(layer): layer
                                for layer in layers}
            self._name = config.get("name", "Network")
            self._show_connection_help()
        else:
            self._pre_layers = {}
            self._name = None
        self._connections = []
        # Place to put models between layers:
        self._predict_models = {}
        # Place to map layer to its input layers:
        self._input_layer_names = {}
        # For saving HTML for watchers
        self._svg = None
        self._history = {"weights": [], "metrics": []}
        self._epoch = 0
        self._tolerance = 0.1
        name = self._model.name if self._model is not None else "Network"
        self.config = {
            "name": name,  # for svg title
            "class_id": "keras-network",  # for svg network classid
            "id": "keras-network",  # for svg id
            "font_size": 12,  # for svg
            "font_family": "monospace",  # for svg
            "font_color": "black",  # for svg
            "font_color_dynamic": "red",  # for svg
            "font_color_not_visible": "green",  # for svg
            "font_color_layer_name": "blue",  # for svg
            "border_top": 25,  # for svg
            "border_bottom": 25,  # for svg
            "hspace": 150,  # for svg
            "vspace": 30,  # for svg, arrows
            "image_maxdim": 200,  # for svg
            "image_pixels_per_unit": 50,  # for svg
            "max_draw_units": 20,
            "activation": "linear",  # Dense default, if none specified
            "arrow_color": "black",
            "arrow_width": "2",
            "border_width": "2",
            "border_color": "black",
            "background_color": "#B0C4DE",  # must use hex format here
            "show_targets": False,
            "show_error": False,
            "pixels_per_unit": 1,
            "precision": 2,
            "scale": None,  # for svg, 0 - 1, or None for optimal
            "rotate": False,  # for rotating SVG
            "smoothing": 0.02,  # smoothing curves
            "preferred_size": 400,  # in pixels
            "max_width": 800,  # in pixels
            "dashboard.dataset": "Train",
            "dashboard.features.bank": "",
            "dashboard.features.columns": 3,
            "dashboard.features.scale": 1.0,
            "layers": {},
            # layer_name: {vshape, feature, keep_aspect_ratio, visible
            # colormap, border_color, border_width}
        }
        # Get all of the layers, even implicit ones, in order:
        if self._model is not None:
            self.initialize_model()
        else:
            self._layers = []
            self._layers_map = {}
            self.input_bank_order = []
            self.output_bank_order = []
            self._level_ordering = []

        # Override settings:
        self.set_config(**config)

    def __getattr__(self, attr):
        return getattr(self._model, attr)

    def __getitem__(self, layer_name):
        return self._layers_map.get(layer_name, None)

    def _init_state(self):
        if "_state" not in dir(self):
            self._state = {
                "tolerance_accuracy_used": False,
                "pca": {},
            }

    def _show_connection_help(self):
        print("Connect layers with Network.connect(NAME, NAME) where NAMEs are in:")
        print("    ", list(self._pre_layers.keys()))

    def initialize_model(self):
        self._layers = topological_sort(self._model.layers)
        # Make a mapping of names to layers:
        self._layers_map = {layer.name: layer for layer in self._layers}
        # Get the input bank names, in order:
        self.input_bank_order = self._get_input_layers()
        # Get the output bank names, in order:
        self.output_bank_order = self._get_output_layers()
        # Get the best (shortest path) between layers:
        self._level_ordering = self._get_level_ordering()
        # Build intermediary models:
        self._build_predict_models()
        # Setup layer config dicts:
        self.config["layers"] = {layer.name: {} for layer in self._layers}
        # Set the colormap, etc for each layer:
        self.initialize()

    def initialize(self, inputs=None, reset=True):
        """
        Set colormap for each layer based on inputs or
        activation functions per layer.

        Args:
            inputs: inputs in single pattern format (not a dataset)
            reset: if True, reset the colormap ranges

        If inputs is None, just make best guess for all layers.

        If inputs is not None, use these for input layer
        colormap, and all other layers get best guess.

        If reset is True, don't use previous colormap
        for input layers, but sample from inputs again.
        If reset is False, consider previous input
        layer colormap's with new input values.
        """
        if inputs is None:
            # We don't have direct values, so we base colormap
            # on activation output ranges
            for layer in self._layers:
                if layer.name not in self.config["layers"]:
                    self.config["layers"][layer.name] = {}
                if self._get_layer_type(layer.name) == "input":
                    self.config["layers"][layer.name]["colormap"] = ("gray", -2, 2)
                else:
                    minmax = self._get_act_minmax(layer.name)
                    self.config["layers"][layer.name]["colormap"] = ("gray", minmax[0], minmax[1])
        else:
            self._initialized = True
            input_dataset = self.input_to_dataset(inputs)
            input_array = np.array(inputs)
            # If reset is true, we set to extremes so any value will adjust
            # Only do this on input layers:
            if reset:
                for layer in self._layers:
                    if self._get_layer_type(layer.name) == "input":
                        if layer.name not in self.config["layers"]:
                            self.config["layers"][layer.name] = {}
                        if input_array.shape == 3 and input_array.shape[2] == 3:
                            self.config["layers"][layer.name]["colormap"] = (
                                "gray",
                                float("+inf"), # extreme too big
                                float("-inf"), # extreme too small
                            )
                        else:
                            self.config["layers"][layer.name]["colormap"] = (
                                "color",
                                float("+inf"), # extreme too big
                                float("-inf"), # extreme too small
                            )
            # Now we set the minmax for input layer, based on past values
            # or extremes:
            for layer in self._layers:
                if self._get_layer_type(layer.name) == "input":
                    outputs = self.predict_to(input_dataset, layer.name)
                    color_orig, min_orig, max_orig = self.config["layers"][layer.name]["colormap"]
                    min_new, max_new = (
                        min(outputs.min(), min_orig),
                        max(outputs.max(), max_orig),
                    )
                    if min_new != max_new:
                        self.config["layers"][layer.name]["colormap"] = (color_orig, min_new, max_new)
                    else:
                        # Don't let them be equal:
                        self.config["layers"][layer.name]["colormap"] = (
                            color_orig,
                            min_new - 1,
                            max_new + 1,
                        )

    def connect(self, from_layer_name=None, to_layer_name=None):
        """
        """
        if len(self._pre_layers) == 0:
            raise Exception("no layers have been added")
        if from_layer_name is not None and not isinstance(from_layer_name, str):
            raise Exception("from_layer_name should be a string or None")
        if to_layer_name is not None and not isinstance(to_layer_name, str):
            raise Exception("to_layer_name should be a string or None")
        if from_layer_name is None and to_layer_name is None:
            #if (any([layer.outgoing_connections for name, layer in self.layers]) or
            #    any([layer.incoming_connections for layer in self.layers])):
            #    raise Exception("layers already have connections")
            for i in range(len(self._pre_layers) - 1):
                names = list(self._pre_layers)
                from_layer = self._pre_layers[names[i]]
                to_layer = self._pre_layers[names[i + 1]]
                self.connect(from_layer.name, to_layer.name)
        else:
            if from_layer_name == to_layer_name:
                raise Exception("self connections are not allowed")
            if not isinstance(from_layer_name, str):
                raise Exception("from_layer_name should be a string")
            if from_layer_name not in self._pre_layers:
                raise Exception('unknown layer: %s' % from_layer_name)
            if not isinstance(to_layer_name, str):
                raise Exception("to_layer_name should be a string")
            if to_layer_name not in self._pre_layers:
                raise Exception('unknown layer: %s' % to_layer_name)
            from_layer = self._pre_layers[from_layer_name]
            to_layer = self._pre_layers[to_layer_name]
            # Check for input going to a Dense to warn:
            #if len(from_layer.shape) > 2 and to_layer.__class__.__name__ == "Dense":
            #    print("WARNING: connected multi-dimensional input layer '%s' to layer '%s'; consider adding a FlattenLayer between them" % (
            #        from_layer.name, to_layer.name), file=sys.stderr)
            self._connections.append((from_layer_name, to_layer_name))

    def fit(self, *args, **kwargs):
        """
        Train the model.

        kwargs:
            * accuracy: (float) stop when accuracy gets equal or above this value
            * val_accuracy: (float) stop when validation accuracy gets equal or above this value
            * tolerance: (float) use this value for measuring accuracies
            * report_rate: (int) update the graphs every N epochs
            * loss: (float) stop when loss gets equal or below this value
            * val_loss: (float) stop when validation loss gets equal or below this value
            * patience: (int) number of epochs to wait without improvements until stopping
            * save: (bool) whether to save the weights at end of each epoch
            * monitor: (str) metric to monitor to determine whether to stop
            * callbacks: (list) list of callbacks
        """
        from .callbacks import UpdateCallback, make_early_stop, make_stop, make_save

        # plot = True
        # if plot:
        #    import matplotlib
        #    mpl_backend = matplotlib.get_backend()
        # else:
        #    mpl_backend = None

        # Get any kwargs that are not standard:
        report_rate = kwargs.pop("report_rate", 1)
        # Early stopping and Stop on Accuracy, Val_accuracy
        patience = kwargs.pop("patience", 0)
        # Our stopping criteria:
        accuracy = kwargs.pop("accuracy", None)
        tolerance = kwargs.pop("tolerance", None)
        if tolerance is not None:
            self.set_tolerance(tolerance)
        val_accuracy = kwargs.pop("val_accuracy", None)
        loss = kwargs.pop("loss", None)
        val_loss = kwargs.pop("val_loss", None)
        # Monitoring callbacks:
        # Save weights:
        save = kwargs.pop("save", 0)
        # Keras Early stopping:
        monitor = kwargs.pop("monitor", None)

        update_callback = UpdateCallback(self, report_rate)
        kwargs = get_argument_bindings(self._model.fit, args, kwargs)
        # get callbacks, if any:
        callbacks = kwargs.get("callbacks", None)
        if callbacks is None:
            callbacks = []
        # add our update callback to it:
        callbacks.append(update_callback)
        # add any other callbacks:
        if monitor is not None:
            callbacks.append(make_early_stop(monitor, patience))
        if save > 0:
            callbacks.append(make_save(self, save))
        if accuracy is not None:
            callbacks.append(make_stop("accuracy", accuracy, patience, False))
        if val_accuracy is not None:
            callbacks.append(make_stop("accuracy", val_accuracy, patience, True))
        if loss is not None:
            callbacks.append(make_stop("loss", loss, patience, False))
        if val_loss is not None:
            callbacks.append(make_stop("loss", val_loss, patience, True))
        kwargs["callbacks"] = callbacks
        kwargs["verbose"] = 0

        self._fit_inputs = kwargs.get("x", None) # inputs
        self._fit_targets = kwargs.get("y", None) # targets

        # call underlying model fit:
        try:
            history = self._model.fit(**kwargs)
        except KeyboardInterrupt:
            plt.close()
            raise KeyboardInterrupt() from None

        if save > 0:
            # FIXME: don't save if didn't go through loop?
            self._history["weights"].append((self._epoch, self.get_weights()))


        metrics = {key: history.history[key][-1] for key in history.history
                   if len(history.history[key]) > 0}

        ## FIXME: getting epochs by keyword:
        print("Epoch %d/%d %s" % (self._epoch, kwargs["epochs"], " - ".join(
            ["%s: %s" % (key, value) for (key, value) in metrics.items()])))
        return history

    def in_console(self, mpl_backend: str) -> bool:
        """
        Return True if running connected to a console; False if connected
        to notebook, or other non-console system.

        Possible values:
            * 'TkAgg' - console with Tk
            * 'Qt5Agg' - console with Qt
            * 'MacOSX' - mac console
            * 'module://ipykernel.pylab.backend_inline' - default for notebook and
              non-console, and when using %matplotlib inline
            * 'NbAgg' - notebook, using %matplotlib notebook

        Here, None means not plotting, or just use text.

        Note:
            If you are running ipython without a DISPLAY with the QT
            background, you may wish to:

            export QT_QPA_PLATFORM='offscreen'
        """
        return mpl_backend not in [
            "module://ipykernel.pylab.backend_inline",
            "NbAgg",
        ]

    def on_epoch_end(self, callback, logs, report_rate=None, clear=True):
        """
        plots loss and accuracy on separate graphs, ignoring any other
        metrics for now.

        Args:
            report_rate: if None, then just update the plot, else
                advance self._epoch, and possible update plot
        """
        format = "svg"

        if report_rate is not None:
            # update epoch and history, and then possibly draw
            # The metrics after the epoch of training, so advanced first
            self._epoch += 1
            self._history["metrics"].append((self._epoch, logs))

            if (self._epoch % report_rate) != 0:
                return

        if len(self._watchers) > 0:
            index = random.randint(0, self.get_input_length(self._fit_inputs) - 1)
            inputs = self.get_input_from_dataset(index, self._fit_inputs)
            targets = self.get_target_from_dataset(index, self._fit_targets)
            self.propagate(inputs, targets) # update watchers

        metrics = [list(history[1].keys()) for history in self._history["metrics"]]
        metrics = set([item for sublist in metrics for item in sublist])

        def match_acc(name):
            return (name.endswith("acc") or
                    name.endswith("accuracy"))

        def match_val(name):
            return name.startswith("val_")

        if callback._figure is not None:
            # figure and axes objects have already been created
            fig, loss_ax, acc_ax = callback._figure
            loss_ax.clear()
            if acc_ax is not None:
                acc_ax.clear()
        else:
            # first time called, so create figure and axes objects
            if any([match_acc(metric) for metric in metrics]):
                fig, (loss_ax, acc_ax) = plt.subplots(1, 2, figsize=(10, 4))
            else:
                fig, loss_ax = plt.subplots(1)
                acc_ax = None
            callback._figure = fig, loss_ax, acc_ax

        def get_xy(name):
            return [
                (history[0], history[1][name])
                for history in self._history["metrics"]
                if name in history[1]
            ]

        for metric in metrics:
            xys = get_xy(metric)
            x_values = [xy[0] for xy in xys]
            y_values = [xy[1] for xy in xys]
            if metric == "loss":
                loss_ax.plot(x_values, y_values, label=metric, color="r")  # red
            elif metric == "val_loss":
                loss_ax.plot(x_values, y_values, label=metric, color="orange")
            elif match_acc(metric) and not match_val(metric) and acc_ax is not None:
                acc_ax.plot(x_values, y_values, label=metric, color="b")  # blue
            elif match_acc(metric) and match_val(metric)  and acc_ax is not None:
                acc_ax.plot(x_values, y_values, label=metric, color="c")  # cyan
            # FIXME: add a chart for each metric
            # else:
            #    loss_ax.plot(x_values, y_values, label=metric)

        loss_ax.set_ylim(bottom=0)
        loss_ax.set_title("%s: Training Loss" % (self.name,))
        loss_ax.set_xlabel("Epoch")
        loss_ax.set_ylabel("Loss")
        loss_ax.legend(loc="best")
        if acc_ax is not None:
            acc_ax.set_ylim([-0.1, 1.1])
            acc_ax.set_title("%s: Training Accuracy" % (self.name,))
            acc_ax.set_xlabel("Epoch")
            acc_ax.set_ylabel("Accuracy")
            acc_ax.legend(loc="best")


        if True or format == "svg":
            # FIXME: work in console
            # if (callback is not None and not callback.in_console) or format == "svg":
            bytes = io.BytesIO()
            plt.savefig(bytes, format="svg")
            img_bytes = bytes.getvalue()
            if HTML is not None:
                if clear:
                    clear_output(wait=True)
                display(HTML(img_bytes.decode()))
            else:
                raise Exception("need to install `IPython` to display matplotlib plots")
        else:  # format is None
            plt.pause(0.01)
            # plt.show(block=False)

    def _extract_inputs(self, inputs, input_names):
        """
        Get the input_names from the inputs
        """
        # inputs is either a dict or a list, where index matches
        # the input banks.
        if isinstance(inputs, dict):
            return [np.array(inputs[name]) for name in input_names]
        elif len(self.input_bank_order) == 1:
            return inputs
        else:
            return [
                np.array(inputs[index])
                for index in [self.input_bank_order.index(name) for name in input_names]
            ]

    def build_model(self):
        from tensorflow.keras.models import Model

        # Assumes layers either added or passed in via layers
        # and connected via Network.connect()
        froms = [connect[0] for connect in self._connections]
        tos = [connect[1] for connect in self._connections]
        input_layers = []
        output_layers = []
        for layer_name in self._pre_layers:
            if layer_name not in tos:
                input_layers.append(layer_name)
            if layer_name not in froms:
                output_layers.append(layer_name)
        outputs = [self._get_tensor_to(output_layer)
                   for output_layer in output_layers]
        inputs = [self._pre_layers[layer_name]
                  for layer_name in input_layers]
        self._model = Model(inputs=inputs, outputs=outputs, name=self._name)
        self.initialize_model()

    def _get_layers_to(self, layer_name):
        return [connection[0] for connection in self._connections
                if connection[1] == layer_name]

    def _get_tensor_to(self, layer_name):
        from tensorflow.keras.layers import Concatenate

        # recursive
        layers = self._get_layers_to(layer_name)
        if len(layers) == 0:
            # An input layer:
            return self._pre_layers[layer_name]

        incoming_layers = [self._get_tensor_to(incoming_layer_name)
                           for incoming_layer_name in layers]

        if len(incoming_layers) == 1:
            incoming_layer = incoming_layers[0]
        else: # more than one
            incoming_layer = Concatenate()(incoming_layers)

        layer = self._pre_layers[layer_name]
        return layer(incoming_layer)

    def compile(self, *args, **kwargs):
        """
        """
        # First, build model if necessary:
        if self._model is None:
            self.build_model()
        # Next, handle any special names:
        metrics = kwargs.pop("metrics", [])
        if len(metrics) > 0:
            metrics = [self.get_metric(metric) for metric in metrics]
            kwargs["metrics"] = metrics
        # Let the standard keras model do the rest:
        return self._model.compile(*args, **kwargs)


    def predict(self, inputs, format="numpy"):
        """
        Propagate input patterns to a bank in the network.

        * format: (str) "numpy", "list", or "image"
        """
        input_vectors = self._extract_inputs(inputs, self.input_bank_order)
        try:
            outputs = self._model(input_vectors, training=False).numpy()
        except Exception as exc:
            input_layers_shapes = [
                self._get_raw_output_shape(layer_name) for layer_name in self.input_bank_order
            ]
            hints = ", ".join(
                [
                    ("%s: %s" % (name, shape))
                    for name, shape in zip(self.input_bank_order, input_layers_shapes)
                ]
            )
            raise Exception(
                "You must supply the inputs for these banks in order and in the right shape: %s"
                % hints
            ) from None

        if format == "numpy":
            return outputs
        elif format == "list":
            return outputs.tolist()

    def set_pca_spaces(self, inputs):
        """
        Make the pca_space with the current weights.
        """
        from sklearn.decomposition import PCA

        for layer in self.layers:
            pca = PCA(2)
            hidden_raw = self.predict_to(inputs, layer.name)
            try:
                pca_space = pca.fit(hidden_raw)
            except ValueError:
                # Only one hidden layer unit
                pca_space = None
            self._state["pca"][layer.name] = pca_space

    def get_input_length(self, inputs):
        if len(self.input_bank_order) == 1:
            return len(inputs)
        else:
            return len(inputs[0])

    def predict_histogram_to(self, inputs, layer_name):
        hidden_raw = self.predict_to(inputs, layer_name)

        plt.hist(hidden_raw)
        plt.axis('off')
        fp = io.BytesIO()
        plt.savefig(fp, format="png")
        plt.close()
        image = Image.open(fp)
        return image

    def predict_pca_to(self, inputs, layer_name, colors, sizes):
        if layer_name not in self._state["pca"]:
            raise Exception("Need to set_pca_spaces first")

        hidden_raw = self.predict_to(inputs, layer_name)
        pca_space = self._state["pca"][layer_name]
        if pca_space is not None:
            hidden_pca = pca_space.transform(hidden_raw)
            x = hidden_pca[:,0]
            y = hidden_pca[:,1]
        else:
            # Only one hidden layer unit; we'll use zeros for Y axis
            x = hidden_raw
            y = np.zeros(len(hidden_raw))

        plt.scatter(x, y, c=colors, s=sizes)
        plt.axis('off')
        fp = io.BytesIO()
        plt.savefig(fp, format="png")
        plt.close()
        image = Image.open(fp)
        return image

    def predict_pca(
        self,
        inputs=None,
        targets=None,
        show_error=False,
        show_targets=False,
        format=None,
        rotate=False,
        scale=None,
        clear=True,
        set_pca_spaces=True,
        colors=None,
        sizes=None,
        **config,
    ):
        """
        """
        # This are not sticky; need to set each time:
        config["rotate"] = rotate
        config["scale"] = scale
        # Everything else is sticky:
        self.config.update(config)

        if set_pca_spaces:
            self.set_pca_spaces(inputs)

        try:
            svg = self.to_svg(inputs=inputs, targets=targets, mode="pca",
                              colors=colors, sizes=sizes)
        except KeyboardInterrupt:
            raise KeyboardInterrupt() from None

        if format is None:
            try:
                get_ipython()  # noqa: F821
                format = "html"
            except Exception:
                format = "image"

        if format == "html":
            if HTML is not None:
                if clear:
                    clear_output(wait=True)
                display(HTML(svg))
            else:
                raise Exception(
                    "need to install `IPython` or use Network.predict_pca(format='image')"
                )
        elif format == "svg":
            return svg
        elif format == "image":
            return svg_to_image(svg, self.config)
        else:
            raise ValueError("unable to convert to format %r" % format)


    def predict_histogram(
        self,
        inputs=None,
        targets=None,
        show_error=False,
        show_targets=False,
        format=None,
        rotate=False,
        scale=None,
        clear=True,
        **config,
    ):
        """
        """
        # This are not sticky; need to set each time:
        config["rotate"] = rotate
        config["scale"] = scale
        # Everything else is sticky:
        self.config.update(config)

        try:
            svg = self.to_svg(inputs=inputs, targets=targets, mode="histogram")
        except KeyboardInterrupt:
            raise KeyboardInterrupt() from None

        if format is None:
            try:
                get_ipython()  # noqa: F821
                format = "html"
            except Exception:
                format = "image"

        if format == "html":
            if HTML is not None:
                if clear:
                    clear_output(wait=True)
                display(HTML(svg))
            else:
                raise Exception(
                    "need to install `IPython` or use Network.predict_pca(format='image')"
                )
        elif format == "svg":
            return svg
        elif format == "image":
            return svg_to_image(svg, self.config)
        else:
            raise ValueError("unable to convert to format %r" % format)


    def predict_to(self, inputs, layer_name, format="numpy"):
        """
        Propagate input patterns to a bank in the network.

        * format: (str) "numpy", "list", or "image"
        """
        input_names = self._input_layer_names[layer_name]
        model = self._predict_models[input_names, layer_name]
        input_vectors = self._extract_inputs(inputs, input_names)
        try:
            outputs = model(input_vectors, training=False).numpy()
        except Exception as exc:
            input_layers_shapes = [
                self._get_raw_output_shape(layer_name) for layer_name in input_names
            ]
            hints = ", ".join(
                [
                    ("%s: %s" % (name, shape))
                    for name, shape in zip(input_names, input_layers_shapes)
                ]
            )
            raise Exception(
                "You must supply the inputs for these banks in order and in the right shape: %s"
                % hints
            ) from None

        if format == "numpy":
            return outputs
        elif format == "image":
            return self.make_image(layer_name, outputs)
        elif format == "list":
            return outputs.tolist()

    def predict_from(self, inputs, from_layer_name, to_layer_name):
        """
        Propagate patterns from one bank to another bank in the network.
        """
        from tensorflow.keras.models import Model

        key = (tuple([from_layer_name]), to_layer_name)
        if key not in self._predict_models:
            from_layer = self[from_layer_name]
            path = find_path(from_layer, to_layer_name)
            # Input should be what next layer expects:
            current = input_layer = make_input_from_shape(self[path[0]].input_shape)
            for layer_name in path:
                current = self[layer_name](current)
            self._predict_models[key] = Model(inputs=input_layer, outputs=current)
        model = self._predict_models[key]
        return model(inputs, training=False).numpy()


    def display_colormap(self, colormap=None):
        """
        Display one, or all of the colormaps available on your system.
        """
        if colormap is None:
            if HTML:
                for colorname in cm.cmap_d.keys():
                    display(colorname, self.display_colormap(colorname))
            else:
                raise Exception("you need to install IPython for this function to work")
        else:
            width, height = (400, 25)
            vector = np.arange(0, 1, 0.01)
            vector = vector.reshape((1, 100))
            cm_hot = cm.get_cmap(colormap)
            vector = cm_hot(vector)
            vector = np.uint8(vector * 255)
            image = Image.fromarray(vector)
            image = image.resize((width, height))
            return image

    def get_image(
        self,
        inputs=None,
        targets=None,
        show_error=False,
        show_targets=False,
        format="image",
        rotate=False,
        scale=None,
        **config,
    ):
        """
        Create an SVG of the network given some inputs (optional).

        Inputs and targets are for a single pattern.

        Arguments:
            inputs: input values to propagate
            targets: target values to show
            show_error (bool): show the output error in resulting picture
            show_targets (bool): show the targets in resulting picture
            format (str): optional "html", "image", or "svg"

        Examples:
            >>> net = Network("Picture", 2, 2, 1)
            >>> net.compile(error="mse", optimizer="adam")
            >>> net.display()
            <IPython.core.display.HTML object>
            >>> net.display([.5, .5])
            <IPython.core.display.HTML object>
            >>> net.display([.5, .5])
            <IPython.core.display.HTML object>
        """
        # This are not sticky; need to set each time:
        config["rotate"] = rotate
        config["scale"] = scale
        # Everything else is sticky:
        self.config.update(config)

        if not self._initialized and inputs is not None:
            self.initialize(inputs)

        try:
            svg = self.to_svg(inputs=inputs, targets=targets, mode="activation",
                              colors=None, sizes=None)
        except KeyboardInterrupt:
            raise KeyboardInterrupt() from None

        if format == "svg":
            return svg
        elif format == "image":
            return svg_to_image(svg, self.config)
        else:
            raise ValueError("unable to convert to format %r" % format)

    def display(
        self,
        inputs=None,
        targets=None,
        show_error=False,
        show_targets=False,
        format=None,
        rotate=False,
        scale=None,
        clear=True,
        **config,
    ):
        if format is None:
            try:
                get_ipython()  # noqa: F821
                format = "html"
            except Exception:
                format = "image"

        if format == "html":
            svg = self.get_image(inputs, targets, show_error, show_targets, "svg",
                                 rotate, scale, **config)
            if HTML is not None:
                if clear:
                    clear_output(wait=True)
                display(HTML(svg))
            else:
                raise Exception(
                    "need to install `IPython` or use Network.display(format='image')"
                )
        else:
            image = self.get_image(inputs, targets, show_error, show_targets, format,
                                   rotate, scale, **config)
            return image

    def watch_weights(self, to_name):
        """
        """
        from .watchers import WeightWatcher

        name = "WeightWatcher: to %s" % (to_name,)
        names = [watcher.name for watcher in self._watchers]
        if name not in names:
            watcher = WeightWatcher(self, to_name)
            self._watchers.append(watcher)
        else:
            watcher = self._watchers[names.index(name)]

        display(watcher._widget)

    def watch_layer(self, layer_name):
        """
        """
        from .watchers import LayerWatcher

        name = "LayerWatcher: %s" % (layer_name,)
        names = [watcher.name for watcher in self._watchers]
        if name not in names:
            watcher = LayerWatcher(self, layer_name)
            self._watchers.append(watcher)
        else:
            watcher = self._watchers[names.index(name)]

        display(watcher._widget)

    def watch(self,
        show_error=None,
        show_targets=None,
        rotate=None,
        scale=None,
    ):
        """
        """
        from .watchers import NetworkWatcher

        name = "NetworkWatcher"
        names = [watcher.name for watcher in self._watchers]
        if name not in names:
            watcher = NetworkWatcher(self, show_error, show_targets, rotate, scale)
            self._watchers.append(watcher)
            widget = watcher._widget
        else:
            watcher = self._watchers[names.index(name)]
            widget = watcher.get_widget(show_error, show_targets, rotate, scale)
        display(widget)

    def propagate(self,
                  inputs=None,
                  targets=None,
                  show=True,
    ):
        """
        Update all of the watchers whatever they may be watching,
        and return the propagation of the inputs through the network.
        """
        if show:
            for watcher in self._watchers:
                watcher.update(inputs, targets)
        return self._model(self.input_to_dataset(inputs), training=False).numpy()

    def propagate_each(self,
               inputs=None,
               targets=None,
    ):
        """
        Update all of the watchers whatever they may be watching.
        """
        if inputs is not None and targets is not None:
            count = 0
            for ins, targs in self.enumerate_dataset(inputs, targets):
                for watcher in self._watchers:
                    watcher.update(ins, targs)
                    if HTML is not None:
                        clear_output()
                    input("Pattern %s; press enter to continue: " % count)
                    count += 1
        elif inputs is not None:
            count = 0
            for ins in self.enumerate_dataset(inputs):
                for watcher in self._watchers:
                    watcher.update(ins)
                    if HTML is not None:
                        clear_output()
                    input("Pattern %s; press enter to continue: " % count)
                    count += 1

    def _build_predict_models(self):
        from tensorflow.keras.models import Model

        # for all layers, inputs to here:
        for layer in self._layers:
            if self._get_layer_type(layer.name) != "input":
                inputs = self._get_input_tensors(layer.name, [])
                input_names = [inp[0] for inp in inputs]
                input_tensors = [inp[1] for inp in inputs]
                self._input_layer_names[layer.name] = tuple(input_names)
                self._predict_models[tuple(input_names), layer.name] = Model(
                    inputs=input_tensors,
                    outputs=self[layer.name].output,  # tensors, tensor
                )
            else:
                self._input_layer_names[layer.name] = tuple([layer.name])
                self._predict_models[tuple([layer.name]), layer.name] = Model(
                    inputs=[layer.input], outputs=[layer.output],
                )

    def _get_input_tensors(self, layer_name, input_list):
        """
        Given a layer_name, return {input_layer_name: tensor}
        """
        # Recursive; results in input_list of [(name, tensor), ...]
        for layer in self.incoming_layers(layer_name):
            if layer.name in self._input_layer_names:
                for layer_name in self._input_layer_names[layer.name]:
                    if layer_name not in [name for (name, tensor) in input_list]:
                        input_list.append((layer_name, self[layer_name].input))
            else:
                if self._get_layer_type(layer.name) == "input":
                    if layer.name not in [name for (name, tensor) in input_list]:
                        input_list.append((layer.name, layer.input))
                else:
                    self._get_input_tensors(layer.name, input_list)
        return input_list

    def make_dummy_vector(self, layer_name):
        """
        """
        shape = self._get_output_shape(layer_name)
        if (shape is None) or (isinstance(shape, (list, tuple)) and None in shape):
            v = np.random.rand(100)
        else:
            v = np.random.rand(*shape)

        color, lo, hi = self._get_colormap(layer_name)
        # scale the vector to the min and the max of this layer:
        return np.interp(v, (v.min(), v.max()), (lo, hi))

    def make_image(self, layer_name, vector, colormap=None):
        """
        Given an activation name (or function), and an output vector, display
        make and return an image widget.
        """
        import tensorflow.keras.backend as K

        vshape = self.vshape(layer_name)
        if vshape and vshape != self._get_output_shape(layer_name):
            vector = vector.reshape(vshape)
        if len(vector.shape) > 2:
            # Drop dimensions of vector:
            s = slice(None, None)
            args = []
            # The data is in the same format as Keras
            # so we can ask Keras what that format is:
            # ASSUMES: that the network that loaded the
            # dataset has the same image_data_format as
            # now:
            if K.image_data_format() == "channels_last":
                for d in range(len(vector.shape)):
                    if d in [0, 1]:
                        args.append(s)  # keep the first two
                    else:
                        args.append(self._get_feature(layer_name))  # pick which to use
            else:  # 'channels_first'
                count = 0
                for d in range(len(vector.shape)):
                    if d in [0]:
                        args.append(self._get_feature(layer_name))  # pick which to use
                    else:
                        if count < 2:
                            args.append(s)
                            count += 1
            vector = vector[tuple(args)]
        if colormap is None:
            color, mini, maxi = self._get_colormap(layer_name)
        else:
            color, mini, maxi = colormap
        vector = scale_output_for_image(
            vector, (mini, maxi), truncate=True
        )
        if len(vector.shape) == 1:
            vector = vector.reshape((1, vector.shape[0]))
        size = self.config.get("pixels_per_unit", 1)
        new_width = vector.shape[0] * size  # in, pixels
        new_height = vector.shape[1] * size  # in, pixels
        if color == "color":
            # don't colormap it
            pass
        else:
            try:
                cm_hot = cm.get_cmap(color)
            except Exception:
                cm_hot = cm.get_cmap("gray")
            vector = cm_hot(vector)
            vector = np.uint8(vector * 255)
        if max(vector.shape) <= self.config["max_draw_units"]:
            # Need to make it bigger, to draw circles:
            # Make this value too small, and borders are blocky;
            # too big and borders are too thin
            scale = int(250 / max(vector.shape))
            size = size * scale
            image = Image.new(
                "RGBA", (new_height * scale, new_width * scale), color="white"
            )
            draw = ImageDraw.Draw(image)
            for row in range(vector.shape[1]):
                for col in range(vector.shape[0]):
                    # upper-left, lower-right:
                    draw.rectangle(
                        (
                            row * size,
                            col * size,
                            (row + 1) * size - 1,
                            (col + 1) * size - 1,
                        ),
                        fill=self._make_color(vector[col][row]),
                        outline="black",
                    )
        else:
            image = Image.fromarray(vector)
            image = image.resize((new_height, new_width))
        # If rotated, and has features, rotate it:
        if self.config.get("rotate", False):
            output_shape = self._get_output_shape(layer_name)
            vshape = self.vshape(layer_name)
            if (isinstance(output_shape, tuple) and len(output_shape) >= 3) or (
                vshape is not None and len(vshape) == 2
            ):
                image = image.rotate(90, expand=1)
        return image

    def _make_color(self, item):
        if isinstance(item, numbers.Number):
            return (item, item, item)
        else:
            return tuple(item)

    def _get_input_layers(self):
        return tuple(
            [x.name for x in self._layers if self._get_layer_type(x.name) == "input"]
        )

    def _get_output_layers(self):
        return tuple(
            [x.name for x in self._layers if self._get_layer_type(x.name) == "output"]
        )

    def vshape(self, layer_name):
        """
        Find the vshape of layer.
        """
        if (
            layer_name in self.config["layers"]
            and "vshape" in self.config["layers"][layer_name]
        ):
            return self.config["layers"][layer_name]["vshape"]
        else:
            return None

    def _get_output_shape(self, layer_name):
        layer = self[layer_name]
        if isinstance(layer.output_shape, list):
            return layer.output_shape[0][1:]
        else:
            return layer.output_shape[1:]

    def _get_raw_output_shape(self, layer_name):
        layer = self[layer_name]
        return layer.output_shape

    def _get_feature(self, layer_name):
        """
        Which feature plane is selected to show? Defaults to 0
        """
        if (
            layer_name in self.config["layers"]
            and "feature" in self.config["layers"][layer_name]
        ):
            return self.config["layers"][layer_name]["feature"]
        else:
            return slice(None, None)

    def _get_keep_aspect_ratio(self, layer_name):
        if (
            layer_name in self.config["layers"]
            and "keep_aspect_ratio" in self.config["layers"][layer_name]
        ):
            return self.config["layers"][layer_name]["keep_aspect_ratio"]
        else:
            return False

    def _get_tooltip(self, layer_name):
        """
        String (with newlines) for describing layer."
        """

        def format_range(minmax):
            minv, maxv = minmax
            if minv <= -2:
                minv = "-Infinity"
            if maxv >= +2:
                maxv = "+Infinity"
            return "(%s, %s)" % (minv, maxv)

        layer = self[layer_name]
        activation = self._get_activation_name(layer)
        retval = "Layer: %s %r" % (
            html.escape(self[layer_name].name),
            layer.__class__.__name__,
        )
        if activation:
            retval += "\nAct function: %s" % activation
            retval += "\nAct output range: %s" % (
                format_range(self._get_act_minmax(layer_name),)
            )
        retval += "\nShape = %s" % (self._get_raw_output_shape(layer_name),)
        return retval

    def _get_visible(self, layer_name):
        if (
            layer_name in self.config["layers"]
            and "visible" in self.config["layers"][layer_name]
        ):
            return self.config["layers"][layer_name]["visible"]
        else:
            return True

    def _get_colormap(self, layer_name):
        if (
            layer_name in self.config["layers"]
            and "colormap" in self.config["layers"][layer_name]
        ):
            return self.config["layers"][layer_name]["colormap"]
        else:
            return ("gray", -2, 2)

    def _get_activation_name(self, layer):
        if hasattr(layer, "activation"):
            return layer.activation.__name__

    def _get_act_minmax(self, layer_name):
        """
        Get the activation (output) min/max for a layer.

        Note: +/- 2 represents infinity
        """
        layer = self[layer_name]
        if layer.__class__.__name__ == "Flatten":
            in_layer = self.incoming_layers(layer_name)[0]
            return self._get_act_minmax(in_layer.name)
        elif self._get_layer_type(layer_name) == "input":
            color, mini, maxi = self._get_colormap(layer)
            return (mini, maxi)
        else:  # try to get from activation function
            activation = self._get_activation_name(layer)
            if activation in ["tanh", "softsign"]:
                return (-1, +1)
            elif activation in ["sigmoid", "softmax", "hard_sigmoid"]:
                return (0, +1)
            elif activation in ["relu", "elu", "softplus"]:
                return (0, +2)
            elif activation in ["selu", "linear"]:
                return (-2, +2)
            else:  # default, or unknown activation function
                return (-2, +2)

    def _get_border_color(self, layer_name):
        if (
            layer_name in self.config["layers"]
            and "border_color" in self.config["layers"][layer_name]
        ):
            return self.config["layers"][layer_name]["border_color"]
        else:
            return self.config["border_color"]

    def _get_border_width(self, layer_name):
        if (
            layer_name in self.config["layers"]
            and "border_width" in self.config["layers"][layer_name]
        ):
            return self.config["layers"][layer_name]["border_width"]
        else:
            return self.config["border_width"]

    def _find_spacing(self, row, ordering, max_width):
        """
        Find the spacing for a row number
        """
        return max_width / (len(ordering[row]) + 1)

    def describe_connection_to(self, layer1, layer2):
        """
        Returns a textual description of the weights for the SVG tooltip.
        """
        retval = "Weights from %s to %s" % (layer1.name, layer2.name)
        for klayer in self._layers:
            if klayer.name == layer2.name:
                weights = klayer.get_weights()
                for w in range(len(klayer.weights)):
                    retval += "\n %s has shape %s" % (
                        klayer.weights[w].name,
                        weights[w].shape,
                    )
        return retval

    def _get_input_layers_in_order(self, layer_names):
        """
        Get the input layers in order
        """
        return [
            layer_name
            for layer_name in self.input_bank_order
            if layer_name in layer_names
        ]

    def get_input_from_dataset(self, index, dataset):
        """
        Get input tensor(s) at index from dataset.
        """
        if len(self.input_bank_order) == 1:
            data = dataset[index]
        else:
            data = [bank[index] for bank in dataset]
        return data

    def get_target_from_dataset(self, index, dataset):
        """
        Get target tensor(s) at index from dataset.
        """
        if len(self.output_bank_order) == 1:
            data = dataset[index]
        else:
            data = [bank[index] for bank in dataset]
        return data

    def enumerate_dataset(self, dataset1, dataset2=None):
        """"
        Takes a dataset and turns it into individual
        sets of one pattern each.
        """
        count = 0
        while True:
            if len(self.input_bank_order) == 1:
                if count < len(dataset1):
                    data1 = dataset1[count]
                else:
                    break
            else:
                if count < len(dataset1[0]):
                    data1 = [bank[count] for bank in dataset1]
                else:
                    break

            if dataset2 is not None:
                if len(self.output_bank_order) == 1:
                    if count < len(dataset2):
                        data2 = dataset2[count]
                    else:
                        break
                else:
                    if count < len(dataset2[0]):
                        data2 = [bank[count] for bank in dataset2]
                    else:
                        break

                yield data1, data2
            else:
                yield data1

            count += 1

    def input_to_dataset(self, input):
        """
        Take input tensor(s) and turn into an appropriate
        dataset.
        """
        if len(self.input_bank_order) == 1:
            inputs = [np.array([input])]
        else:
            inputs = [np.array([bank]) for bank in input]
        return inputs

    def target_to_dataset(self, target):
        # FIXME: handle labels
        if len(self.output_bank_order) == 1:
            targets = [np.array([target])]
        else:
            targets = [np.array([bank]) for bank in target]
        return targets

    def to_svg(self, inputs=None, targets=None, mode="activation", colors=None, sizes=None):
        """
        """
        # First, turn single patterns into a dataset:
        if inputs is not None:
            if mode == "activation":
                inputs = self.input_to_dataset(inputs)
        if targets is not None:
            if mode == "activation":
                targets = self.target_to_dataset(targets)
        # Next, build the structures:
        struct = self.build_struct(inputs, targets, mode, colors, sizes)
        templates = get_templates(self.config)
        # get the header:
        svg = None
        for (template_name, dict) in struct:
            if template_name == "head_svg":
                dict["background_color"] = self.config["background_color"]
                svg = templates["head_svg"].format(**dict)
                break
        # build the rest:
        for index in range(len(struct)):
            (template_name, dict) = struct[index]
            # From config:
            dict["class_id"] = self.config["class_id"]
            if template_name != "head_svg" and not template_name.startswith("_"):
                rotate = dict.get("rotate", self.config["rotate"])
                if template_name == "label_svg" and rotate:
                    dict["x"] += 8
                    dict["text_anchor"] = "middle"
                    dict["transform"] = (
                        """ transform="rotate(-90 %s %s) translate(%s)" """
                        % (dict["x"], dict["y"], 2)
                    )
                else:
                    dict["transform"] = ""
                if template_name == "curve":
                    if not dict["drawn"]:
                        curve_svg = render_curve(
                            dict,
                            struct[(index + 1) :],  # noqa: E203
                            templates[template_name],
                            self.config,
                        )
                        svg += curve_svg
                else:
                    t = templates[template_name]
                    svg += t.format(**dict)
        svg += """</svg></g></svg>"""
        return svg

    def build_struct(self, inputs, targets, mode, colors, sizes):
        ordering = list(
            reversed(self._level_ordering)
        )  # list of names per level, input to output
        (
            max_width,
            max_height,
            row_heights,
            images,
            image_dims,
        ) = self._pre_process_struct(inputs, ordering, targets, mode, colors, sizes)
        # Now that we know the dimensions:
        struct = []
        cheight = self.config["border_top"]  # top border
        # Display targets?
        if self.config["show_targets"]:
            spacing = self._find_spacing(0, ordering, max_width)
            # draw the row of targets:
            cwidth = 0
            for (layer_name, anchor, fname) in ordering[0]:  # no anchors in output
                if layer_name + "_targets" not in images:
                    continue
                image = images[layer_name + "_targets"]
                (width, height) = image_dims[layer_name]
                cwidth += spacing - width / 2
                struct.append(
                    [
                        "image_svg",
                        {
                            "name": layer_name + "_targets",
                            "x": cwidth,
                            "y": cheight,
                            "image": image_to_uri(image),
                            "width": width,
                            "height": height,
                            "tooltip": self._get_tooltip(layer_name),
                            "border_color": self._get_border_color(layer_name),
                            "border_width": self._get_border_width(layer_name),
                            "rx": cwidth - 1,  # based on arrow width
                            "ry": cheight - 1,
                            "rh": height + 2,
                            "rw": width + 2,
                        },
                    ]
                )
                # show a label
                struct.append(
                    [
                        "label_svg",
                        {
                            "x": cwidth + width + 5,
                            "y": cheight + height / 2 + 2,
                            "label": "targets",
                            "font_size": self.config["font_size"],
                            "font_color": self.config["font_color"],
                            "font_family": self.config["font_family"],
                            "text_anchor": "start",
                        },
                    ]
                )
                cwidth += width / 2
            # Then we need to add height for output layer again, plus a little bit
            cheight += row_heights[0] + 10  # max height of row, plus some
        # Display error
        if self.config["show_error"]:
            spacing = self._find_spacing(0, ordering, max_width)
            # draw the row of errores:
            cwidth = 0
            for (layer_name, anchor, fname) in ordering[0]:  # no anchors in output
                if layer_name + "_errors" not in images:
                    continue
                image = images[layer_name + "_errors"]
                (width, height) = image_dims[layer_name]
                cwidth += spacing - (width / 2)
                struct.append(
                    [
                        "image_svg",
                        {
                            "name": layer_name + "_errors",
                            "x": cwidth,
                            "y": cheight,
                            "image": image_to_uri(image),
                            "width": width,
                            "height": height,
                            "tooltip": self._get_tooltip(layer_name),
                            "border_color": self._get_border_color(layer_name),
                            "border_width": self._get_border_width(layer_name),
                            "rx": cwidth - 1,  # based on arrow width
                            "ry": cheight - 1,
                            "rh": height + 2,
                            "rw": width + 2,
                        },
                    ]
                )
                # show a label
                struct.append(
                    [
                        "label_svg",
                        {
                            "x": cwidth + width + 5,
                            "y": cheight + height / 2 + 2,
                            "label": "error",
                            "font_size": self.config["font_size"],
                            "font_color": self.config["font_color"],
                            "font_family": self.config["font_family"],
                            "text_anchor": "start",
                        },
                    ]
                )
                cwidth += width / 2
            # Then we need to add height for output layer again, plus a little bit
            cheight += row_heights[0] + 10  # max height of row, plus some
        # Show a separator that takes no space between output and targets/errors
        if self.config["show_error"] or self.config["show_targets"]:
            spacing = self._find_spacing(0, ordering, max_width)
            # Draw a line for each column in putput:
            cwidth = spacing / 2 + spacing / 2  # border + middle of first column
            # number of columns:
            for level_tups in ordering[0]:
                struct.append(
                    [
                        "line_svg",
                        {
                            "x1": cwidth - spacing / 2,
                            "y1": cheight - 5,  # half the space between them
                            "x2": cwidth + spacing / 2,
                            "y2": cheight - 5,
                            "arrow_color": "green",
                            "tooltip": "",
                        },
                    ]
                )
                cwidth += spacing
        # Now we go through again and build SVG:
        positioning = {}
        level_num = 0
        # For each level:
        hiding = {}
        for row in range(len(ordering)):
            level_tups = ordering[row]
            # how many space at this level for this column?
            spacing = self._find_spacing(row, ordering, max_width)
            cwidth = 0
            # See if there are any connections up:
            any_connections_up = False
            for (layer_name, anchor, fname) in level_tups:
                if not self._get_visible(layer_name):
                    continue
                elif anchor:
                    continue
                for out in self.outgoing_layers(layer_name):
                    if (
                        out.name not in positioning
                    ):  # is it drawn yet? if not, continue,
                        # if yes, we need vertical space for arrows
                        continue
                    any_connections_up = True
            if any_connections_up:
                cheight += self.config["vspace"]  # for arrows
            else:  # give a bit of room:
                # FIXME: determine if there were spaces drawn last layer
                # Right now, just skip any space at all
                # cheight += 5
                pass
            row_height = 0  # for row of images
            # Draw each column:
            for column in range(len(level_tups)):
                (layer_name, anchor, fname) = level_tups[column]
                if not self._get_visible(layer_name):
                    if not hiding.get(
                        column, False
                    ):  # not already hiding, add some space:
                        struct.append(
                            [
                                "label_svg",
                                {
                                    "x": cwidth + spacing - 80,  # center the text
                                    "y": cheight + 15,
                                    "label": "[layer(s) not visible]",
                                    "font_size": self.config["font_size"],
                                    "font_color": self.config["font_color_not_visible"],
                                    "font_family": self.config["font_family"],
                                    "text_anchor": "start",
                                    "rotate": False,
                                },
                            ]
                        )
                        row_height = max(row_height, self.config["vspace"])
                    hiding[column] = True
                    cwidth += spacing  # leave full column width
                    continue
                # end run of hiding
                hiding[column] = False
                # Anchor
                if anchor:
                    anchor_name = "%s-%s-anchor%s" % (layer_name, fname, level_num)
                    cwidth += spacing
                    positioning[anchor_name] = {
                        "x": cwidth,
                        "y": cheight + row_heights[row],
                    }
                    x1 = cwidth
                    # now we are at an anchor. Is the thing that it anchors in the
                    # lower row? level_num is increasing
                    prev = [
                        (oname, oanchor, lfname)
                        for (oname, oanchor, lfname) in ordering[level_num - 1]
                        if (
                            ((layer_name == oname) and (oanchor is False))
                            or (
                                (layer_name == oname)
                                and (oanchor is True)
                                and (fname == lfname)
                            )
                        )
                    ]
                    if prev:
                        tooltip = html.escape(
                            self.describe_connection_to(self[fname], self[layer_name])
                        )
                        if prev[0][1]:  # anchor
                            anchor_name2 = "%s-%s-anchor%s" % (
                                layer_name,
                                fname,
                                level_num - 1,
                            )
                            # draw a line to this anchor point
                            x2 = positioning[anchor_name2]["x"]
                            y2 = positioning[anchor_name2]["y"]
                            struct.append(
                                [
                                    "curve",
                                    {
                                        "endpoint": False,
                                        "drawn": False,
                                        "name": anchor_name2,
                                        "x1": cwidth,
                                        "y1": cheight,
                                        "x2": x2,
                                        "y2": y2,
                                        "arrow_color": self.config["arrow_color"],
                                        "tooltip": tooltip,
                                    },
                                ]
                            )
                            struct.append(
                                [
                                    "curve",
                                    {
                                        "endpoint": False,
                                        "drawn": False,
                                        "name": anchor_name2,
                                        "x1": cwidth,
                                        "y1": cheight + row_heights[row],
                                        "x2": cwidth,
                                        "y2": cheight,
                                        "arrow_color": self.config["arrow_color"],
                                        "tooltip": tooltip,
                                    },
                                ]
                            )
                        else:
                            # draw a line to this bank
                            x2 = (
                                positioning[layer_name]["x"]
                                + positioning[layer_name]["width"] / 2
                            )
                            y2 = (
                                positioning[layer_name]["y"]
                                + positioning[layer_name]["height"]
                            )
                            tooltip = "TODO"
                            struct.append(
                                [
                                    "curve",
                                    {
                                        "endpoint": True,
                                        "drawn": False,
                                        "name": layer_name,
                                        "x1": cwidth,
                                        "y1": cheight,
                                        "x2": x2,
                                        "y2": y2,
                                        "arrow_color": self.config["arrow_color"],
                                        "tooltip": tooltip,
                                    },
                                ]
                            )
                            struct.append(
                                [
                                    "curve",
                                    {
                                        "endpoint": False,
                                        "drawn": False,
                                        "name": layer_name,
                                        "x1": cwidth,
                                        "y1": cheight + row_heights[row],
                                        "x2": cwidth,
                                        "y2": cheight,
                                        "arrow_color": self.config["arrow_color"],
                                        "tooltip": tooltip,
                                    },
                                ]
                            )
                    else:
                        print("that's weird!", layer_name, "is not in", prev)
                    continue
                else:
                    # Bank positioning
                    image = images[layer_name]
                    (width, height) = image_dims[layer_name]
                    cwidth += spacing - (width / 2)
                    positioning[layer_name] = {
                        "name": layer_name
                        + ("-rotated" if self.config["rotate"] else ""),
                        "x": cwidth,
                        "y": cheight,
                        "image": image_to_uri(image),
                        "width": width,
                        "height": height,
                        "tooltip": self._get_tooltip(layer_name),
                        "border_color": self._get_border_color(layer_name),
                        "border_width": self._get_border_width(layer_name),
                        "rx": cwidth - 1,  # based on arrow width
                        "ry": cheight - 1,
                        "rh": height + 2,
                        "rw": width + 2,
                    }
                    x1 = cwidth + width / 2
                y1 = cheight - 1
                # Arrows going up
                for out in self.outgoing_layers(layer_name):
                    if out.name not in positioning:
                        continue
                    # draw an arrow between layers:
                    anchor_name = "%s-%s-anchor%s" % (
                        out.name,
                        layer_name,
                        level_num - 1,
                    )
                    # Don't draw this error, if there is an anchor in the next level
                    if anchor_name in positioning:
                        tooltip = html.escape(
                            self.describe_connection_to(self[layer_name], out)
                        )
                        x2 = positioning[anchor_name]["x"]
                        y2 = positioning[anchor_name]["y"]
                        struct.append(
                            [
                                "curve",
                                {
                                    "endpoint": False,
                                    "drawn": False,
                                    "name": anchor_name,
                                    "x1": x1,
                                    "y1": y1,
                                    "x2": x2,
                                    "y2": y2,
                                    "arrow_color": self.config["arrow_color"],
                                    "tooltip": tooltip,
                                },
                            ]
                        )
                        continue
                    else:
                        tooltip = html.escape(
                            self.describe_connection_to(self[layer_name], out)
                        )
                        x2 = (
                            positioning[out.name]["x"]
                            + positioning[out.name]["width"] / 2
                        )
                        y2 = (
                            positioning[out.name]["y"] + positioning[out.name]["height"]
                        )
                        struct.append(
                            [
                                "curve",
                                {
                                    "endpoint": True,
                                    "drawn": False,
                                    "name": out.name,
                                    "x1": x1,
                                    "y1": y1,
                                    "x2": x2,
                                    "y2": y2 + 2,
                                    "arrow_color": self.config["arrow_color"],
                                    "tooltip": tooltip,
                                },
                            ]
                        )
                # Bank images
                struct.append(["image_svg", positioning[layer_name]])
                struct.append(
                    [
                        "label_svg",
                        {
                            "x": positioning[layer_name]["x"]
                            + positioning[layer_name]["width"]
                            + 5,
                            "y": positioning[layer_name]["y"]
                            + positioning[layer_name]["height"] / 2
                            + 2,
                            "label": layer_name,
                            "font_size": self.config["font_size"],
                            "font_color": self.config["font_color_layer_name"],
                            "font_family": self.config["font_family"],
                            "text_anchor": "start",
                        },
                    ]
                )
                output_shape = self._get_output_shape(layer_name)
                # FIXME: how to determine a layer that has images as input?
                if (
                    isinstance(output_shape, tuple)
                    and len(output_shape) == 4
                    and self[layer_name].__class__.__name__ != "ImageLayer"
                ):
                    features = str(output_shape[3])
                    # FIXME:
                    feature = str(self._get_feature(layer_name))
                    if self.config["rotate"]:
                        struct.append(
                            [
                                "label_svg",
                                {
                                    "x": positioning[layer_name]["x"] + 5,
                                    "y": positioning[layer_name]["y"] - 10 - 5,
                                    "label": features,
                                    "font_size": self.config["font_size"],
                                    "font_color": self.config["font_color"],
                                    "font_family": self.config["font_family"],
                                    "text_anchor": "start",
                                },
                            ]
                        )
                        struct.append(
                            [
                                "label_svg",
                                {
                                    "x": positioning[layer_name]["x"]
                                    + positioning[layer_name]["width"]
                                    - 10,
                                    "y": positioning[layer_name]["y"]
                                    + positioning[layer_name]["height"]
                                    + 10
                                    + 5,
                                    "label": feature,
                                    "font_size": self.config["font_size"],
                                    "font_color": self.config["font_color"],
                                    "font_family": self.config["font_family"],
                                    "text_anchor": "start",
                                },
                            ]
                        )
                    else:
                        struct.append(
                            [
                                "label_svg",
                                {
                                    "x": positioning[layer_name]["x"]
                                    + positioning[layer_name]["width"]
                                    + 5,
                                    "y": positioning[layer_name]["y"] + 5,
                                    "label": features,
                                    "font_size": self.config["font_size"],
                                    "font_color": self.config["font_color"],
                                    "font_family": self.config["font_family"],
                                    "text_anchor": "start",
                                },
                            ]
                        )
                        struct.append(
                            [
                                "label_svg",
                                {
                                    "x": positioning[layer_name]["x"]
                                    - (len(feature) * 7)
                                    - 5
                                    - 5,
                                    "y": positioning[layer_name]["y"]
                                    + positioning[layer_name]["height"]
                                    - 5,
                                    "label": feature,
                                    "font_size": self.config["font_size"],
                                    "font_color": self.config["font_color"],
                                    "font_family": self.config["font_family"],
                                    "text_anchor": "start",
                                },
                            ]
                        )
                if False:  # (self[layer_name].dropout > 0): FIXME:
                    struct.append(
                        [
                            "label_svg",
                            {
                                "x": positioning[layer_name]["x"]
                                - 1 * 2.0
                                - 18,  # length of chars * 2.0
                                "y": positioning[layer_name]["y"] + 4,
                                "label": "o",  # "&#10683;"
                                "font_size": self.config["font_size"] * 2.0,
                                "font_color": self.config["font_color"],
                                "font_family": self.config["font_family"],
                                "text_anchor": "start",
                            },
                        ]
                    )
                    struct.append(
                        [
                            "label_svg",
                            {
                                "x": positioning[layer_name]["x"]
                                - 1 * 2.0
                                - 15
                                + (
                                    -3 if self.config["rotate"] else 0
                                ),  # length of chars * 2.0
                                "y": positioning[layer_name]["y"]
                                + 5
                                + (-1 if self.config["rotate"] else 0),
                                "label": "x",  # "&#10683;"
                                "font_size": self.config["font_size"] * 1.3,
                                "font_color": self.config["font_color"],
                                "font_family": self.config["font_family"],
                                "text_anchor": "start",
                            },
                        ]
                    )
                cwidth += width / 2
                row_height = max(row_height, height)
            cheight += row_height
            level_num += 1
        cheight += self.config["border_bottom"]
        # DONE!
        # Draw the title:
        if mode == "activation":
            title = "Activations for %s"  % self.config["name"]
        elif mode == "pca":
            title = "PCAs for %s" % self.config["name"]
        elif mode == "histogram":
            title = "Histograms for %s" % self.config["name"]
        if self.config["rotate"]:
            struct.append(
                [
                    "label_svg",
                    {
                        "x": 10,  # really border_left
                        "y": cheight / 2,
                        "label": title,
                        "font_size": self.config["font_size"] + 3,
                        "font_color": self.config["font_color"],
                        "font_family": self.config["font_family"],
                        "text_anchor": "middle",
                    },
                ]
            )
        else:
            struct.append(
                [
                    "label_svg",
                    {
                        "x": max_width / 2,
                        "y": self.config["border_top"] / 2,
                        "label": title,
                        "font_size": self.config["font_size"] + 3,
                        "font_color": self.config["font_color"],
                        "font_family": self.config["font_family"],
                        "text_anchor": "middle",
                    },
                ]
            )
        # figure out scale optimal, if scale is None
        # the fraction:
        if self.config["scale"] is not None:  # scale is given:
            if self.config["rotate"]:
                scale_value = (self.config["max_width"] / cheight) * self.config[
                    "scale"
                ]
            else:
                scale_value = (self.config["max_width"] / max_width) * self.config[
                    "scale"
                ]
        else:
            if self.config["rotate"]:
                scale_value = self.config["max_width"] / max(cheight, max_width)
            else:
                scale_value = self.config["preferred_size"] / max(cheight, max_width)
        # svg_scale = "%s%%" % int(scale_value * 100)
        scaled_width = max_width * scale_value
        scaled_height = cheight * scale_value
        # Need a top-level width, height because Jupyter peeks at it
        if self.config["rotate"]:
            svg_transform = (
                """ transform="rotate(90) translate(0 -%s)" """ % scaled_height
            )
            # Swap them:
            top_width = scaled_height
            top_height = scaled_width
        else:
            svg_transform = ""
            top_width = scaled_width
            top_height = scaled_height
        struct.append(
            [
                "head_svg",
                {
                    "viewbox_width": int(max_width),  # view port width
                    "viewbox_height": int(cheight),  # view port height
                    "width": int(scaled_width),  # actual pixels of image in page
                    "height": int(scaled_height),  # actual pixels of image in page
                    "id": self.config["id"],
                    "top_width": int(top_width),
                    "top_height": int(top_height),
                    "arrow_color": self.config["arrow_color"],
                    "arrow_width": self.config["arrow_width"],
                    "transform": svg_transform,
                },
            ]
        )
        return struct

    def incoming_layers(self, layer_name):
        layer = self[layer_name]
        layers = []
        for node in layer.inbound_nodes:
            if hasattr(node.inbound_layers, "__iter__"):
                for layer in node.inbound_layers:
                    if layer not in layers:
                        layers.append(layer)
            else:
                if node.inbound_layers not in layers:
                    layers.append(node.inbound_layers)
        return layers

    def outgoing_layers(self, layer_name):
        layer = self[layer_name]
        layers = []
        for node in layer.outbound_nodes:
            if node.outbound_layer not in layers:
                layers.append(node.outbound_layer)
        return layers

    def _get_layer_type(self, layer_name):
        """
        Determines whether a layer is a "input", "hidden", or "output"
        layer based on its connections. If no connections, then it is
        "unconnected".
        """
        incoming_connections = self.incoming_layers(layer_name)
        outgoing_connections = self.outgoing_layers(layer_name)
        if len(incoming_connections) == 0 and len(outgoing_connections) == 0:
            return "unconnected"
        elif len(incoming_connections) > 0 and len(outgoing_connections) > 0:
            return "hidden"
        elif len(incoming_connections) > 0:
            return "output"
        else:
            return "input"

    def _get_layer_class(self, layer_name):
        """
        """
        layer = self[layer_name]
        return layer.__class__.__name__

    def _get_level_ordering(self):
        """
        Returns a list of lists of tuples from
        input to output of levels.

        Each tuple contains: (layer_name, anchor?, from_name/None)

        If anchor is True, this is just an anchor point.
        """
        # First, get a level for all layers:
        levels = {}
        for layer in self._layers:
            level = max(
                [levels[lay.name] for lay in self.incoming_layers(layer.name)] + [-1]
            )
            levels[layer.name] = level + 1
        max_level = max(levels.values())
        ordering = []
        for i in range(max_level + 1):  # input to output
            layer_names = [
                layer.name for layer in self._layers if levels[layer.name] == i
            ]
            ordering.append(
                [
                    (name, False, [x.name for x in self.incoming_layers(name)])
                    for name in layer_names
                ]
            )  # (going_to/layer_name, anchor, coming_from)
        # promote all output banks to last row:
        for level in range(len(ordering)):  # input to output
            tuples = ordering[level]
            index = 0
            for (name, anchor, none) in tuples[:]:
                if self._get_layer_type(name) == "output":
                    # move it to last row
                    # find it and remove
                    ordering[-1].append(tuples.pop(index))
                else:
                    index += 1
        # insert anchor points for any in next level
        # that doesn't go to a bank in this level
        # order_cache = {}
        for level in range(len(ordering)):  # input to output
            tuples = ordering[level]
            for (name, anchor, fname) in tuples:
                if anchor:
                    # is this in next? if not add it
                    next_level = [
                        (n, anchor) for (n, anchor, hfname) in ordering[level + 1]
                    ]
                    if (
                        name,
                        False,
                    ) not in next_level:  # actual layer not in next level
                        ordering[level + 1].append(
                            (name, True, fname)
                        )  # add anchor point
                    else:
                        pass  # finally!
                else:
                    # if next level doesn't contain an outgoing
                    # connection, add it to next level as anchor point
                    for layer in self.outgoing_layers(name):
                        next_level = [
                            (n, anchor) for (n, anchor, fname) in ordering[level + 1]
                        ]
                        if (layer.name, False) not in next_level:
                            ordering[level + 1].append(
                                (layer.name, True, name)
                            )  # add anchor point
        ordering = self._optimize_ordering(ordering)
        return ordering

    def _optimize_ordering(self, ordering):
        def perms(l):
            return list(itertools.permutations(l))

        def distance(xy1, xy2):
            return math.sqrt((xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2)

        def find_start(cend, canchor, name, plevel):
            """
            Return position and weight of link from cend/name to
            col in previous level.
            """
            col = 1
            for bank in plevel:
                pend, panchor, pstart_names = bank
                if name == pend:
                    if not panchor and not canchor:
                        weight = 10.0
                    else:
                        weight = 1.0
                    return col, weight
                elif cend == pend and name == pstart_names:
                    return col, 5.0
                col += 1
            raise Exception("connecting layer not found!")

        # First level needs to be in bank_order, and cannot permutate:
        first_level = [(bank_name, False, []) for bank_name in self.input_bank_order]
        perm_count = functools.reduce(
            operator.mul, [math.factorial(len(level)) for level in ordering[1:]]
        )
        if perm_count < 70000:  # globally minimize
            permutations = itertools.product(*[perms(x) for x in ordering[1:]])
            # measure arrow distances for them all and find the shortest:
            best = (10000000, None)
            for ordering in permutations:
                ordering = (first_level,) + ordering
                sum = 0.0
                for level_num in range(1, len(ordering)):
                    level = ordering[level_num]
                    plevel = ordering[level_num - 1]
                    col1 = 1
                    for bank in level:  # starts at level 1
                        cend, canchor, cstart_names = bank
                        if canchor:
                            cstart_names = [cstart_names]  # put in list
                        for name in cstart_names:
                            col2, weight = find_start(cend, canchor, name, plevel)
                            dist = (
                                distance(
                                    (col1 / (len(level) + 1), 0),
                                    (col2 / (len(plevel) + 1), 0.1),
                                )
                                * weight
                            )
                            sum += dist
                        col1 += 1
                if sum < best[0]:
                    best = (sum, ordering)
            return best[1]
        else:  # locally minimize, between layers:
            ordering[0] = first_level  # replace first level with sorted
            for level_num in range(1, len(ordering)):
                best = (10000000, None, None)
                plevel = ordering[level_num - 1]
                for level in itertools.permutations(ordering[level_num]):
                    sum = 0.0
                    col1 = 1
                    for bank in level:  # starts at level 1
                        cend, canchor, cstart_names = bank
                        if canchor:
                            cstart_names = [cstart_names]  # put in list
                        for name in cstart_names:
                            col2, weight = find_start(cend, canchor, name, plevel)
                            dist = (
                                distance(
                                    (col1 / (len(level) + 1), 0),
                                    (col2 / (len(plevel) + 1), 0.1),
                                )
                                * weight
                            )
                            sum += dist
                        col1 += 1
                    if sum < best[0]:
                        best = (sum, level)
                ordering[level_num] = best[1]
            return ordering

    def _pre_process_struct(self, inputs, ordering, targets, mode, colors, sizes):
        """
        Determine sizes and pre-compute images.
        """
        # find max_width, image_dims, and row_height
        # Go through and build images, compute max_width:
        row_heights = []
        max_width = 0
        max_height = 0
        images = {}
        image_dims = {}
        # if targets, then need to propagate for error:
        if targets is not None:
            outputs = self._model(inputs, training=False).numpy()
            if len(self.output_bank_order) == 1:
                targets = [targets]
                errors = (np.array(outputs) - np.array(targets)).tolist()
            else:
                errors = []
                for bank in range(len(self.output_bank_order)):
                    errors.append(
                        (np.array(outputs[bank]) - np.array(targets[bank])).tolist()
                    )
        # For each level:
        hiding = {}
        for level_tups in ordering:  # output to input:
            # first make all images at this level
            row_width = 0  # for this row
            row_height = 0  # for this row
            # For each column:
            for column in range(len(level_tups)):
                (layer_name, anchor, fname) = level_tups[column]
                if not self._get_visible(layer_name):
                    if not hiding.get(column, False):
                        row_height = max(
                            row_height, self.config["vspace"]
                        )  # space for hidden indicator
                    hiding[column] = True  # in the middle of hiding some layers
                    row_width += self.config["hspace"]  # space between
                    max_width = max(max_width, row_width)  # of all rows
                    continue
                elif anchor:
                    # No need to handle anchors here
                    # as they occupy no vertical space
                    hiding[column] = False
                    # give it some hspace for this column
                    # in case there is nothing else in this column
                    row_width += self.config["hspace"]
                    max_width = max(max_width, row_width)
                    continue
                hiding[column] = False
                # The rest of this for loop is handling image of bank
                if inputs is not None:
                    v = inputs
                # elif len(self.dataset.inputs) > 0 and not isinstance(self.dataset, VirtualDataset):
                #    # don't change cache if virtual... could take some time to rebuild cache
                #    v = self.dataset.inputs[0]
                # else:
                # if False:  # FIXME not compiled
                #    if len(self.input_bank_order) > 1:
                #        v = []
                #        for in_name in self.input_bank_order:
                #            v.append(self.make_dummy_vector(in_name))
                #    else:
                #        in_layer_name = self.input_bank_order[0]
                #        v = self.make_dummy_vector(in_layer_name)
                keep_aspect_ratio = None
                if mode == "pca":
                    image = self.predict_pca_to(v, layer_name, colors, sizes)
                    keep_aspect_ratio = True
                elif mode == "histogram":
                    image = self.predict_histogram_to(v, layer_name)
                    keep_aspect_ratio = True
                else: # activations
                    try:
                        # FIXME get one per output bank:
                        image = self.make_image(
                            layer_name, self.predict_to(v, layer_name)[0]
                        )
                    except Exception:
                        image = self.make_image(
                            layer_name, np.array(self.make_dummy_vector(layer_name)),
                        )
                (width, height) = image.size
                images[layer_name] = image  # little image
                if self._get_layer_type(layer_name) == "output":
                    if targets is not None:
                        # Target image, targets set above:
                        target_colormap = ("grey", -2, 2)  # FIXME: self[layer_name].colormap
                        target_bank = targets[self.output_bank_order.index(layer_name)]
                        target_array = np.array(target_bank)
                        target_image = self.make_image(
                            layer_name, target_array, target_colormap
                        )
                        # Error image, error set above:
                        error_colormap = (get_error_colormap(), -2, 2) # FIXME
                        error_bank = errors[self.output_bank_order.index(layer_name)]
                        error_array = np.array(error_bank)
                        error_image = self.make_image(
                            layer_name, error_array, error_colormap
                        )
                        images[layer_name + "_errors"] = error_image
                        images[layer_name + "_targets"] = target_image
                    else:
                        images[layer_name + "_errors"] = image
                        images[layer_name + "_targets"] = image
                # Layer settings:
                # FIXME:
                # if self[layer_name].image_maxdim:
                #    image_maxdim = self[layer_name].image_maxdim
                # else:
                image_maxdim = self.config["image_maxdim"]
                # FIXME:
                # if self[layer_name].image_pixels_per_unit:
                #    image_pixels_per_unit = self[layer_name].image_pixels_per_unit
                # else:
                image_pixels_per_unit = self.config["image_pixels_per_unit"]
                # First, try based on shape:
                # pwidth, pheight = np.array(image.size) * image_pixels_per_unit
                vshape = self.vshape(layer_name)
                keep_aspect_ratio = (keep_aspect_ratio if keep_aspect_ratio is not None
                                     else self._get_keep_aspect_ratio(layer_name))
                if vshape is None or keep_aspect_ratio:
                    pass  # let the image set the shape
                elif len(vshape) == 1:
                    if vshape[0] is not None:
                        width = vshape[0] * image_pixels_per_unit
                        height = image_pixels_per_unit
                elif len(vshape) >= 2:
                    if vshape[0] is not None:
                        height = vshape[0] * image_pixels_per_unit
                        if vshape[1] is not None:
                            width = vshape[1] * image_pixels_per_unit
                    else:
                        if len(vshape) > 2:
                            if vshape[1] is not None:
                                height = vshape[1] * image_pixels_per_unit
                                width = vshape[2] * image_pixels_per_unit
                        elif vshape[1] is not None:  # flatten
                            width = vshape[1] * image_pixels_per_unit
                            height = image_pixels_per_unit
                # keep aspect ratio:
                if keep_aspect_ratio:
                    scale = image_maxdim / max(width, height)
                    image = image.resize((int(width * scale), int(height * scale)))
                    width, height = image.size
                else:
                    # Change aspect ratio if too big/small
                    if width < image_pixels_per_unit:
                        width = image_maxdim
                    if height < image_pixels_per_unit:
                        height = image_pixels_per_unit
                    # make sure not too big:
                    if height > image_maxdim:
                        height = image_maxdim
                    if width > image_maxdim:
                        width = image_maxdim
                image_dims[layer_name] = (width, height)
                row_width += width + self.config["hspace"]  # space between
                row_height = max(row_height, height)
            row_heights.append(row_height)
            max_width = max(max_width, row_width)  # of all rows
        return max_width, max_height, row_heights, images, image_dims

    def set_config(self, **items):
        """
        Set one or more configurable item:
        """
        for item in items:
            if item in self.config:
                if item == "layers":
                    self.set_config_layers(**self.config["layers"])
                else:
                    self.config[item] = items[item]
            else:
                raise AttributeError("no such config item: %r" % item)

    def set_config_layers_by_class(self, class_name, **items):
        """
        Set one or more configurable items in layers
        identified by layer instance type.

        Examples:
        ```python
        >>> net.set_config_layers_by_class("Conv", colormap=("RdGy", -1, 1))
        ```
        """
        for layer in self._layers:
            if layer.__class__.__name__.lower().startswith(class_name.lower()):
                self.set_config_layer(layer.name, **items)

    def set_config_layers_by_name(self, name, **items):
        """
        Set one or more configurable items in layers
        identified by layer instance name.

        Examples:
        ```python
        >>> net.set_config_layers_by_name("input", colormap=("RdGy", -1, 1))
        ```
        """
        for layer in self._layers:
            if layer.name.lower().startswith(name.lower()):
                self.set_config_layer(layer.name, **items)

    def set_config_layers(self, **items):
        """
        Set one or more configurable items in all layers:

        >>> net.set_config_layers(colormap=("plasma", -1, 0))

        """
        for layer in self._layers:
            self.set_config_layer(layer.name, **items)

    def set_config_layer(self, layer_name, **items):
        """
        Set one or more configurable items in a layer:
        """
        proper_items = {
            "vshape": ("integer", "integer"),
            "feature": "integer",
            "keep_aspect_ratio": "boolean",
            "visible": "boolean",
            "colormap": ("string", "number", "number"),
            "border_color": "string",
            "border_width": "integer",
        }
        def validate_type(value, format):
            if format == "integer":
                return isinstance(value, int)
            elif format == "number":
                return isinstance(value, (int, float))
            elif format == "string":
                return isinstance(value, str)
            elif format == "boolean":
                return isinstance(value, bool)
            else:
                return all([validate_type(v,f) for v,f in zip(value, format)])

        if layer_name in self.config["layers"]:
            for item in items:
                if item in proper_items:
                    if validate_type(items[item], proper_items[item]):
                        self.config["layers"][layer_name][item] = items[item]
                    else:
                        raise AttributeError("invalid form for: %r; should be: %s" % (item, proper_items[item]))
                else:
                    raise AttributeError("no such config layer item: %r" % item)
        else:
            raise AttributeError("no such config layer: %r" % layer_name)

    def get_weights_from_history(self):
        """
        Get the weights saved in history.
        """
        return self._history["weights"]

    def get_weights(self, flat=False):
        """
        Return the weights of the entire network.

        Args:
            flat: (bool) if True, return the weights
                as a single dimensional array.
        """
        if flat:
            array = []
            for layer in self._model.layers:
                for weight in layer.get_weights():
                    array.extend(weight.flatten())
            return array
        else:
            return self._model.get_weights()

    def set_weights(self, weights):
        """
        Set the weights in a network.

        Args:
            weights: a list of pairs of weights and biases for each layer,
                or a single (flat) array of values
        """
        if len(weights) > 0 and isinstance(weights[0], numbers.Number):
            current = 0
            for layer in self._model.layers:
                orig = layer.get_weights()
                new_weights = []
                for item in orig:
                    total = functools.reduce(operator.mul, item.shape, 1)
                    w = np.array(weights[current:current + total])
                    new_weights.append(w.reshape(item.shape))
                    current += total
                layer.set_weights(new_weights)
        else:
            self._model.set_weights(weights)

    def set_learning_rate(self, learning_rate):
        """
        Sometimes called `epsilon`.
        """
        if hasattr(self._model, "optimizer") and hasattr(self._model.optimizer, "lr"):
            self._model.optimizer.lr = learning_rate
        else:
            print("WARNING: you need to use an optimizer with lr")

    def get_learning_rate(self):
        """
        Sometimes called `epsilon`.
        """
        if hasattr(self._model, "optimizer") and hasattr(self._model.optimizer, "lr"):
            return self._model.optimizer.lr.numpy()
        else:
            print("WARNING: you need to use an optimizer with lr")

    def get_metric(self, name):
        import tensorflow.keras.backend as K

        if name == "tolerance_accuracy":
            self._state["tolerance_accuracy_used"] = True
            def tolerance_accuracy(targets, outputs):
                return K.mean(
                    K.all(
                        K.less_equal(K.abs(targets - outputs),
                                     self._tolerance), axis=-1),
                    axis=-1)
            return tolerance_accuracy
        else:
            return name

    def get_momentum(self):
        """
        """
        if hasattr(self._model, "optimizer") and hasattr(
            self._model.optimizer, "momentum"
        ):
            return self._model.optimizer.momentum.numpy()
        else:
            print("WARNING: you need to use an optimizer with momentum")

    def set_momentum(self, momentum):
        """
        """
        if hasattr(self._model, "optimizer") and hasattr(
            self._model.optimizer, "momentum"
        ):
            self._model.optimizer.momentum = momentum
        else:
            print("WARNING: you need to use an optimizer with momentum")

    def get_tolerance(self):
        """
        """
        if not self._state["tolerance_accuracy_used"]:
            print("WARNING: you need Network.compile(metrics=['tolerance_accuracy']) to use tolerance")
        return self._tolerance

    def set_tolerance(self, tolerance):
        """
        """
        if not self._state["tolerance_accuracy_used"]:
            print("WARNING: you need Network.compile(metrics=['tolerance_accuracy']) to use tolerance")
        self._tolerance = tolerance


class SimpleNetwork(Network):
    def __init__(
        self,
        *layers,
        name="SimpleNetwork",
        activation="sigmoid",
        loss="mse",
        optimizer="sgd",
        metrics=None,
    ):
        """
        Create a simple, stochasitic gradient descent network of Dense
        layers.

        Args:
            layers (list): a list of layer forms (see below)
            name (str): a name for the network
            activation (str): the name of a default activation function
                if one is not given with layer.
            loss (str): the name of a metric to use to compute loss (eg, error)
            optimizer (str or Optimizer): the name of an optimizer or an actual
                 Optimizer instance
            metrics (list): a list of metrics to compute in addition to
                "accuracy" and "tolerance_accuracy".

        Layers may be given in the following formats:
            * int: the number of units in the layer
            * (int, str): number of units, and activation function (non-input layers
                only)
            * (int, int, ...): (input layers only) the shape of the input patterns
            * keras layer instance: an instance of a keras layer, like Flatten()
        """
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Dense, Input, Layer
        import tensorflow.keras.layers

        def make_name(index, total):
            if index == 0:
                return "input"
            elif index == total - 1:
                return "output"
            elif index == 1 and total == 3:
                return "hidden"
            else:
                return "hidden_%d" % index

        def make_layer(index, layers, activation):
            if isinstance(layers[index], Layer):
                return layers[index]
            elif (isinstance(layers[index], str) and
                  hasattr(tensorflow.keras.layers, layers[index])):
                layer_class = getattr(tensorflow.keras.layers, layers[index])
                return layer_class()
            else:
                name = make_name(index, len(layers))
                if index == 0:
                    size = layers[index]
                    return Input(size, name=name)
                else:
                    size = layers[index]
                    if isinstance(size, int):
                        activation_function = activation
                    elif len(size) == 2 and isinstance(size[1], str):
                        size, activation_function = size
                    else:
                        raise Exception("Invalid SimpleNetwork layer representation: %r" % size)
                    return Dense(size, activation=activation_function, name=name)

        layers = [
            make_layer(index, layers, activation)
            for index in range(len(layers))
        ]
        current_layer = layers[0]
        for layer in layers[1:]:
            current_layer = layer(current_layer)
        model = Model(inputs=layers[0], outputs=current_layer, name=name)
        if metrics is None:
            metrics = ["tolerance_accuracy"]
        # Replaced special named metrics with ours:
        super()._init_state()
        metrics = [self.get_metric(name) for name in metrics]
        model.compile(optimizer=self._make_optimizer(optimizer), loss=loss, metrics=metrics)
        super().__init__(model)

    def _make_optimizer(self, optimizer):
        import tensorflow as tf

        # Get optimizer with some defaults
        if optimizer == "sgd":
            return tf.keras.optimizers.SGD(
                learning_rate=0.1, momentum=0.9, nesterov=False, name="SGD"
            )
        else:
            return optimizer
