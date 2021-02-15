# -*- coding: utf-8 -*-
# ******************************************************
# aitk.networks: Keras model wrapper with visualizations
#
# Copyright (c) 2021 Douglas S. Blank
#
# https://github.com/ArtificialIntelligenceToolkit/aitk.networks
#
# ******************************************************

import pytest
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import RMSprop

from aitk.networks import Network


def build_model(model_name, input_shape):
    if model_name == "sequential":
        return build_sequential_model(input_shape)
    elif model_name == "model1":
        return build_model1(input_shape)


def make_shape(count, input_shape):
    if isinstance(input_shape, int):
        input_shape = tuple(input_shape)
    return tuple([count] + list(input_shape))


def build_dataset(input_shape):
    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(make_shape(60000, input_shape))
    x_test = x_test.reshape(make_shape(10000, input_shape))
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    x_train /= 255
    x_test /= 255

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    return x_train, y_train, x_test, y_test


def build_sequential_model(input_shape):
    model = Sequential()
    model.add(Dense(128, activation="sigmoid", input_shape=input_shape))
    model.add(Dense(128, activation="sigmoid"))
    model.add(Dense(128, activation="sigmoid"))
    model.add(Dense(10, activation="softmax"))
    model.compile(
        loss="categorical_crossentropy", optimizer=RMSprop(), metrics=["accuracy"]
    )
    return model


def build_model1(input_shape):
    l0 = Input(input_shape)
    l1 = Dense(128, activation="sigmoid")(l0)
    l2 = Dense(128, activation="sigmoid")(l1)
    l3 = Dense(128, activation="sigmoid")(l2)
    l4 = Dense(10, activation="softmax")(l3)
    model = Model(inputs=[l0], outputs=[l4])
    model.compile(
        loss="categorical_crossentropy", optimizer=RMSprop(), metrics=["accuracy"]
    )
    return model


@pytest.mark.parametrize("model_name", ["sequential", "model1"])
def test_layer_type(model_name):
    input_shape = (784,)
    model = build_model(model_name, input_shape)
    network = Network(model)

    assert ["input", "hidden", "hidden", "hidden", "output"] == [
        network._get_layer_type(layer.name) for layer in network._layers
    ]


@pytest.mark.parametrize("model_name", ["sequential", "model1"])
def test_describe(model_name):
    input_shape = (784,)
    model = build_model(model_name, input_shape)
    network = Network(model)

    for i, layer in enumerate(network._layers[:-1]):
        desc = network.describe_connection_to(
            network._layers[i], network._layers[i + 1]
        )
        assert ("Weights from %s" % layer.name) in desc


@pytest.mark.parametrize("model_name", ["sequential", "model1"])
def test_incoming_layers(model_name):
    input_shape = (784,)
    model = build_model(model_name, input_shape)
    network = Network(model)

    for i, layer in enumerate(network._layers):
        if i > 0:
            input_layers = network.incoming_layers(network._layers[i].name)
            assert [network._layers[i - 1]] == input_layers, "i is %r" % i


@pytest.mark.parametrize("model_name", ["sequential", "model1"])
def test_with_data(model_name):
    input_shape = (784,)
    model = build_model(model_name, input_shape)
    network = Network(model)

    x_train, y_train, x_test, y_test = build_dataset(input_shape)

    network.predict(x_test)


"""
'_find_spacing',
 '_get_act_minmax',
 '_get_activation_name',
 '_get_border_color',
 '_get_border_width',
 '_get_colormap',
 '_get_feature',
 '_get_input_layers',
 '_get_keep_aspect_ratio',
 '_get_level_ordering',
 '_get_output_shape',
 '_get_tooltip',
 '_get_visible',
 '_layers',
 '_layers_map',
 '_level_ordering',
 '_model',
 '_optimize_ordering',
 '_pre_process_struct',
 '_svg_counter',
 '_get_feature',
 'build_struct',
 'input_bank_order',
 'make_dummy_vector',
 'make_image',
 'max_draw_units',
 'minmax',
 'num_input_layers',
 'outgoing_layers',
 'to_svg',
 'vshape']
"""
