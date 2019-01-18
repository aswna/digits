#!/usr/bin/env python3

# This is going to be the Python implementation of
# https://mmlind.github.io/Simple_1-Layer_Neural_Network_for_MNIST_Handwriting_Recognition/

import random
from collections import namedtuple

MNIST_Image = namedtuple(
    "MNIST_Image", [
        "pixel",  # 28x28
    ]
)

MNIST_Label = namedtuple(
    "MNIST_Label", [
        "value",
    ]
)

MNIST_ImageFileHeader = namedtuple(
    "MNIST_ImageFileHeader", [
        "magicNumber",
        "maxImages",
        "imgWidth",
        "imgHeight",
    ]
)

MNIST_LabelFileHeader = namedtuple(
    "MNIST_LabelFileHeader", [
        "magicNumber",
        "maxLabels",
    ]
)

Cell = namedtuple(
    "Cell", [
        "input",  # 28x28
        "weight",  # 28x28
        "output",
    ]
)

Layer = namedtuple(
    "Layer", [
        "cell",  # cells (10)
    ]
)

Vector = namedtuple(
    "Vector", [
        "val",  # vals (10)
    ]
)


def get_init_input():
    return [0 for x in range(28*28)]


def get_init_weight():
    return [random.uniform(0, 1) for x in range(28*28)]


def get_init_cell():
    return Cell(get_init_input(), get_init_weight(), 0)


def initLayer():  # get_init_layer
    cells = [get_init_cell() for x in range(10)]
    return Layer(cells)

# See file formats at http://yann.lecun.com/exdb/mnist/
