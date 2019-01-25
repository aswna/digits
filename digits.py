#!/usr/bin/env python3

"""This is going to be the Python implementation of
https://mmlind.github.io/Simple_1-Layer_Neural_Network_for_MNIST_Handwriting_Recognition/

Note: download and extract data files from http://yann.lecun.com/exdb/mnist/
  train-images-idx3-ubyte.gz:  training set images (9912422 bytes)
  train-labels-idx1-ubyte.gz:  training set labels (28881 bytes)
  t10k-images-idx3-ubyte.gz:   test set images (1648877 bytes)
  t10k-labels-idx1-ubyte.gz:   test set labels (4542 bytes)
"""

import struct
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


def main():
    read_training_file()


def read_training_file():
    """ See file formats at http://yann.lecun.com/exdb/mnist/ """
    filename = "./train-images-idx3-ubyte"
    with open(filename, "rb") as f:
        data = f.read()
        if not data:
            print("No data!")
            exit(1)

    struct_fmt = '>4i'  # int[4]
    struct_len = struct.calcsize(struct_fmt)
    buffer = data[:struct_len]
    s = struct.unpack(struct_fmt, buffer)
    print("%u %d %d %d" % s)
    header = MNIST_ImageFileHeader(*s)

    buffer = data[struct_len:]
    struct_fmt = '>{}B'.format(header.imgHeight * header.imgWidth)
    struct_len = struct.calcsize(struct_fmt)
    images = [MNIST_Image(x) for x in struct.iter_unpack(struct_fmt, buffer)]
    print("{}".format(images[0]))
    for i, pixel in enumerate(images[0].pixel):
        print("%3d " % pixel, end='')
        if i % header.imgWidth == 0:
            print("")


def get_init_input():
    # TODO: use header.imgHeight * header.imgWidth
    return [0 for x in range(28*28)]


def get_init_weight():
    # TODO: use header.imgHeight * header.imgWidth
    return [random.uniform(0, 1) for x in range(28*28)]


def get_init_cell():
    return Cell(get_init_input(), get_init_weight(), 0)


def initLayer():  # get_init_layer
    cells = [get_init_cell() for x in range(10)]
    return Layer(cells)


if __name__ == "__main__":
    main()
