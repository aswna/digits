#!/usr/bin/env python3

"""This is going to be the Python implementation of
https://mmlind.github.io/Simple_1-Layer_Neural_Network_for_MNIST_Handwriting_Recognition/

Note: download and extract data files from http://yann.lecun.com/exdb/mnist/
  train-images-idx3-ubyte.gz:  training set images (9912422 bytes)
  train-labels-idx1-ubyte.gz:  training set labels (28881 bytes)
  t10k-images-idx3-ubyte.gz:   test set images (1648877 bytes)
  t10k-labels-idx1-ubyte.gz:   test set labels (4542 bytes)

See also:
    https://github.com/mmlind/mnist-1lnn/
"""

import struct
import random
from collections import namedtuple


MNIST_Image = namedtuple(
    "MNIST_Image", [
        "pixels",  # 28x28
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


class Cell:
    def __init__(self, input, weight, output):
        self.input = input  # 28x28
        self.weight = weight  # 28x28
        self.size = len(self.input)
        assert self.size == len(self.weight), "Mismatching input - weight lengths"
        self.output = output

    def __str__(self):
        # TODO: this is a dummy implementation
        return "weight = {}".format(self.weight[300])


Layer = namedtuple(
    "Layer", [
        "cells",  # 10
    ]
)

Vector = namedtuple(
    "Vector", [
        "val",  # vals (10)
    ]
)


def main():
    train_images = read_image_file("./train-images-idx3-ubyte")
    train_labels = read_label_file("./train-labels-idx1-ubyte")
    initial_layer = initLayer()
    trained_layer = trainLayer(initial_layer, train_images, train_labels)

    # test_images = read_image_file("./t10k-images-idx3-ubyte")
    # test_labels = read_label_file("./t10k-labels-idx1-ubyte")
    # testLayer(trained_layer, test_images, test_labels)


def trainLayer(layer, images, labels):
    errorCount = 0
    for index, (image, label) in enumerate(zip(images, labels)):
        targetOutput = getTargetOutput(label)
        # print("before = {}".format(layer.cells[0]))
        for cell, target in zip(layer.cells, targetOutput):
            trainCell(cell, image, target)
        predictedNumber = getLayerPrediction(layer)
        if predictedNumber != label.value:
            errorCount += 1
            if not errorCount % 1000:
                print(
                    "Prediction: {}, Actual: {} at image {}".format(
                        predictedNumber,
                        label.value,
                        index,
                    )
                )
        # print("after  = {}".format(layer.cells[0]))



def getTargetOutput(label):
    """Create target vector according to target number."""
    return [1 if x == label else 0 for x in range(10)]


def trainCell(cell, image, target):
    setCellInput(cell, image);
    calcCellOutput(cell);
    error = getCellError(cell, target);
    updateCellWeights(cell, error);


def read_image_file(filename):
    """ See file formats at http://yann.lecun.com/exdb/mnist/ """
    with open(filename, "rb") as f:
        data = f.read()
        if not data:
            print("No data!")
            exit(1)

    struct_fmt = '>4i'  # int[4]
    struct_len = struct.calcsize(struct_fmt)
    buffer = data[:struct_len]
    s = struct.unpack(struct_fmt, buffer)
    print("%d %d %d %d" % s)
    header = MNIST_ImageFileHeader(*s)

    buffer = data[struct_len:]
    struct_fmt = '>{}B'.format(header.imgHeight * header.imgWidth)
    struct_len = struct.calcsize(struct_fmt)
    images = [MNIST_Image(x) for x in struct.iter_unpack(struct_fmt, buffer)]
    # print("first image:\n{}".format(images[0]))
    print("first image:")
    for i, pixel in enumerate(images[0].pixels):
        print("%3d " % pixel, end='')
        if i % header.imgWidth == 0:
            print("")
    print("")
    # print("last image:\n{}".format(images[-1]))
    # for i, pixel in enumerate(images[-1].pixels):
    #     print("%3d " % pixel, end='')
    #     if i % header.imgWidth == 0:
    #         print("")
    return images

def read_label_file(filename):
    """ See file formats at http://yann.lecun.com/exdb/mnist/ """
    with open(filename, "rb") as f:
        data = f.read()
        if not data:
            print("No data!")
            exit(1)

    struct_fmt = '>2i'  # int[2]
    struct_len = struct.calcsize(struct_fmt)
    buffer = data[:struct_len]
    s = struct.unpack(struct_fmt, buffer)
    print("%d %d" % s)
    header = MNIST_LabelFileHeader(*s)

    buffer = data[struct_len:]
    struct_fmt = '>B'
    struct_len = struct.calcsize(struct_fmt)
    labels = [MNIST_Label(x[0]) for x in struct.iter_unpack(struct_fmt, buffer)]
    assert header.maxLabels == len(labels)
    print("first label: {}".format(labels[0]))
    print("last label : {}".format(labels[-1]))
    return labels


def setCellInput(cell, image):
    for i in range(len(image.pixels)):
        cell.input[i] = int(bool(image.pixels[i]))
        # TODO: try?
        # cell.input[i] = 1 if image.pixels[i] > 50 else 0


def calcCellOutput(cell):
    cell.output = 0
    size = len(cell.input)  # TODO: check size
    for i in range(size):
        cell.output += cell.input[i] * cell.weight[i]
    cell.output /= size  # normalize output [0, 1]


def getCellError(cell, target):  # target: 0 or 1
    return target - cell.output


def updateCellWeights(cell, error):
    LEARNING_RATE = 0.05
    for i in range(cell.size):
        cell.weight[i] += LEARNING_RATE * cell.input[i] * error


def getLayerPrediction(layer):
    maxOut = 0
    maxInd = 0
    for i in range(len(layer.cells)):
        if layer.cells[i].output > maxOut:
            maxOut = layer.cells[i].output
            maxInd = i
    return maxInd


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
