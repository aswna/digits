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


LEARNING_RATE = 0.05


MNISTImage = namedtuple(
    "MNISTImage", [
        "pixels",  # 28x28
    ]
)

MNISTLabel = namedtuple(
    "MNISTLabel", [
        "value",
    ]
)

MNISTImageFileHeader = namedtuple(
    "MNISTImageFileHeader", [
        "magicNumber",
        "maxImages",
        "imgWidth",
        "imgHeight",
    ]
)

MNISTLabelFileHeader = namedtuple(
    "MNISTLabelFileHeader", [
        "magicNumber",
        "maxLabels",
    ]
)


class Cell:
    def __init__(self, input_vector, weight, output):
        # TODO: input_vector/output should not be member?
        self.input_vector = input_vector  # 28x28
        self.weight = weight  # 28x28
        self.size = len(self.input_vector)
        assert self.size == len(self.weight), "Mismath in input/weight lengths"
        self.output = output  # range: [0, 1]

    def __str__(self):
        # TODO: this is a dummy implementation for testing
        return "weight = {}".format(self.weight[300])


Layer = namedtuple(
    "Layer", [
        "cells",  # 10 (for the digits 0-9)
    ]
)


def main():
    train_images = read_image_file("./train-images-idx3-ubyte")
    train_labels = read_label_file("./train-labels-idx1-ubyte")
    initial_layer = init_layer()
    trained_layer = use_layer(initial_layer, train_images, train_labels,
                              train=True)

    test_images = read_image_file("./t10k-images-idx3-ubyte")
    test_labels = read_label_file("./t10k-labels-idx1-ubyte")
    use_layer(trained_layer, test_images, test_labels)


def use_layer(layer, images, labels, train=True):
    error_count = 0
    for index, (image, label) in enumerate(zip(images, labels)):
        if train:
            target_output = get_target_output(label)
            for cell, target in zip(layer.cells, target_output):
                train_cell(cell, image, target)
        predicted_number = get_layer_prediction(layer)
        if predicted_number != label.value:
            error_count += 1
            if not error_count % 100:
                print("Prediction: {}, actual: {}, "
                      "success rate: {:.02} (error count: {}) "
                      "at {} image {}".format(
                          predicted_number,
                          label.value,
                          (1 - error_count / (index + 1)),
                          error_count,
                          'training' if train else 'test',
                          index)
                      )
                print_image(image.pixels)
    print("Overall success rate: {:.02} (error count: {}) [{}]".format(
        (1 - error_count / (len(images))),
        error_count,
        'train' if train else 'test'))
    return layer


def get_target_output(label):
    # print("label = {}, value = {}".format(label, label.value))
    """Create target vector according to target number."""
    return [1 if x == label.value else 0 for x in range(10)]


def train_cell(cell, image, target):
    set_cell_input(cell, image)
    calc_cell_output(cell)
    error = get_cell_error(cell, target)
    update_cell_weights(cell, error)


def read_image_file(filename):
    """ See file formats at http://yann.lecun.com/exdb/mnist/ """
    with open(filename, "rb") as image_file:
        data = image_file.read()
        if not data:
            print("No data!")
            exit(1)

    struct_fmt = '>4i'  # int[4]
    struct_len = struct.calcsize(struct_fmt)
    buffer = data[:struct_len]
    image_struct = struct.unpack(struct_fmt, buffer)
    print("%d %d %d %d" % image_struct)
    header = MNISTImageFileHeader(*image_struct)

    buffer = data[struct_len:]
    struct_fmt = '>{}B'.format(header.imgHeight * header.imgWidth)
    struct_len = struct.calcsize(struct_fmt)
    images = [MNISTImage(x) for x in struct.iter_unpack(struct_fmt, buffer)]
    # print("first image:\n{}".format(images[0]))
    print("first image:")
    print_image(images[0].pixels)
    print("last image:")
    print_image(images[-1].pixels)
    return images


def print_image(image_pixels):
    for i, pixel in enumerate(image_pixels):
        if pixel > 200:
            print('■ ', end='')
        elif pixel > 150:
            print('▣ ', end='')
        elif pixel > 100:
            print('▩ ', end='')
        elif pixel > 50:
            print('▨ ', end='')
        else:
            print('□ ', end='')
        # TODO: 28 => use imgWidth
        if (i + 1) % 28 == 0:
            print("")
    print("")


def read_label_file(filename):
    """ See file formats at http://yann.lecun.com/exdb/mnist/ """
    with open(filename, "rb") as label_file:
        data = label_file.read()
        if not data:
            print("No data!")
            exit(1)

    struct_fmt = '>2i'  # int[2]
    struct_len = struct.calcsize(struct_fmt)
    buffer = data[:struct_len]
    label_struct = struct.unpack(struct_fmt, buffer)
    print("%d %d" % label_struct)
    header = MNISTLabelFileHeader(*label_struct)

    buffer = data[struct_len:]
    struct_fmt = '>B'
    struct_len = struct.calcsize(struct_fmt)
    labels = [MNISTLabel(x[0]) for x in struct.iter_unpack(struct_fmt, buffer)]
    assert header.maxLabels == len(labels)
    print("first label: {}".format(labels[0]))
    print("last label : {}".format(labels[-1]))
    return labels


def set_cell_input(cell, image):
    for i in range(len(image.pixels)):
        cell.input_vector[i] = int(bool(image.pixels[i]))
        # TODO: try?
        # cell.input_vector[i] = 1 if image.pixels[i] > 100 else 0


def calc_cell_output(cell):
    cell.output = 0
    size = len(cell.input_vector)  # TODO: check size
    for i in range(size):
        cell.output += cell.input_vector[i] * cell.weight[i]
    cell.output /= size  # normalize output [0, 1]


def get_cell_error(cell, target):  # target: 0 or 1
    return target - cell.output


def update_cell_weights(cell, error):
    for i in range(cell.size):
        cell.weight[i] += LEARNING_RATE * cell.input_vector[i] * error


def get_layer_prediction(layer):
    max_output = 0
    index_of_cell_with_max_output = 0
    for i in range(len(layer.cells)):
        if layer.cells[i].output > max_output:
            max_output = layer.cells[i].output
            index_of_cell_with_max_output = i
    return index_of_cell_with_max_output
    # TODO: show seconds most probable prediction


def get_init_input():
    # TODO: use header.imgHeight * header.imgWidth
    return [0 for x in range(28*28)]


def get_init_weight():
    # TODO: use header.imgHeight * header.imgWidth
    return [random.uniform(0, 1) for x in range(28*28)]


def get_init_cell():
    return Cell(get_init_input(), get_init_weight(), 0)


def init_layer():  # get_init_layer
    cells = [get_init_cell() for x in range(10)]
    return Layer(cells)


if __name__ == "__main__":
    main()
