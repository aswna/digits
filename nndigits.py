#!/usr/bin/env python3

"""This is going to be the Python implementation of
https://mmlind.github.io/Simple_1-Layer_Neural_Network_for_MNIST_Handwriting_Recognition/
using Genetic Algorithms in the background.

Note: download and extract data files from http://yann.lecun.com/exdb/mnist/
  train-images-idx3-ubyte.gz:  training set images (9912422 bytes)
  train-labels-idx1-ubyte.gz:  training set labels (28881 bytes)
  t10k-images-idx3-ubyte.gz:   test set images (1648877 bytes)
  t10k-labels-idx1-ubyte.gz:   test set labels (4542 bytes)

See also:
    https://github.com/mmlind/mnist-1lnn/
"""

import os
import pickle
import random
from collections import namedtuple

from mnist_utils import print_image
from mnist_utils import read_image_file
from mnist_utils import read_label_file


LEARNING_RATE = 0.05


class Cell:
    def __init__(self):
        self.weight = self._get_init_weight()  # 28x28
        self.size = len(self.weight)
        self.output = 0  # range: [0, 1]

    def __str__(self):
        # TODO: this is a dummy implementation for testing
        return "weight = {}".format(self.weight[300])

    @staticmethod
    def _get_init_weight():
        # TODO: use header.imgHeight * header.imgWidth
        return [random.uniform(0, 1) for x in range(28*28)]


Layer = namedtuple(
    "Layer", [
        "cells",  # 10 (for the digits 0-9)
    ]
)


def main():
    trained_layer = train_layer()
    test_layer(trained_layer)


def train_layer():
    pickle_filename = "nn_trained_layer.dat"
    trained_layer = None
    if os.path.isfile(pickle_filename):
        print("Found pickle file: {}".format(pickle_filename))
        with open(pickle_filename, "rb") as file_obj:
            trained_layer = pickle.load(file_obj)
    else:
        print("Could not find pickle file: {}".format(pickle_filename))
        initial_layer = init_layer()
        train_images = read_image_file("./train-images-idx3-ubyte")
        train_labels = read_label_file("./train-labels-idx1-ubyte")
        trained_layer = use_layer(initial_layer, train_images, train_labels,
                                  train=True)
        with open(pickle_filename, "wb") as file_obj:
            pickle.dump(trained_layer, file_obj)
    return trained_layer


def test_layer(trained_layer):
    test_images = read_image_file("./t10k-images-idx3-ubyte")
    test_labels = read_label_file("./t10k-labels-idx1-ubyte")
    use_layer(trained_layer, test_images, test_labels, train=False)


def use_layer(layer, images, labels, train=False):
    error_count = 0
    for index, (image, label) in enumerate(zip(images, labels)):
        target_output = get_target_output(label)
        for cell, target in zip(layer.cells, target_output):
            calc_cell_output(cell, image)
            if train:
                train_cell(cell, image, target)
        predicted_number = get_layer_prediction(layer)
        if predicted_number != label.value:
            error_count += 1
            if error_count % 100 == 0:
                print("Prediction: {}, actual: {}, "
                      "success rate: {:.02} (error count: {}) "
                      "at {} image {}".format(
                          predicted_number,
                          label.value,
                          (1 - error_count / (index + 1)),
                          error_count,
                          'training' if train else 'testing',
                          index)
                      )
                print_image(image.pixels)
    print("Overall success rate: {:.02} "
          "(error count: {}, image count: {}) [{}]"
          .format(
              (1 - error_count / (len(images))),
              error_count,
              len(images),
              'train' if train else 'test')
          )
    return layer


def get_target_output(label):
    # print("label = {}, value = {}".format(label, label.value))
    """Create target vector according to target number."""
    return [1 if x == label.value else 0 for x in range(10)]


def train_cell(cell, image, target):
    error = get_cell_error(cell, target)
    update_cell_weights(cell, image, error)


def calc_cell_output(cell, image):
    cell.output = 0
    for i in range(cell.size):
        cell.output += image.bw_pixels[i] * cell.weight[i]
    cell.output /= cell.size  # normalize output [0, 1]


def get_cell_error(cell, target):  # target: 0 or 1
    return target - cell.output


def update_cell_weights(cell, image, error):
    for i in range(cell.size):
        cell.weight[i] += LEARNING_RATE * (image.pixels[i] / 255) * error


def get_layer_prediction(layer):
    max_output = 0
    index_of_cell_with_max_output = 0
    for i in range(len(layer.cells)):
        if layer.cells[i].output > max_output:
            max_output = layer.cells[i].output
            index_of_cell_with_max_output = i
    return index_of_cell_with_max_output
    # TODO: show second most probable prediction


def init_layer():
    cells = [Cell() for x in range(10)]
    return Layer(cells)


if __name__ == "__main__":
    main()
