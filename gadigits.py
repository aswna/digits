#!/usr/bin/env python3

"""Experimenting with Genetic Algorithms using the MNIST data set.

Note: download and extract data files from http://yann.lecun.com/exdb/mnist/
  train-images-idx3-ubyte.gz:  training set images (9912422 bytes)
  train-labels-idx1-ubyte.gz:  training set labels (28881 bytes)
  t10k-images-idx3-ubyte.gz:   test set images (1648877 bytes)
  t10k-labels-idx1-ubyte.gz:   test set labels (4542 bytes)
"""

import os
import pickle
import random

from mnist_utils import read_image_file


class Weight():
    min_value = -32.766
    max_value = 32.769

    def __init__(self, value):
        if isinstance(value, str):
            self.as_string = value
            self.as_number = (int(value, base=2) - 32766) / 1000
        else:
            self.as_number = value
            self.as_string = bin(int(value * 1000 + 32766))[2:].zfill(16)

    def __str__(self):
        return 'Weight({})'.format(self.as_number)


class DigitRecognizer():
    def __init__(self, dim=28):
        self.dim = dim
        self.weights = self.get_init_weight()
        self.weights_combined = ''.join(x.as_string for x in self.weights)

    def fit(self, mnist_image):
        """Return fitness in range [-1, 1]."""

        summa = 0
        for pixel, weight in zip(mnist_image.bw_pixels, self.weights):
            print("weight = {}".format(weight))
            val = (
                (pixel * weight.as_number - Weight.min_value) /
                (Weight.max_value - Weight.min_value)
            )
            print("val = {}".format(val))
            summa += val
        summa /= len(self.weights)
        summa = summa * 2 - 1
        print("==> summa = {}".format(summa))
        return summa

    def get_init_weight(self):
        return [
            Weight(random.uniform(Weight.min_value, Weight.max_value))
            for x in range(self.dim ** 2)
        ]

    def __str__(self):
        return 'DigitRecognizer({})'.format(
            ', '.join(str(weight) for weight in self.weights))


class DigitRecognizers():
    def __init__(self):
        self.digit_recognizers = [DigitRecognizer(dim=2) for i in range(4)]

    def __str__(self):
        return 'DigitRecognizers({})'.format(', '.join(str(dr) for dr in self.digit_recognizers))


def main():
    train_population()


def train_population():
    pickle_filename = "ga_trained_population.dat"
    trained_population = None
    if os.path.isfile(pickle_filename):
        print("Found pickle file: {}".format(pickle_filename))
        with open(pickle_filename, "rb") as file_obj:
            trained_population = pickle.load(file_obj)
    else:
        print("Could not find pickle file: {}".format(pickle_filename))
        # initial_population = init_population()
        digit0_recognizer = DigitRecognizer()
        train_images = read_image_file("./train-images-idx3-ubyte")
        # train_labels = read_label_file("./train-labels-idx1-ubyte")
        s = digit0_recognizer.fit(train_images[0])
        print('summa = {}'.format(s))
        # trained_population = use_layer(initial_layer,
        #                                train_images,
        #                                train_labels,
        #                                train=True)
        # with open(pickle_filename, "wb") as file_obj:
        #     pickle.dump(trained_population, file_obj)
    return trained_population


if __name__ == "__main__":
    main()
