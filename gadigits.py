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
from mnist_utils import read_label_file


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
    def __init__(self, digit, dim=28):
        self.digit = digit
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
        return 'DigitRecognizer(digit={}, weights=[{}])'.format(
            self.digit,
            ', '.join(str(weight) for weight in self.weights))


class DigitRecognizers():
    def __init__(self, digit):
        self.digit = digit
        self.digit_recognizers = [DigitRecognizer(digit=digit, dim=2) for i in range(4)]

    def __str__(self):
        return 'DigitRecognizers(digit={}, recognizers=[{}])'.format(
            self.digit,
            ', '.join(str(dr) for dr in self.digit_recognizers))


class DigitsRecognizer():
    def __init__(self):
        self.digits_recognizers = [DigitRecognizers(i) for i in range(10)]

    def train(self, images, labels):
        pass

    def test(self, images, labels):
        # print recognized digits as top list with probability/confidence
        pass


def main():
    digits_recognizer = train_recognizers()
    test_recognizers(digits_recognizer)


def train_recognizers():
    pickle_filename = "ga_trained_recognizers.dat"
    trained_recognizers = None
    if os.path.isfile(pickle_filename):
        print("Found pickle file: {}".format(pickle_filename))
        with open(pickle_filename, "rb") as file_obj:
            trained_recognizers = pickle.load(file_obj)
    else:
        print("Could not find pickle file: {}".format(pickle_filename))
        dr = DigitsRecognizer()

        train_images = read_image_file("./train-images-idx3-ubyte")
        train_labels = read_label_file("./train-labels-idx1-ubyte")
        dr.train(train_images, train_labels)
        # with open(pickle_filename, "wb") as file_obj:
        #     pickle.dump(dr, file_obj)
        trained_recognizers = dr
    return trained_recognizers


def test_recognizers(digits_recognizer):
    test_images = read_image_file("./t10k-images-idx3-ubyte")
    test_labels = read_label_file("./t10k-labels-idx1-ubyte")
    digits_recognizer.test(test_images, test_labels)


if __name__ == "__main__":
    main()
