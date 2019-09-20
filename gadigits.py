#!/usr/bin/env python3

"""Experimenting with Genetic Algorithms using the MNIST data set.

Note: download and extract data files from http://yann.lecun.com/exdb/mnist/
  train-images-idx3-ubyte.gz:  training set images (9912422 bytes)
  train-labels-idx1-ubyte.gz:  training set labels (28881 bytes)
  t10k-images-idx3-ubyte.gz:   test set images (1648877 bytes)
  t10k-labels-idx1-ubyte.gz:   test set labels (4542 bytes)
"""

# import random
from collections import namedtuple


class MNISTImage:
    def __init__(self, pixels):
        self.pixels = pixels
        self.bw_pixels = [int(bool(pixel)) for pixel in pixels]


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


class Weight():
    def __init__(self, value):
        if isinstance(value, str):
            self._as_string = value
            self._as_number = int(value, base=2) - 32766
        else:
            self._as_number = value
            self._as_string = bin(value + 32766)[2:].zfill(16)

    def __str__(self):
        return 'value = {} as string = {}'.format(self._as_number,
                                                  self._as_string)


# class DigitRecongnizer():
#     def __init__(self, digit_to_be_recognized):
#         self._weights = self._get_init_weight()  # 28x28
#
#     @staticmethod
#     def _get_init_weight():
#         return [Weight(random.uniform(0, 1)) for x in range(28*28)]
#
#
def main():
    print('-32766: {}'.format(Weight(-32766)))
    print('-32766: {}'.format(Weight('0000000000000000')))

    print('-32765: {}'.format(Weight(-32765)))

    print('-1: {}'.format(Weight(-1)))
    print('-1: {}'.format(Weight('0111111111111101')))

    print('0: {}'.format(Weight(0)))
    print('0: {}'.format(Weight('0111111111111110')))

    print('1: {}'.format(Weight(1)))
    print('2: {}'.format(Weight(2)))
    print('3: {}'.format(Weight(3)))

    print('32766: {}'.format(Weight(32766)))
    print('32766: {}'.format(Weight('1111111111111100')))

    print('32767: {}'.format(Weight(32767)))
    print('32768: {}'.format(Weight(32768)))

    print('32769: {}'.format(Weight(32769)))
    print('32769: {}'.format(Weight('1111111111111111')))
    # digit0 = DigitRecongnizer(0)


if __name__ == "__main__":
    main()
