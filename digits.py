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

TODO:
https://en.wikipedia.org/wiki/Hilbert_curve#Applications_and_mapping_algorithms
"""

import glob
import os
import struct
import pickle
import random
# import sys
from collections import namedtuple


from PIL import Image

LEARNING_RATE = 0.05
DIM = 32


MNISTImageFileHeader = namedtuple(
    "MNISTImageFileHeader", [
        "magicNumber",
        "maxImages",
        "imgWidth",
        "imgHeight",
    ]
)


class MNISTImage:
    xy2d_map = dict()

    def __init__(self, header, pixels):
        self.pixels = pixels
        assert header.imgWidth == header.imgHeight
        # TODO: get closest (upper) power of 2
        n = DIM
        n2 = n ** 2
        self.bw_pixels = [0] * n2
        # print('n2 = {}, pixels = {}'.format(n2, self.bw_pixels))
        # self.bw_pixels = [int(bool(pixel)) for pixel in pixels]
        j = 0
        for i in range(n2):
            x = i % n
            y = i // n
            if (x, y) not in self.xy2d_map:
                d = xy2d(n, x, y)
                self.xy2d_map[(x, y)] = d
            d = self.xy2d_map[(x, y)]
            # print('  i = {}, x = {}, y = {}, d = {}'.format(i, x, y, d))
            if 2 <= x < n - 2 and 2 <= y < n - 2:
                # print('* i = {}, x = {}, y = {}, d = {}'.format(i, x, y, d))
                self.bw_pixels[d] = int(bool(pixels[j]))
                # self.bw_pixels[d] = int(bool(pixels[j] > 123))
                # self.bw_pixels[d] = pixels[j] / 255
                # print('* i = {}, x = {}, y = {}, d = {}, pixel = {}'
                #       .format(i, x, y, d, self.bw_pixels[d]))
                j += 1
            else:
                # print('E i = {}, x = {}, y = {}, d = {}'.format(i, x, y, d))
                self.bw_pixels[d] = 0
        # sys.exit()


# convert (x, y) to d
def xy2d(n, x, y):
    d = 0
    s = n//2
    while s > 0:
        rx = (x & s) > 0
        ry = (y & s) > 0
        d += s * s * ((3 * rx) ^ ry)
        x, y = rot(n, x, y, rx, ry)
        s //= 2
    return d


# rotate/flip a quadrant appropriately
def rot(n, x, y, rx, ry):
    if ry == 0:
        if rx == 1:
            x = n - 1 - x
            y = n - 1 - y
        x, y = y, x
    return x, y


MNISTLabel = namedtuple(
    "MNISTLabel", [
        "value",
    ]
)

MNISTLabelFileHeader = namedtuple(
    "MNISTLabelFileHeader", [
        "magicNumber",
        "maxLabels",
    ]
)


class Cell:
    def __init__(self):
        # TODO: use header.imgHeight * header.imgWidth
        self.weight = [random.uniform(0, 1) for x in range(DIM**2)]
        self.size = len(self.weight)
        self.output = 0  # range: [0, 1]

    def __str__(self):
        # TODO: this is a dummy implementation for testing
        return "weights = {}".format(
            ", ".join(str(int(w * 100) / 100) for w in self.weight))
        # return "weight = {}".format(self.weight[300])


Layer = namedtuple(
    "Layer", [
        "cells",  # 10 (for the digits 0-9)
    ]
)


def main():
    images = []
    labels = []
    # image_filenames = glob.glob("digit_9/*-25.png")
    image_filenames = glob.glob("digit_9/*-*.png")
    for image_filename in image_filenames:
        im = Image.open(image_filename, 'r')
        pixel_values = list(im.getdata())
        print("im size = {}".format(im.size))
        for x in range(im.size[0]):
            for y in range(im.size[1]):
                print("{:3} ".format(pixel_values[x * im.size[0] + y]), end='')
            print("")
        header = MNISTImageFileHeader(0, 0, 28, 28)
        image = MNISTImage(header, pixel_values)
        images.append(image)

        label = MNISTLabel(9)
        labels.append(label)
    layer = train_layer()
    use_layer(layer, images, labels)


def train_layer():
    pickle_filename = "nn_trained_layer.dat"
    layer = None
    if os.path.isfile(pickle_filename):
        print("found pickle file = {}".format(pickle_filename))
        with open(pickle_filename, "rb") as file_obj:
            layer = pickle.load(file_obj)
    else:
        print("could not found pickle file = {}".format(pickle_filename))
        layer = Layer([Cell() for x in range(10)])

        train_images = read_image_file("./train-images-idx3-ubyte")
        train_labels = read_label_file("./train-labels-idx1-ubyte")
        layer = use_layer(layer, train_images, train_labels, train=True)

        train_images = read_image_file("./t10k-images-idx3-ubyte")
        train_labels = read_label_file("./t10k-labels-idx1-ubyte")
        layer = use_layer(layer, train_images, train_labels, train=True)

        with open(pickle_filename, "wb") as file_obj:
            pickle.dump(layer, file_obj)
    return layer


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
            # if not error_count % 100:
            if True:
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
    print("Overall success rate: {:.02} "
          "(error count: {}, image count: {}) [{}]".format(
            (1 - error_count / (len(images))),
            error_count,
            len(images),
            'train' if train else 'test'))
    return layer


def get_target_output(label):
    # print("label = {}, value = {}".format(label, label.value))
    """Create target vector according to target number."""
    return [1 if x == label.value else 0 for x in range(10)]


def train_cell(cell, image, target):
    # print("cell = {}".format(cell))
    calc_cell_output(cell, image)
    error = get_cell_error(cell, target)
    update_cell_weights(cell, image, error)
    # print("cell = {}".format(cell))


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
    images = [
        MNISTImage(header, pixels)
        for pixels in struct.iter_unpack(struct_fmt, buffer)
    ]
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


def calc_cell_output(cell, image):
    cell.output = 0
    for i in range(cell.size):
        cell.output += image.bw_pixels[i] * cell.weight[i]
    cell.output /= cell.size  # normalize output [0, 1]


def get_cell_error(cell, target):  # target: 0 or 1
    return target - cell.output


def update_cell_weights(cell, image, error):
    for i in range(cell.size):
        cell.weight[i] += LEARNING_RATE * image.bw_pixels[i] * error


def get_layer_prediction(layer):
    max_output = 0
    index_of_cell_with_max_output = 0
    for i in range(len(layer.cells)):
        if layer.cells[i].output > max_output:
            max_output = layer.cells[i].output
            index_of_cell_with_max_output = i
    return index_of_cell_with_max_output
    # TODO: show seconds most probable prediction


if __name__ == "__main__":
    main()
