#!/usr/bin/env python3

from collections import namedtuple
import struct


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
        MNISTImage(x)
        for x in struct.iter_unpack(struct_fmt, buffer)
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
