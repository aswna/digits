#!/usr/bin/env python3

# TODO: see:
# https://mmlind.github.io/Simple_1-Layer_Neural_Network_for_MNIST_Handwriting_Recognition/


import math
import random

import training_data

DIM = len(training_data.TRAINING_DATA[0].data)


def main():
    random.seed(12345)
    (weights, b) = train()
    show_training_results(weights, b)


def train():
    print('TRAINING DATA')
    weights = [random.uniform(-10, 10) for i in range(DIM)]
    best_weights = weights.copy()
    best_b = b = random.uniform(-10, 10)

    min_sum_error_squared = 9999999999.9
    for _ in range(10000):
        weights = [random.uniform(-10, 10) for i in range(DIM)]
        b = random.uniform(-10, 10)
        sum_error_squared = 0.0

        for test_data in training_data.TRAINING_DATA:
            guessed_digit = guess_digit(test_data.data, weights, b)
            error_squared = (guessed_digit - test_data.digit) ** 2
            sum_error_squared += error_squared

        if sum_error_squared < min_sum_error_squared:
            min_sum_error_squared = sum_error_squared
            best_weights = weights.copy()
            best_b = b
            # print(
            #     'Found better parameters: weights = {}, b = {} [{}]'
            #     .format(weights, b, min_sum_error_squared))

    print('Using best parameters: weights = {}, b = {}'
          .format(best_weights, best_b))
    return (best_weights, best_b)


def show_training_results(weights, b):
    for test_data in training_data.TRAINING_DATA:
        guessed_digit = guess_digit(test_data.data, weights, b)
        print('Actual digit = {} vs. guessed digit = {}'
              .format(test_data.digit, guessed_digit))


def guess_digit(data, weights, b):
    return int(
        sigmoid(sum(weights[i] * data[i] for i in range(DIM)) + b) * 9.0)


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


if __name__ == "__main__":
    main()
