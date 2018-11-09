#!/usr/bin/env python3

# TODO: see:
# https://mmlind.github.io/Simple_1-Layer_Neural_Network_for_MNIST_Handwriting_Recognition/


import math
import random

import test_data
import training_data

DIM = len(training_data.TRAINING_DATA[0].data)


def main():
    random.seed(12345)
    digits_to_weights = {digit: train(digit) for digit in range(10)}
    for digit, (weights, b) in digits_to_weights.items():
        show_training_results(digit, weights, b)
    test(digits_to_weights)


def train(digit):
    print('TRAINING DATA')
    best_weights = [random.uniform(-10, 10) for i in range(DIM)]
    best_b = random.uniform(-10, 10)

    min_sum_error_squared = 9999999999.9
    for _ in range(10000):
        weights = [random.uniform(-10, 10) for i in range(DIM)]
        b = random.uniform(-10, 10)

        sum_error_squared = 0.0
        for data in [
                x for x in training_data.TRAINING_DATA if x.digit == digit]:
            guessed_digit = guess_digit(data.data, weights, b)
            error_squared = (guessed_digit - data.digit) ** 2
            sum_error_squared += error_squared

        if sum_error_squared < min_sum_error_squared:
            min_sum_error_squared = sum_error_squared
            best_weights = weights.copy()
            best_b = b
            # print(
            #     'Found better parameters: weights = {}, b = {} [{}]'
            #     .format(weights, b, min_sum_error_squared))

    print('Using best parameters: weights = {}, b = {} for digit = {}'
          .format(best_weights, best_b, digit))
    return (best_weights, best_b)


def show_training_results(digit, weights, b):
    print('SHOWING TRAINING DATA for DIGIT = {}'.format(digit))
    for data in training_data.TRAINING_DATA:
        guessed_digit = int(guess_digit(data.data, weights, b))
        print('Actual digit = {} vs. guessed digit = {}'
              .format(data.digit, guessed_digit))


def test(digits_to_weights):
    print('SHOWING TEST DATA')
    for data in test_data.TEST_DATA:
        print('Working on digit = {}'.format(data.digit))
        guessed_digit = fit_digit(data.data, digits_to_weights)
        print('Actual digit = {} vs. guessed digit = {}'
              .format(data.digit, guessed_digit))


def guess_digit(data, weights, b):
    return sigmoid(sum(weights[i] * data[i] for i in range(DIM)) + b) * 9.0


def fit_digit(data, digits_to_weights):
    min_error_squared = 9999999999.9
    best_fit_digit = -1
    for digit, (weights, b) in digits_to_weights.items():
        guessed_digit = guess_digit(data, weights, b)
        error_squared = (guessed_digit - digit) ** 2
        print('  guess_digit = {}, error_squared = {}'.format(
            guessed_digit, error_squared))
        if error_squared < min_error_squared:
            min_error_squared = error_squared
            best_fit_digit = digit
    return best_fit_digit


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


if __name__ == "__main__":
    main()
