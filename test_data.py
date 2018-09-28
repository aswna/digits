from collections import namedtuple

TestData = namedtuple(
    'TestData', (
        'data',
        'digit',
    )
)

TRAINING_DATA = [
    TestData(
        data=[
            0, 1, 0,
            1, 0, 1,
            1, 0, 1,
            1, 0, 1,
            0, 1, 1
        ],
        digit=0
    ),
    TestData(
        data=[
            0, 0, 0,
            0, 1, 0,
            1, 1, 0,
            0, 1, 0,
            0, 1, 0
        ],
        digit=1
    ),
    TestData(
        data=[
            1, 1, 1,
            0, 0, 1,
            0, 1, 0,
            1, 0, 0,
            0, 1, 1
        ],
        digit=2
    ),
    TestData(
        data=[
            1, 1, 0,
            0, 0, 1,
            0, 1, 0,
            0, 0, 1,
            1, 1, 1
        ],
        digit=3
    ),
    TestData(
        data=[
            0, 1, 0,
            1, 0, 0,
            1, 1, 1,
            0, 0, 1,
            0, 0, 1
        ],
        digit=4
    ),
    TestData(
        data=[
            0, 1, 1,
            1, 0, 0,
            1, 1, 0,
            0, 0, 1,
            1, 1, 1
        ],
        digit=5
    ),
    TestData(
        data=[
            0, 1, 1,
            1, 0, 0,
            1, 1, 0,
            1, 0, 1,
            0, 1, 1
        ],
        digit=6
    ),
    TestData(
        data=[
            1, 1, 1,
            1, 0, 1,
            0, 0, 1,
            0, 0, 1,
            0, 0, 1
        ],
        digit=7
    ),
    TestData(
        data=[
            0, 1, 1,
            1, 0, 1,
            0, 1, 0,
            1, 0, 1,
            0, 1, 1
        ],
        digit=8
    ),
    TestData(
        data=[
            0, 1, 1,
            1, 0, 1,
            0, 1, 1,
            0, 0, 1,
            1, 1, 1
        ],
        digit=9
    ),
]
