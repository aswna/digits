from collections import namedtuple

TrainingData = namedtuple(
    'TrainingData', (
        'data',
        'digit',
    )
)

TRAINING_DATA = [
    TrainingData(
        data=[
            1, 1, 1,
            1, 0, 1,
            1, 0, 1,
            1, 0, 1,
            1, 1, 1
        ],
        digit=0
    ),
    TrainingData(
        data=[
            0, 1, 1,
            1, 0, 1,
            1, 0, 1,
            1, 0, 1,
            1, 1, 0
        ],
        digit=0
    ),
    TrainingData(
        data=[
            1, 1, 0,
            1, 0, 1,
            1, 0, 1,
            1, 0, 1,
            0, 1, 1
        ],
        digit=0
    ),
    TrainingData(
        data=[
            0, 0, 1,
            0, 0, 1,
            0, 0, 1,
            0, 0, 1,
            0, 0, 1
        ],
        digit=1
    ),
    TrainingData(
        data=[
            0, 1, 0,
            0, 1, 0,
            0, 1, 0,
            0, 1, 0,
            0, 1, 0
        ],
        digit=1
    ),
    TrainingData(
        data=[
            0, 0, 1,
            0, 1, 1,
            0, 0, 1,
            0, 0, 1,
            0, 0, 1
        ],
        digit=1
    ),
    TrainingData(
        data=[
            0, 0, 1,
            0, 1, 1,
            1, 0, 1,
            0, 0, 1,
            0, 0, 1
        ],
        digit=1
    ),
    TrainingData(
        data=[
            0, 1, 0,
            1, 1, 0,
            0, 1, 0,
            0, 1, 0,
            0, 1, 0
        ],
        digit=1
    ),
    TrainingData(
        data=[
            1, 1, 1,
            0, 0, 1,
            1, 1, 1,
            1, 0, 0,
            1, 1, 1
        ],
        digit=2
    ),
    TrainingData(
        data=[
            1, 1, 1,
            0, 0, 1,
            0, 1, 1,
            1, 0, 0,
            1, 1, 1
        ],
        digit=2
    ),
    TrainingData(
        data=[
            1, 1, 1,
            0, 0, 1,
            0, 1, 1,
            0, 0, 1,
            1, 1, 1
        ],
        digit=3
    ),
    TrainingData(
        data=[
            1, 1, 0,
            0, 0, 1,
            0, 1, 1,
            0, 0, 1,
            1, 1, 0
        ],
        digit=3
    ),
    TrainingData(
        data=[
            1, 1, 1,
            0, 0, 1,
            0, 1, 0,
            0, 0, 1,
            1, 1, 1
        ],
        digit=3
    ),
    TrainingData(
        data=[
            1, 0, 1,
            1, 0, 1,
            1, 1, 1,
            0, 0, 1,
            0, 0, 1
        ],
        digit=4
    ),
    TrainingData(
        data=[
            1, 0, 0,
            1, 0, 0,
            1, 1, 1,
            0, 1, 0,
            0, 1, 0
        ],
        digit=4
    ),
    TrainingData(
        data=[
            1, 1, 1,
            1, 0, 0,
            1, 1, 1,
            0, 0, 1,
            1, 1, 1
        ],
        digit=5
    ),
    TrainingData(
        data=[
            0, 1, 1,
            1, 0, 0,
            1, 1, 1,
            0, 0, 1,
            1, 1, 0
        ],
        digit=5
    ),
    TrainingData(
        data=[
            1, 1, 1,
            1, 0, 0,
            1, 1, 1,
            1, 0, 1,
            1, 1, 1
        ],
        digit=6
    ),
    TrainingData(
        data=[
            0, 1, 1,
            1, 0, 0,
            1, 1, 1,
            1, 0, 1,
            0, 1, 1
        ],
        digit=6
    ),
    TrainingData(
        data=[
            0, 1, 1,
            1, 0, 0,
            1, 1, 0,
            1, 0, 1,
            0, 1, 0
        ],
        digit=6
    ),
    TrainingData(
        data=[
            1, 1, 1,
            0, 0, 1,
            0, 0, 1,
            0, 0, 1,
            0, 0, 1
        ],
        digit=7
    ),
    TrainingData(
        data=[
            1, 1, 1,
            0, 0, 1,
            0, 0, 1,
            0, 1, 0,
            1, 0, 0
        ],
        digit=7
    ),
    TrainingData(
        data=[
            1, 1, 1,
            1, 0, 1,
            1, 1, 1,
            1, 0, 1,
            1, 1, 1
        ],
        digit=8
    ),
    TrainingData(
        data=[
            0, 1, 0,
            1, 0, 1,
            0, 1, 0,
            1, 0, 1,
            0, 1, 0
        ],
        digit=8
    ),
    TrainingData(
        data=[
            0, 1, 0,
            1, 0, 1,
            1, 1, 1,
            1, 0, 1,
            0, 1, 0
        ],
        digit=8
    ),
    TrainingData(
        data=[
            1, 1, 1,
            1, 0, 1,
            1, 1, 1,
            0, 0, 1,
            1, 1, 1
        ],
        digit=9
    ),
    TrainingData(
        data=[
            0, 1, 0,
            1, 0, 1,
            0, 1, 1,
            0, 0, 1,
            1, 1, 0
        ],
        digit=9
    ),
    TrainingData(
        data=[
            0, 1, 1,
            1, 0, 1,
            0, 1, 1,
            0, 0, 1,
            1, 1, 1
        ],
        digit=9
    )
]
