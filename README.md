# Flappy Bird Using Reinforcement Learning in Keras

## Objective
Learn to play Flappy Bird game using [Reinforcement Learning](https://en.wikipedia.org/wiki/Reinforcement_learning). For deep learning I have used keras for its simplicity. I am using Q-learning algorithm to achieve this. You can find paper [here](https://arxiv.org/pdf/1312.5602v1.pdf).

## How to run
* Install tensorflow, keras, pygame and opencv
* `python flappy_reinforced.py` ==> You can either test using pretrained weights or choose to train from scratch

## Keras Model
|Layer (type)                 |Output Shape              |Param #   |
|-----------------------------|:------------------------:|---------:|
|input_1 (InputLayer)         |(None, 80, 80, 4)         |0         |
|conv2d_1 (Conv2D)            |(None, 20, 20, 32)        |8224      |
|max_pooling2d_1 (MaxPooling2 |(None, 10, 10, 32)        |0         |
|conv2d_2 (Conv2D)            |(None, 5, 5, 64)          |32832     |
|conv2d_3 (Conv2D)            |(None, 5, 5, 64)          |4160      |
|flatten_1 (Flatten)          |(None, 1600)              |0         |
|dense_1 (Dense)              |(None, 512)               |819712    |
|dense_2 (Dense)              |(None, 2)                 |1026      |

## Credits
* [yenchenlin/DeepLearningFlappyBird](https://github.com/yenchenlin/DeepLearningFlappyBird) for wrapped bird game of [sourabhv/FlapPyBird](https://github.com/sourabhv/FlapPyBird) and his wonderful documentation.
* [Q-Learning Paper](https://arxiv.org/pdf/1312.5602v1.pdf).




