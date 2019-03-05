from __future__ import print_function
import cv2
import sys
sys.path.append("game/")
import wrapped_flappy_bird as game
import random
from collections import deque
import numpy as np
from keras import backend as K
import keras
import keras.initializers as initializer
from keras.models import Model
from keras.layers import *
from keras.layers.merge import Concatenate
from keras.layers.pooling import GlobalMaxPooling1D
from keras.activations import *
from keras.utils import to_categorical
from keras.models import Sequential, Model
from functools import reduce
import math
import os

ACTIONS = 2 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVE = 100000. # timesteps to observe before training
EXPLORE = 2000000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.0001 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
FRAME_PER_ACTION = 1


def image_preprocess(img):
	resized = cv2.resize(img, (80, 80))
	gray =  np.expand_dims(cv2.transpose(cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)), axis=2)
	#print(x_t.shape, resized.shape)
	s_t = np.concatenate((resized, gray),axis=2)
	#print(s_t.shape, resized.shape)
	return s_t


def custom_loss(fc,a,y):

	# Create a loss function that adds the MSE loss to the mean of all squared activations of a specific layer
	def loss(y_true,y_pred):
		mul = K.sum(multiply([fc,a]),axis=-1)
		return K.mean(K.square(mul - y),axis=-1)
   
	# Return a function
	return loss

def network():
	inputs = Input((80,80,4))
	a = Input(shape=[2])
	y = Input(shape=[1])
	conv1 = Conv2D(filters=32, strides=4, activation='relu',padding='same',use_bias=True, 
		kernel_size=[8,8], kernel_initializer=initializer.TruncatedNormal(stddev=0.01),
		bias_initializer=initializer.Constant(value=0.01))(inputs)
	maxpool1 = MaxPooling2D(pool_size=2, strides=2, padding='same')(conv1)
	conv2 = Conv2D(filters=64, strides=2, activation='relu',padding='same',use_bias=True, 
		kernel_size=[4,4], kernel_initializer=initializer.TruncatedNormal(stddev=0.01),
		bias_initializer=initializer.Constant(value=0.01))(maxpool1)
	maxpool2 = MaxPooling2D(pool_size=2, strides=2, padding='same')(conv2)
	conv3 = Conv2D(filters=64, strides=1, activation='relu',padding='same',use_bias=True, 
		kernel_size=[1,1], kernel_initializer=initializer.TruncatedNormal(stddev=0.01),
		bias_initializer=initializer.Constant(value=0.01))(maxpool2)
	maxpool3 = MaxPooling2D(pool_size=2, strides=2, padding='same')(conv3)
	fci = Flatten()(maxpool3)
	fc1 = Dense(256, activation='relu')(fci)
	fc2 = Dense(ACTIONS, activation='softmax')(fc1)

	model = Model([inputs,a,y], fc2)
	model.compile(optimizer='adam',loss=custom_loss(fc2,a,y), metrics=['accuracy'])
	model.summary()
	return model


def train():
	game_state = game.GameState()
	do_nothing = np.zeros(2)
	do_nothing[0] = 1
	x_t1_colored, r_t, terminal = game_state.frame_step(do_nothing)
	image_preprocess(x_t1_colored)
	model = network()


	#print(x_t1_colored.shape)



def main():
	train()

if __name__ == "__main__":
	main()