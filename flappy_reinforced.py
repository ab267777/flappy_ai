from __future__ import print_function
import cv2
import sys
sys.path.append("game/")
import wrapped_flappy_bird as game
import random
from collections import deque
import numpy as np
from keras import backend as K
import tensorflow as tf 
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
import time


ACTIONS = 2 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVE = 5000. # timesteps to observe before training
EXPLORE = 2000000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.2 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
FRAME_PER_ACTION = 1
	
def image_preprocess(img):
	resized = cv2.resize(img, (80, 80))
	gray =  np.expand_dims(cv2.transpose(cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)), axis=2)
	r ,t = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
	t = np.reshape(t, (80, 80, 1))
	#print(x_t.shape, resized.shape)
	s_t = np.concatenate((resized, gray),axis=2)
	#print(s_t.shape, resized.shape)
	return s_t


def custom_loss(fc,a,y):

	# Create a loss function that adds the MSE loss to the mean of all squared activations of a specific layer
	def loss(y_true,y_pred):
		mul = K.sum(K.dot(fc, K.transpose(a)))
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
	fc2 = Dense(ACTIONS, activation='linear')(fc1)

	model = Model([inputs,a,y], fc2)
	model.compile(optimizer='adam',loss=custom_loss(fc2,a,y), metrics=['accuracy'])
	model.summary()
	return model

def train():
	starttime=time.time()
	model = network()
	D = deque()
	epsilon = INITIAL_EPSILON
	game_state = game.GameState()

	dummy_a = np.zeros(ACTIONS)
	dummy_a[0] = 1

	x_tc, r_t, T_t = game_state.frame_step(dummy_a)
	x_t = image_preprocess(x_tc)
	x_tt = x_t
	t = 0
	while 1:
		t += 1
		
		x_t = np.asarray([x_tt])
		a_t = np.zeros([ACTIONS])
		dummy_a = np.asarray([a_t])
		dummy_y = np.asarray([[0.0]])
		action_index = 0
		
		Q_t = model.predict([x_t,dummy_a,dummy_y])[0]
		
		if random.random() <= epsilon:
			print("RANDOM step")
			a_t[random.randrange(ACTIONS)] = 1
		else:
			a_t[np.argmax(Q_t)] = 1

		if epsilon > FINAL_EPSILON and t > OBSERVE:
			epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

		x_tc, r_t, T_t = game_state.frame_step(a_t)
		x_tn = image_preprocess(x_tc)
		D.append((x_tt, a_t, r_t, x_tn, T_t))
		x_tt = x_tn

		if len(D) > REPLAY_MEMORY:
			D.popleft()

		if t % 10000 == 0:
			model.save('saved_networks/' + 'dqn' + str(t))
			print("Time : ", t)


		if t > OBSERVE:
			minibatch = random.sample(D, BATCH)

			s_t_batch = np.asarray([d[0] for d in minibatch])
			a_batch = np.asarray([d[1] for d in minibatch])
			r_batch = np.asarray([d[2] for d in minibatch])
			s_tn_batch = np.asarray([d[3] for d in minibatch])
			t_batch = [d[4] for d in minibatch]

			y_batch = np.zeros(shape=(BATCH,1))
			y_batch_dummy = np.zeros(shape=(BATCH,1))
			Q_tn_batch = model.predict([s_tn_batch,a_batch,y_batch_dummy])

			for i in range(0, len(minibatch)):
				# if terminal, only equals reward
				if t_batch[i]:
					y_batch[i] = r_batch[i]
				else:
					y_batch[i] = GAMMA * np.max(Q_tn_batch[i])
			model.fit(x=[s_t_batch,a_batch,y_batch_dummy], y=y_batch,batch_size=32)
		time.sleep(0.1 - ((time.time() - starttime) % 0.1))

def main():
	train()

if __name__ == "__main__":
	main()