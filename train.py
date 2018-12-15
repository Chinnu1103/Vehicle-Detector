import cv2
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
import random
from tensorflow.python.framework import ops
import os
import matplotlib.pyplot as plt

IMG_SIZE = 50
CHANNELS = 3
nx = IMG_SIZE*IMG_SIZE*CHANNELS
ny = 4
TRAIN_DIR = "replace_this"

def create_trining_data():
	Images = []
	Labels = []
	Files = os.listdir(TRAIN_DIR)
	for File in Files:
		img = cv2.resize(cv2.imread(os.path.join(TRAIN_DIR,File),1), (IMG_SIZE, IMG_SIZE))
		Images.append(img)
		label = files[0].split('/')[1].split('_')
		
		Labels.append([int(label[0]),int(label[1]),int(label[2])])
		Images, Labels = shuffle(Images, Labels)
		
	return np.array(Images)/255, Labels

def create_placeholders():
	X = tf.placeholder(tf.float32, shape = (None, IMG_SIZE, IMG_SIZE, CHANNELS), name="X")
	Y = tf.placeholder(tf.float32, shape = (None, ny), name="Y")

	return X, Y

def initialize_parameters():
	W1 = tf.get_variable("W1", [8, 8, 3, 64], initializer=tf.contrib.layers.xavier_initializer(seed=0))
	W2 = tf.get_variable("W2", [6, 6, 64, 32], initializer=tf.contrib.layers.xavier_initializer(seed=0))
	W3 = tf.get_variable("W3", [4, 4, 32, 32], initializer=tf.contrib.layers.xavier_initializer(seed=0))
	W4 = tf.get_variable("W4", [2, 2, 32, 16], initializer=tf.contrib.layers.xavier_initializer(seed=0))
	W5 = tf.get_variable("W5", [2, 2, 16, 8], initializer=tf.contrib.layers.xavier_initializer(seed=0))

	parameters = {"W1": W1,
				  "W2": W2,
				  "W3": W3,
				  "W4": W4,
				  "W5": W5}
	
	return parameters

def random_minibatches(X, Y, minibatch_size, num_minibatches):
	minibatches = []
	for i in range(num_minibatches):
		set_X = X[minibatch_size*i:minibatch_size*(i+1)]
		set_Y = Y[minibatch_size*i:minibatch_size*(i+1)]
		minibatches.append((set_X, set_Y))

	set_X = X[num_minibatches*minibatch_size:]
	set_Y = Y[num_minibatches*minibatch_size:]
	minibatches.append((set_X, set_Y))

	return minibatches

def compute_cost(Z, Y):
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = Z, labels = Y))

	return cost

def forward_propagation(X, parameters):
	W1 = parameters['W1']
	W2 = parameters['W2']
	W3 = parameters['W3']
	W4 = parameters['W4']
	W5 = parameters['W5']

	Z1 = tf.nn.conv2d(X,W1, strides = [1,1,1,1], padding = 'SAME')
	A1 = tf.nn.relu(Z1)
	P1 = tf.nn.max_pool(A1, ksize = [1,8,8,1], strides = [1,1,1,1], padding = 'SAME')
	
	Z2 = tf.nn.conv2d(P1,W2, strides = [1,1,1,1], padding = 'SAME')
	A2 = tf.nn.relu(Z2)
	P2 = tf.nn.max_pool(A2, ksize = [1,8,8,1], strides = [1,1,1,1], padding = 'VALID')

	Z3 = tf.nn.conv2d(P2,W3, strides = [1,1,1,1], padding = 'SAME')
	A3 = tf.nn.relu(Z3)
	P3 = tf.nn.max_pool(A3, ksize = [1,4,4,1], strides = [1,1,1,1], padding = 'SAME')
	
	Z4 = tf.nn.conv2d(P3,W4, strides = [1,1,1,1], padding = 'SAME')
	A4 = tf.nn.relu(Z4)
	P4 = tf.nn.max_pool(A4, ksize = [1,4,4,1], strides = [1,1,1,1], padding = 'SAME')
	
	Z5 = tf.nn.conv2d(P4,W5, strides = [1,1,1,1], padding = 'SAME')
	A5 = tf.nn.relu(Z5)
	P5 = tf.nn.max_pool(A5, ksize = [1,4,4,1], strides = [1,4,4,1], padding = 'VALID')
	P5 = tf.contrib.layers.flatten(P5)
	
	Z6 = tf.contrib.layers.fully_connected(P5, 128, activation_fn=tf.nn.relu)
	Z6 = tf.nn.dropout(Z6, 0.86)

	Z7 = tf.contrib.layers.fully_connected(Z6, ny, activation_fn=None)

	return Z7

def model(learning_rate, epochs, minibatch_size, print_cost=True):
	tf.reset_default_graph()
	ops.reset_default_graph()
	X_train, Y_train = create_trining_data()
	print(X_train.shape, Y_train.shape)
	m = X_train.shape[0]
	costs = []
	sess = tf.Session()
	Y_train = tf.reshape(tf.one_hot(Y_train, ny, axis=1), [m, ny])
	Y_train = sess.run(Y_train)
	sess.close()

	parameters = initialize_parameters()
	X, Y = create_placeholders()
	Z = forward_propagation(X, parameters)
	cost = compute_cost(Z, Y)
	optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
	init = tf.global_variables_initializer()
	saver = tf.train.Saver()

	with tf.Session() as sess:
		sess.run(init)
		for epoch in range(epochs):
			minibatch_cost = 0.
			num_minibatches = int(m / minibatch_size)
			minibatches = random_minibatches(X_train, Y_train, minibatch_size, num_minibatches)

			for minibatch in minibatches:
				(minibatch_X, minibatch_Y) = minibatch
				_ , temp_cost = sess.run([optimizer, cost], feed_dict={X:minibatch_X,Y:minibatch_Y})			    
				minibatch_cost += temp_cost / num_minibatches

			if print_cost == True:
				print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))
				costs.append(minibatch_cost)

		y_pred = tf.nn.softmax(Z,name='y_pred')
		plt.plot(np.squeeze(costs))
		plt.ylabel('cost')
		plt.xlabel('iterations (per tens)')
		plt.title("Learning rate =" + str(learning_rate))
		plt.show()
		saver.save(sess, './vehicles')

model(0.0003, 150, 100)