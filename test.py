import numpy as np
import tensorflow as tf
import cv2
import sys,argparse

def findVehicle(item):
	if item == 2:
		print("It is a four-wheeler",end="")
	elif item == 1:
		print("It is a two-wheeler",end="")
	elif item == 0:
		print("It is a three-wheeler",end="")
	elif item == 3:
		print("It is a six wheeler",end="")

def maxItem(result):
	items = list(result[0])
	maxitem = max(items)
	return items.index(maxitem)

img = np.reshape(cv2.resize(cv2.imread(sys.argv[1],1), (64, 64)), [1,64,64,3])

sess = tf.Session()
saver = tf.train.import_meta_graph('vehicles.meta')
saver.restore(sess, tf.train.latest_checkpoint('./'))
graph = tf.get_default_graph()

y_pred = graph.get_tensor_by_name("y_pred:0")
X = graph.get_tensor_by_name("X:0") 

results = []
result = sess.run(y_pred, feed_dict={X:img})
temp = maxItem(result)
results.append(temp)

for i in range(10):
	result = sess.run(y_pred, feed_dict={X:img})
	maxitem = maxItem(result)
	if maxitem != temp and maxitem not in results:
		results.append(maxitem)

for i in range(len(results)):
	findVehicle(results[i])
	if i!= (len(results)-1):
		print(" or ", end="")
	else:
		print("")

sess.close()
