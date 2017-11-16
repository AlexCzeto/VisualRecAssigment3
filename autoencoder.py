import tensorflow as tf
import numpy as np
from input_helper import MNIST
from functools import partial
from tensorflow.contrib.layers import flatten

RATE = 0.001
EPOCHS = 10
BATCH_SIZE = 64
NUM_ITERS = int(55000 / BATCH_SIZE * EPOCHS)

MU = 0
SIGMA = 0.1

def AutoEncoder(input_tensor, is_training):
	# TODO: write your autoencoder here
	# There are two part:
	# - An autoencoder: output is recon
	# - A classification branch from the hidden representation of the autoencoder: output is logits

	n_inputs = 28 * 28
	n_hidden1 = 500
	n_hidden2 = 100
	n_hidden3 = 20 # codings
	n_hidden4 = n_hidden2
	n_hidden5 = n_hidden1
	n_outputs = n_inputs


	initializer = tf.contrib.layers.variance_scaling_initializer()
	my_dense_layer = partial(tf.layers.dense , activation=tf.nn.relu, kernel_initializer=initializer)

	#Encoding 
	hidden1 = my_dense_layer(input_tensor, n_hidden1)
	hidden2 = my_dense_layer(hidden1, n_hidden2)

	#Middle 
	hidden3_mean = my_dense_layer(hidden2,n_hidden3,activation=None)
	hidden3_gamma = my_dense_layer(hidden2,n_hidden3,activation=None)
	noise = tf.random_normal(tf.shape(hidden3_gamma),dtype=tf.float32)
	hidden3 = hidden3_mean + tf.exp(0.5 * hidden3_gamma) * noise

	#Classification 
	fc0   = flatten(hidden3)

	fc1_W = tf.Variable(tf.truncated_normal(shape=(20, 10), mean = MU, stddev = SIGMA), name='fc1_W')
	fc1_b = tf.Variable(tf.zeros(10), name='fc1_b')
	logits   = tf.matmul(fc0, fc1_W) + fc1_b

	#Decoding 
	hidden4 = my_dense_layer(hidden3, n_hidden4)
	hidden5 = my_dense_layer(hidden4, n_hidden5)

	#Output
	newX = my_dense_layer(hidden5, n_outputs, activation=None)
	recon = tf.nn.softmax(newX)

	return recon, logits


def run():
	# You can tune the hyperparameters here


	train_set = MNIST('train', batch_size=BATCH_SIZE)
	valid_set = MNIST('valid')
	test_set = MNIST('test', shuffle=False)

	tf.reset_default_graph()
	x = tf.placeholder(tf.float32, (None, 784))
	y = tf.placeholder(tf.int32, (None, 1))
	is_labeled = tf.placeholder(tf.float32, (None, 1))
	is_training = tf.placeholder(tf.bool, ())
	one_hot_y = tf.one_hot(y, 10)


	recon, logits = AutoEncoder(x, is_training=is_training)
	prediction = tf.argmax(logits, axis=1)
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y) * is_labeled)
	recon_loss = tf.reduce_mean((recon - x) ** 2)
	loss_operation = cross_entropy + recon_loss
	optimizer = tf.train.AdamOptimizer(learning_rate=RATE)
	grads_and_vars = optimizer.compute_gradients(loss_operation, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
	training_operation = optimizer.apply_gradients(grads_and_vars)

	def evaluation(images, true_labels):
		eval_batch_size = 100
		predicted_labels = []
		for start_index in range(0, len(images), eval_batch_size):
			end_index = start_index + eval_batch_size
			batch_x = images[start_index: end_index]
			batch_predicted_labels = sess.run(prediction, feed_dict={x: batch_x, is_training: False})
			predicted_labels += list(batch_predicted_labels)
		predicted_labels = np.vstack(predicted_labels).flatten()
		true_labels = true_labels.flatten()
		accuracy = float((predicted_labels == true_labels).astype(np.int32).sum()) / len(images)
		return predicted_labels, accuracy

	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	print("Training...")
	for i in range(NUM_ITERS):
		batch_x, batch_y, batch_is_labeled = train_set.get_next_batch()
		_ = sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, is_labeled: batch_is_labeled, is_training: True})
		if (i + 1) % 1000 == 0 or i == NUM_ITERS - 1:
			_, validation_accuracy = evaluation(valid_set._images, valid_set._labels)
			print("Iter {}: Validation Accuracy = {:.3f}".format(i, validation_accuracy))

	print('Evaluating on test set')
	_, test_accuracy = evaluation(test_set._images, test_set._labels)
	print("Test Accuracy = {:.3f}".format(test_accuracy))

	sess.close()
	return test_accuracy



if __name__ == '__main__':
	run()
