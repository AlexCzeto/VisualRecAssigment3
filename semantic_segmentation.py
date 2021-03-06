import tensorflow as tf
import numpy as np
from input_helper import TextureImages
rate = 0.003
EPOCHS = 10
BATCH_SIZE = 64
NUM_ITERS = int(2000 / BATCH_SIZE * EPOCHS)

def SemSeg(input_tensor, is_training):
	# TODO: Implement Semantic Segmentation network here
	# Returned logits must be a tensor of size:
	# (None, image_height, image_width, num_classes + 1)
	# 1st dimension is batch dimension
	# image_height and image_width are the height and width of input_tensor
	# last dimension is the softmax dimension. There are 4 texture classes plus 1 background class
	# therefore last dimension will be 5
	# Hint: To make your output tensor has the same height and width with input_tensor,
	# you can use tf.image.resize_bilinear


	# conv layer 1
	conv1 = tf.layers.conv2d(inputs=input_tensor,filters=8,
						 kernel_size=[5, 5],padding="valid",activation=tf.nn.relu)
	print("conv1",conv1.shape)
	
	# maxpool layer 1
	pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
	print("pool1",pool1.shape)
	
	# Convolutional Layer #2 
	conv2 = tf.layers.conv2d(inputs=pool1,filters=16,kernel_size=[3, 3],
						  padding="valid",activation=tf.nn.relu)
	print("conv2",conv2.shape)

	# Pooling Layer #2
	pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
	print("pool2",pool2.shape)

	conv3 = tf.layers.conv2d(inputs=pool2,filters=32,kernel_size=[3, 3],
						  padding="valid",activation=tf.nn.relu)
	print("conv3",conv3.shape)

	# Conv transpose (aka deconvolution)
	conv_logits = tf.layers.conv2d_transpose(inputs=conv3,filters=5,kernel_size=[8, 8],
						  strides=5, padding="valid",activation=tf.nn.relu)
	print("conv_logits",conv_logits.shape)

	newShape = input_tensor.get_shape().as_list()
	print(newShape[1:3])

	conv_logits = tf.image.resize_bilinear(conv_logits,newShape[1:3])
	print(conv_logits.get_shape().as_list())

	#conv_logits = tf.nn.softmax(conv_logits)

	#logits = tf.placeholder(tf.float32, (None, input_tensor.shape[0], input_tensor.shape[1], 5))
	return conv_logits


def run():
	# You can tune the hyperparameters here.


	train_set = TextureImages('train', batch_size=BATCH_SIZE)
	test_set = TextureImages('test', shuffle=False)

	tf.reset_default_graph()
	x = tf.placeholder(tf.float32, (None, 196, 196, 3))
	y = tf.placeholder(tf.int32, (None, 196, 196, 1))
	one_hot_y = tf.one_hot(y, 5)
	is_training = tf.placeholder(tf.bool, ())


	logits = SemSeg(x, is_training=is_training)
	prediction = tf.argmax(logits, axis=-1)
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y))
	loss_operation = cross_entropy
	optimizer = tf.train.AdamOptimizer(learning_rate=rate)
	grads_and_vars = optimizer.compute_gradients(loss_operation, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
	training_operation = optimizer.apply_gradients(grads_and_vars)

	def evaluation(images, true_labels):
		eval_batch_size = 100
		predicted_labels = []
		for start_index in range(0, len(images), eval_batch_size):
			end_index = start_index + eval_batch_size
			batch_x = images[start_index: end_index]
			#RealLogits = sess.run(logits, feed_dict={x: batch_x, is_training: False}) 
			#print(RealLogits)
			batch_predicted_labels = sess.run(prediction, feed_dict={x: batch_x, is_training: False})
			predicted_labels += list(batch_predicted_labels)
		predicted_labels = np.vstack(predicted_labels).flatten()
		true_labels = true_labels.flatten()
		accuracy = float((predicted_labels == true_labels).astype(np.int32).sum()) / true_labels.size
		return predicted_labels, accuracy

	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	print("Training...")
	for i in range(NUM_ITERS):
		batch_x, batch_y = train_set.get_next_batch()
		_ = sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, is_training: True})
		if (i + 1) % 50 == 0 or i == NUM_ITERS - 1:
			_, test_accuracy = evaluation(test_set._images, test_set._masks)
			print("Iter {}: Test Pixel Accuracy = {:.3f}".format(i, test_accuracy))

	print('Evaluating on test set')
	_, test_accuracy = evaluation(test_set._images, test_set._masks)
	print("Test Pixel Accuracy = {:.3f}".format(test_accuracy))
	sess.close()
	return test_accuracy


if __name__ == '__main__':
	run()
