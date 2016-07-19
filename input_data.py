from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys

from six.moves import urllib, xrange
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 128, """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', 'data', """Path to the data directory.""")

IMAGE_SIZE = 96
NUM_CLASSES = 2
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 10000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000



def read_input_dataset (filename_queue):
	HEIGHT = 128
	WIDTH = 128
	DEPTH = 3

	label_bytes = 1
	image_bytes = HEIGHT * WIDTH * DEPTH
	# Every record consists of a label followed by the image, with a
	# fixed number of bytes for each.
	record_bytes = label_bytes + image_bytes
	
	# Read a record, getting filenames from the filename_queue.  No
	# header or footer in the file format, so we leave header_bytes
	# and footer_bytes at their default of 0.
	reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
	key, value = reader.read(filename_queue)
	
	# Convert from a string to a vector of uint8 that is record_bytes long.
	record_bytes = tf.decode_raw(value, tf.uint8)
	
	# The first bytes represent the label, which we convert from uint8->int32.
	label = tf.cast (tf.slice (record_bytes, [0], [label_bytes]), tf.int32)
	
	# The remaining bytes after the label represent the image, which we reshape
	# from [depth * height * width] to [depth, height, width].
	depth_major = tf.reshape(tf.slice(record_bytes, [label_bytes], [image_bytes]), [DEPTH, HEIGHT, WIDTH])
	# Convert from [depth, height, width] to [height, width, depth].
	uint8image = tf.transpose(depth_major, [1, 2, 0])
	
	return uint8image, label


def _generate_image_and_label_batch(image, label, min_queue_examples, batch_size, shuffle, summary_name):
	num_preprocess_threads = 4
	if shuffle:
		images, label_batch = tf.train.shuffle_batch( [image, label], batch_size=batch_size, num_threads=num_preprocess_threads, capacity=min_queue_examples + 3 * batch_size, min_after_dequeue=min_queue_examples)
	else:
		images, label_batch = tf.train.batch( [image, label], batch_size=batch_size, num_threads=num_preprocess_threads, capacity=min_queue_examples + 3 * batch_size)
	
	# Display the training images in the visualizer.
	tf.image_summary(summary_name, images)
	
	return images, tf.reshape(label_batch, [batch_size])


def distorted_inputs ():
	data_dir = FLAGS.data_dir
	batch_size = FLAGS.batch_size
	filenames = [os.path.join (data_dir, 'train_batch.bin')]
	
	filename_queue = tf.train.string_input_producer(filenames)
	train_image, train_label = read_input_dataset (filename_queue)
	reshaped_image = tf.cast (train_image, tf.float32)
	
	height = IMAGE_SIZE
	width = IMAGE_SIZE
	
	# Image processing for training the network. Note the many random
	# distortions applied to the image.
	
	# Randomly crop a [height, width] section of the image.
	distorted_image = tf.random_crop(reshaped_image, [height, width, 3])
	
	# Randomly flip the image horizontally.
	distorted_image = tf.image.random_flip_left_right(distorted_image)
	
	# Because these operations are not commutative, consider randomizing
	# the order their operation.
	distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)
	distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)
	
	# Subtract off the mean and divide by the variance of the pixels.
	float_image = tf.image.per_image_whitening(distorted_image)
	
	# Ensure that the random shuffling has good mixing properties.
	min_fraction_of_examples_in_queue = 0.4
	min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * min_fraction_of_examples_in_queue)
	print ('Filling queue with %d images before starting to train. '
	       'This will take a few minutes.' % min_queue_examples)
	
	# Generate a batch of images and labels by building up a queue of examples.
	return _generate_image_and_label_batch (float_image, train_label, min_queue_examples, batch_size, shuffle=True, summary_name='train_images')


def inputs (eval_data):
	data_dir = FLAGS.data_dir
	batch_size = FLAGS.batch_size

	if not eval_data:
		filenames = [os.path.join (data_dir, 'train_batch.bin')]
		num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
	else:
		filenames = [os.path.join(data_dir, 'test_batch.bin')]
		num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
	
	filename_queue = tf.train.string_input_producer(filenames)
	test_image, test_label = read_input_dataset (filename_queue)
	reshaped_image = tf.cast (test_image, tf.float32)
	
	height = IMAGE_SIZE
	width = IMAGE_SIZE
	
	# Image processing for evaluation.
	# Crop the central [height, width] of the image.
	resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image, width, height)
	
	# Subtract off the mean and divide by the variance of the pixels.
	float_image = tf.image.per_image_whitening(resized_image)
	
	# Ensure that the random shuffling has good mixing properties.
	min_fraction_of_examples_in_queue = 0.4
	min_queue_examples = int(num_examples_per_epoch * min_fraction_of_examples_in_queue)
	
	# Generate a batch of images and labels by building up a queue of examples.
	return _generate_image_and_label_batch(float_image, test_label, min_queue_examples, batch_size, shuffle=False, summary_name='test_images')


def input_eval (N):
	filenames = [os.path.join (FLAGS.data_dir, 'test_batch.bin')]
	filename_queue = tf.train.string_input_producer (filenames)
	test_image, test_label = read_input_dataset (filename_queue)
	reshaped_image = tf.cast (test_image, tf.float32)
	
	height = IMAGE_SIZE
	width = IMAGE_SIZE

	images = []
	labels = []

	# Randomly crop N times
	for i in xrange (N):
		distorted_image = tf.random_crop (reshaped_image, [height, width, 3])
		float_image = tf.image.per_image_whitening (distorted_image)

		images.append (tf.expand_dims (float_image, 0))
	
	return tf.concat (0, images), test_label
