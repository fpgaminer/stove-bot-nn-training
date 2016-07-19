from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import re
import sys
from six.moves import urllib
import tensorflow as tf
import input_data
import math
from tensorflow.python import control_flow_ops

FLAGS = tf.app.flags.FLAGS

MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 700.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.

TOWER_NAME = 'tower'


def _activation_summary(x):
	tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
	tf.histogram_summary(tensor_name + '/activations', x)
	tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
	with tf.device('/cpu:0'):
		var = tf.get_variable(name, shape, initializer=initializer)
	return var


def _variable_with_weight_decay(name, shape, stddev, wd):
	var = _variable_on_cpu(name, shape, tf.truncated_normal_initializer(stddev=stddev))
	if wd is not None:
		weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
		tf.add_to_collection('losses', weight_decay)
	return var


def conv2d (x, n_in, n_out, k, s, p='SAME', bias=False, scope='conv'):
	with tf.variable_scope(scope):
		kernel = tf.Variable(tf.truncated_normal([k, k, n_in, n_out], stddev=math.sqrt(2/(k*k*n_in))), name='weight')
		tf.add_to_collection('weights', kernel)
		conv = tf.nn.conv2d(x, kernel, [1,s,s,1], padding=p)
		if bias:
			bias = tf.get_variable('bias', [n_out], initializer=tf.constant_initializer(0.0))
			tf.add_to_collection('biases', bias)
			conv = tf.nn.bias_add(conv, bias)
	
	return conv


def batch_norm (x, n_out, phase_train, scope='bn', affine=True):
	with tf.variable_scope(scope):
		beta = tf.Variable(tf.constant(0.0, shape=[n_out]), name='beta', trainable=True)
		gamma = tf.Variable(tf.constant(1.0, shape=[n_out]), name='gamma', trainable=affine)
		tf.add_to_collection('biases', beta)
		tf.add_to_collection('weights', gamma)
	
		batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
		ema = tf.train.ExponentialMovingAverage(decay=0.99)
	
		def mean_var_with_update():
			ema_apply_op = ema.apply([batch_mean, batch_var])
			with tf.control_dependencies([ema_apply_op]):
				return tf.identity(batch_mean), tf.identity(batch_var)
		mean, var = control_flow_ops.cond(phase_train, mean_var_with_update, lambda: (ema.average(batch_mean), ema.average(batch_var)))
	
		normed = tf.nn.batch_norm_with_global_normalization(x, mean, var, beta, gamma, 1e-3, affine)
	return normed


def inference (x, phase_train):
	keep_prob_train = tf.constant (0.5, dtype=tf.float32)
	keep_prob_test = tf.constant (1.0, dtype=tf.float32)
	keep_prob = control_flow_ops.cond (phase_train,
			lambda: keep_prob_train,
			lambda: keep_prob_test)

	# conv1 -> 96x96x16
	with tf.variable_scope('conv1') as scope:
		y = conv2d (x, 3, 16, 3, 1, 'SAME', True, scope='conv')
		_activation_summary (y)
	print (y)
	
	# Block 1 -> 48x48x32
	with tf.variable_scope('block1') as scope:
		y = batch_norm (y, 16, phase_train, scope='bn')
		y = tf.nn.relu (y, name='relu')
		y = conv2d (y, 16, 32, 3, 2, 'SAME', True, scope='conv1')  # Stride
		y = batch_norm (y, 32, phase_train, scope='bn')
		y = tf.nn.relu (y, name='relu')
		y = tf.nn.dropout (y, keep_prob)
		y = conv2d (y, 32, 32, 3, 1, 'SAME', True, scope='conv2')
		_activation_summary (y)
	print (y)

	# Block 2 -> 24x24x64
	with tf.variable_scope('block2') as scope:
		y = batch_norm (y, 32, phase_train, scope='bn')
		y = tf.nn.relu (y, name='relu')
		y = conv2d (y, 32, 64, 3, 2, 'SAME', True, scope='conv1')  # Stride
		y = batch_norm (y, 64, phase_train, scope='bn')
		y = tf.nn.relu (y, name='relu')
		y = tf.nn.dropout (y, keep_prob)
		y = conv2d (y, 64, 64, 3, 1, 'SAME', True, scope='conv2')
		_activation_summary (y)
	print (y)

	# Block 3 -> 12x12x128
	with tf.variable_scope('block3') as scope:
		y = batch_norm (y, 64, phase_train, scope='bn')
		y = tf.nn.relu (y, name='relu')
		y = conv2d (y, 64, 256, 3, 2, 'SAME', True, scope='conv1')  # Stride
		y = batch_norm (y, 256, phase_train, scope='bn')
		y = tf.nn.relu (y, name='relu')
		y = tf.nn.dropout (y, keep_prob)
		y = conv2d (y, 256, 256, 3, 1, 'SAME', True, scope='conv2')
		_activation_summary (y)
	print (y)

	# Block 4 -> 6x6x256
	#with tf.variable_scope('block4') as scope:
	#	y = batch_norm (y, 128, phase_train, scope='bn')
	#	y = tf.nn.relu (y, name='relu')
	#	y = conv2d (y, 128, 256, 3, 2, 'SAME', True, scope='conv1')  # Stride
	#	y = batch_norm (y, 256, phase_train, scope='bn')
	#	y = tf.nn.relu (y, name='relu')
	#	y = tf.nn.dropout (y, keep_prob)
	#	y = conv2d (y, 256, 256, 3, 1, 'SAME', True, scope='conv2')
	#	_activation_summary (y)
	#print (y)

	with tf.variable_scope('final') as scope:
		y = batch_norm (y, 256, phase_train, scope='bn')
		y = tf.nn.relu (y, name='relu')
		y = tf.nn.avg_pool (y, ksize=[1, 12, 12, 1], strides=[1, 1, 1, 1], padding='VALID', name='avg_pool')
		print (y)
		y = tf.reshape (y, [-1, 256])
		# Linear
		W = _variable_with_weight_decay('weights', [256, 2], 1e-4, 1e-4)
		b = _variable_with_weight_decay ('bias', [2], 0.0, 0.0)
		y = tf.matmul (y, W) + b
		_activation_summary (y)
	print (y)
	
	return y


def loss (logits, labels):
	labels = tf.cast(labels, tf.int64)
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels, name='cross_entropy_per_example')
	cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
	tf.add_to_collection ('losses', cross_entropy_mean)

	weight_l2_losses = [tf.nn.l2_loss(o) for o in tf.get_collection ('weights')]
	weight_decay_loss = tf.mul(1e-4, tf.add_n (weight_l2_losses), name='weight_decay_loss')
	tf.add_to_collection ('losses', weight_decay_loss)

	return tf.add_n(tf.get_collection('losses'), name='total_loss'), cross_entropy_mean


def _add_loss_summaries(total_loss):
	# Compute the moving average of all individual losses and the total loss.
	loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
	losses = tf.get_collection('losses')
	loss_averages_op = loss_averages.apply(losses + [total_loss])

	# Attach a scalar summary to all individual losses and the total loss; do the
	# same for the averaged version of the losses.
	for l in losses + [total_loss]:
		# Name each loss as '(raw)' and name the moving average version of the loss
		# as the original loss name.
		tf.scalar_summary(l.op.name +' (raw)', l)
		tf.scalar_summary(l.op.name, loss_averages.average(l))

	return loss_averages_op


def train(total_loss, global_step):
	# Variables that affect learning rate.
	num_batches_per_epoch = input_data.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
	decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

	# Decay the learning rate exponentially based on the number of steps.
	lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE, global_step, decay_steps, LEARNING_RATE_DECAY_FACTOR, staircase=True)
	tf.scalar_summary('learning_rate', lr)

	# Generate moving averages of all losses and associated summaries.
	loss_averages_op = _add_loss_summaries(total_loss)

	# Compute gradients.
	with tf.control_dependencies([loss_averages_op]):
		opt = tf.train.AdamOptimizer (1e-4)
		#opt = tf.train.GradientDescentOptimizer(lr)
		#opt = tf.train.MomentumOptimizer(lr, 0.9)
		grads = opt.compute_gradients(total_loss)

	# Apply gradients.
	apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

	# Add histograms for trainable variables.
	for var in tf.trainable_variables():
		tf.histogram_summary(var.op.name, var)

	# Add histograms for gradients.
	for grad, var in grads:
		if grad is not None:
			tf.histogram_summary(var.op.name + '/gradients', grad)

	# Track the moving averages of all trainable variables.
	variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
	variables_averages_op = variable_averages.apply(tf.trainable_variables())

	with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
		train_op = tf.no_op(name='train')

	return train_op

