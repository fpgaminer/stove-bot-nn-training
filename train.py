from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time

import numpy as np
from six.moves import xrange
import tensorflow as tf
from tensorflow.python import control_flow_ops

import conv_model_1 as m
import input_data

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string ('train_dir', 'traindata',
                            """Directory where to write event logs """
                            """and checkpoint.""")
tf.app.flags.DEFINE_integer ('max_steps', 1000000,
                             """Number of batches to run.""")
tf.app.flags.DEFINE_boolean ('log_device_placement', False,
                             """Whether to log device placement.""")


def train():
	with tf.Graph ().as_default ():
		phase_train = tf.placeholder (tf.bool, name='phase_train')
		global_step = tf.Variable (0, trainable=False, name='global_step')

		# Inputs
		train_image_batch, train_label_batch = input_data.distorted_inputs ()
		val_image_batch, val_label_batch = input_data.inputs (True)
		image_batch, label_batch = control_flow_ops.cond (phase_train,
			lambda: (train_image_batch, train_label_batch),
			lambda: (val_image_batch, val_label_batch))

		# Model
		logits = m.inference (image_batch, phase_train)

		# Loss
		loss, cross_entropy_mean = m.loss (logits, label_batch)

		# Training
		train_op = m.train(loss, global_step)

		# Saver
		saver = tf.train.Saver(tf.all_variables())

		# Session
		sess = tf.Session (config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement))

		# Summary
		summary_op = tf.merge_all_summaries()
		summary_writer = tf.train.SummaryWriter (FLAGS.train_dir, graph=sess.graph)

		# Init
		init_op = tf.initialize_all_variables()
		print ('Initializing...')
		sess.run (init_op, {phase_train.name: True})

		# Start the queue runners
		tf.train.start_queue_runners (sess=sess)

		# Training loop
		print ('Training...')

		for step in xrange(FLAGS.max_steps):
			fetches = [train_op, loss, cross_entropy_mean]
			if step > 0 and step % 100 == 0:
				fetches += [summary_op]

			start_time = time.time ()
			sess_outputs = sess.run (fetches, {phase_train.name: True})
			duration = time.time () - start_time

			loss_value, cross_entropy_value = sess_outputs[1:3]

			if step % 10 == 0:
				num_examples_per_step = FLAGS.batch_size
				examples_per_sec = num_examples_per_step / duration
				sec_per_batch = float(duration)

				format_str = ('%s: step %d, loss = %.2f (%.4f) (%.1f examples/sec; %.3f sec/batch)')
				print (format_str % (datetime.now(), step, loss_value, cross_entropy_value, examples_per_sec, sec_per_batch))
			
			# Summary
			if step > 0 and step % 100 == 0:
				summary_str = sess_outputs[3]
				summary_writer.add_summary (summary_str, step)

			# Validation
			if step > 0 and step % 1000 == 0:
				n_val_samples = 10000
				val_batch_size = FLAGS.batch_size
				n_val_batch = int (n_val_samples / val_batch_size)
				val_logits = np.zeros ((n_val_samples, 2), dtype=np.float32)
				val_labels = np.zeros ((n_val_samples), dtype=np.int64)
				val_losses = []

				for i in xrange (n_val_batch):
					session_outputs = sess.run ([logits, label_batch, loss], {phase_train.name: False})
					val_logits[i*val_batch_size:(i+1)*val_batch_size, :] = session_outputs[0]
					val_labels[i*val_batch_size:(i+1)*val_batch_size] = session_outputs[1]
					val_losses.append (session_outputs[2])

				pred_labels = np.argmax (val_logits, axis=1)
				val_accuracy = np.count_nonzero (pred_labels == val_labels) / (n_val_batch * val_batch_size)
				val_loss = float (np.mean (np.asarray (val_losses)))
				print ('Test accuracy = %f' % val_accuracy)
				print ('Test loss = %f' % val_loss)
				val_summary = tf.Summary ()
				val_summary.value.add (tag='val_accuracy', simple_value=val_accuracy)
				val_summary.value.add (tag='val_loss', simple_value=val_loss)
				summary_writer.add_summary (val_summary, step)


			# Save variables
			if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
				checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
				saver.save(sess, checkpoint_path, global_step=step)


def main (argv=None):
	if tf.gfile.Exists (FLAGS.train_dir):
		tf.gfile.DeleteRecursively (FLAGS.train_dir)
	tf.gfile.MakeDirs (FLAGS.train_dir)
	train ()


if __name__ == '__main__':
	tf.app.run ()
