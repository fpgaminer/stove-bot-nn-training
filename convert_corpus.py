#!/usr/bin/env python
# Convert a corpus of images to a FixedLengthRecord file
# The images will be shuffled within the output
import sys
import os
import random
from PIL import Image


TARGET_SIZE = 128
TEST_RATIO = 0.1


def read_image_to_bytes (filename):
	im = Image.open (filename)
	im.convert ("RGB")
	block_im = Image.new ("RGB", (TARGET_SIZE, TARGET_SIZE), "black")

	if im.size[0] > im.size[1]:
		im = im.resize ((TARGET_SIZE, im.size[1] * TARGET_SIZE / im.size[0]), Image.ANTIALIAS)
		offset = (128 - im.size[1]) / 2
		block_im.paste (im, (0, offset))
	else:
		im = im.resize ((im.size[0] * TARGET_SIZE / im.size[1], TARGET_SIZE), Image.ANTIALIAS)
		offset = (128 - im.size[0]) / 2
		block_im.paste (im, (offset, 0))

	red, green, blue = block_im.split ()
	return red.tostring ("raw") + green.tostring ("raw") + blue.tostring ("raw")


def encode_examples (examples, filename):
	with open (filename, 'wb') as f:
		for example in examples:
			data = read_image_to_bytes (example[1])
			f.write (chr (example[0]))
			f.write (data)


examples = []

for filename in os.listdir ("../corpus/notboiling"):
	examples.append ([0, os.path.join ("../corpus/notboiling", filename)])

for filename in os.listdir ("../corpus/boiling"):
	examples.append ([1, os.path.join ("../corpus/boiling", filename)])

# Shuffle
random.shuffle (examples)

num_tests = int (len (examples) * TEST_RATIO)
test_examples = examples[:num_tests]
train_examples = examples[num_tests:]

# Encode examples
encode_examples (test_examples, 'test_batch.bin')
encode_examples (train_examples, 'train_batch.bin')

print ("Wrote %d total examples, %d training, %d testing\n" % (len (examples), len (train_examples), len (test_examples)))
