#Stove Bot
###Neural Network Training

This repository contains the TensorFlow based code for training [Stove Bot's](https://hackaday.io/project/12685-stove-bot) neural network.  To use it you need the Stove Bot corpus and TensorFlow.  A GPU is not strictly needed, but it helps tremendously.

Execute `convert_corpus.py` to generate `data/train_batch.bin` and `data/test_batch.bin` from the corpus, which should be located at `../corpus`.

Execute `train.py` to train the network.  Progress is output to stdout, but you should also use `tensorboard` (included with TensorFlow) to track progress.  It takes about 40k batches to converge with the current setup.

Execute `eval.py` to evaluate the trained model for accuracy and loss.  This should be slightly better than the validation performed by `train.py` because `eval.py` will average the logits from three different crops for each evaluation, improving loss/accuracy.
