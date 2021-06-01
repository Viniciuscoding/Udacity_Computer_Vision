# Mini-Batch
""" Definition from Udacity
Mini-batching is a technique for training on subsets of the dataset instead of all the data at one time. This provides the ability to train a model, 
even if a computer lacks the memory to store the entire dataset.

Mini-batching is computationally inefficient, since you can't calculate the loss simultaneously across all samples. However, this is a small price to 
pay in order to be able to run the model at all.

It's also quite useful combined with SGD. The idea is to randomly shuffle the data at the start of each epoch, then create the mini-batches. 
For each mini-batch, you train the network weights with gradient descent. Since these batches are random, you're performing SGD with each batch.
"""

# This code comes from Udacity. All the rights goes for them. I have made slight modifications to it
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import math
# from pprint import pprint



n_input = 784  # This is the size of the image. The MNIST dataset has imgages with 28x28 shape
n_classes = 10  # This means the number of unique digits in MNIST. There are 0-9 unique digits as classes

# Importing the MNIST data from input_data
mnist = input_data.read_data_sets('/datasets/ud730/mnist', one_hot=True)

# The features are already scaled and the data is shuffled
train_features = mnist.train.images
test_features = mnist.test.images

train_labels = mnist.train.labels.astype(np.float32)
test_labels = mnist.test.labels.astype(np.float32)

# Weights & bias
weights = tf.Variable(tf.random_normal([n_input, n_classes]))
bias = tf.Variable(tf.random_normal([n_classes]))

# CALCULATING THE SIZE to TRAIN THE MODEL

# float32 memory size is 3.4028235×10^38

# train_features = Shape: (55000, 784) Type: float32 
    # 55000 * 784 * 4 = 172,480,000 = 172.48 MB

# train_labels = Shape: (55000, 10) Type: float32 
    # 55000 * 10 * 4 = 2,200,000 = 2.2 MB

# weights = Shape: (784, 10) Type: float32 
    # 784 X 10 X 4 = 31,360 = 0.03 MB

# bias = Shape: (10,) Type: float32 
    # 10 X 4 = 40 bytes

# TOTAL = 174.71 MB


# DIVIDING DATA INTO BATCHES
"""
Unfortunately, it's sometimes impossible to divide the data into batches of exactly equal size. For example, imagine you'd like to create batches of 128 samples
each from a dataset of 1000 samples. Since 128 does not evenly divide into 1000, you'd wind up with 7 batches of 128 samples, and 1 batch of 104 samples. 
(7*128 + 1*104 = 1000)

In that case, the size of the batches would vary, so you need to take advantage of TensorFlow's tf.placeholder() function to receive the varying batch sizes.

Continuing the example, if each sample had n_input = 784 features and n_classes = 10 possible labels, the dimensions for features would be [None, n_input] and 
labels would be [None, n_classes].
"""

# Features and Labels
features = tf.placeholder(tf.float32, [None, n_input]) # None means at runtime Tensorflow will accept any batch size greater than 0.
labels = tf.placeholder(tf.float32, [None, n_classes]) # The None dimension is a placeholder for the batch size

# batch_size is 128

# features is (50000, 400)
    # 50000 / 128 + 1 = 391

# labels is (50000, 10)
    # 50000 - 128 * 390 = 80


def batches(batch_size, features, labels):
    """
    Create batches of features and labels
    :param batch_size: The batch size
    :param features: List of features
    :param labels: List of labels
    :return: Batches of (Features, Labels)
    """
    assert len(features) == len(labels)

    output_batches = []
    
    sample_size = len(features)
    for start_i in range(0, sample_size, batch_size):
        end_i = start_i + batch_size
        batch = [features[start_i:end_i], labels[start_i:end_i]]
        output_batches.append(batch)
        
    return output_batches
  
# 4 Samples of features
example_features = [['F11','F12','F13','F14'],
                    ['F21','F22','F23','F24'],
                    ['F31','F32','F33','F34'],
                    ['F41','F42','F43','F44']]
# 4 Samples of labels
example_labels = [['L11','L12'],
                  ['L21','L22'],
                  ['L31','L32'],
                  ['L41','L42']]

# PPrint prints data structures like 2d arrays, so they are easier to read
pprint(batches(3, example_features, example_labels))
