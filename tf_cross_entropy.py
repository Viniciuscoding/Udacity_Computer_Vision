# Cross Entropy

import tensorflow as tf

softmax_data = [0.8, 0.1, 0.4]
onehot_data = [0.0, 0.1, 0.0]

softmax = tf.placeholder(tf.float32)
onehot = tf.placeholder(tf.float32)

crossentropy = -tf.reduce_sum(tf.multiply(onehot, tf.log(softmax)))

with tf.Session() as session:
    print(session.run(crossentropy, feed_dict = {softmax: softmax_data, onehot: one_hot_data}))
