# Softmax

import tensorflow as tf

def run():
    output = None
    logdata = [3.0, 2.0, 0.5]
    logits = tf.placeholder(tf.float32)
    
    softmax = tf.nn.softmax(logits)    
    
    with tf.Session() as session:
        output = session.run(softmax, feed_dict={logits: logdata})

    return output
