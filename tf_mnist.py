import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def get_weights(features, labels):
    # Return Tensorflow weights
    return tf.Variable(tf.truncated_normal((features, labels)))

def get_biases(labels):
    # Return Tensorflow biases
    return tf.Variable(tf.zeros(labels))

def linear(tf_input, tf_weights, tf_biases):
    # Return a linear function result
    # Linear function = xW + b
    # tf.matmul it a matrix multiplication in Tensorflow
    return tf.add(tf.matmul(tf_input, tf_weights), tf_biases)

def mnist_features_labels(labels, n_images = 5000):

    mnist_features, mnist_labels = [], []

    mnist = input_data.read_data_sets('ADD your MNIST image data dir here', one_hot = True)

    for mnist_feature, mnist_label in zip(*mnist.train.next_batch(n_images)):
        # Add features and labels if it's for the first <n>th labels
        if mnist_label[:labels].any():
            mnist_features.append(mnist_feature)
            mnist_labels.append(mnist_label[:labels])
    
    # Return a tf tuple of feature and label list
    return mnist_features, mnist_labels

# FINISH WORKING ON THE MAIN FUNCTION

# 784 features means the image size is 28x28
n_features = 784
n_labels = 3

# Seeting up a float tensorflow varialbe for features and labels
features, labels = tf.placeholder(tf.float32), tf.placeholder(tf.float32)
 
# Weights and Biases
weights = get_weights(n_features, n_labels)
biases = get_biases(n_labels)

# Get the linear function results
logits = linear(features, weights, biases)

# Start training the image data
train_features, train_labels = mnist_features_labels(n_labels)

with tf.Session() as session:
    # Initializing the session variables
    session.run(tf.global_variables_initializer())
    # Apply softmax
    prediction = tf.nn.softmax(logits)

    # Cross entropy
    # This quantifies how far off the predictions were.
    cross_entropy = -tf.reduce_sum(labels * tf.log(prediction), reduction_indices=1)

    # Calculating the training loss
    loss = tf.reduce_mean(cross_entropy)

    # Rate of change of the weights
    learning_rate = 0.08

    # Using Gradient Descent to train the model
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    # Run optimizer and get loss
    _, loss_results = session.run([optimizer, loss], feed_dict={features: train_features, labels: train_labels})

# Print loss
print('Loss: {}'.format(loss_results))
