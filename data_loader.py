# data_loader.py

import tensorflow as tf

def load_data():
    # Load the MNIST dataset, which contains handwritten digit images and labels
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Normalize pixel values from range [0, 255] to [0, 1] for faster convergence during training
    x_train, x_test = x_train / 255.0, x_test / 255.0
    return (x_train, y_train), (x_test, y_test)
