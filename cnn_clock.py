from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

from tflearn.data_utils import image_preloader

from sklearn.model_selection import train_test_split

import numpy as np
import imageio

data = np.array([
    [
        np.expand_dims(imageio.imread('clocks/%s/%s.jpg' % (j, i)), axis=-1),
        np.array([ 1 if i == k else 0 for k in range(720) ])
    ]
    for i in range(720)
    for j in range(2)
])

X = np.stack(data[:, 0])
y = np.stack(data[:, 1])

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=32)

# Building convolutional network
network = input_data(shape=[None, 128, 128, 1], name='input')
network = conv_2d(network, 32, 3, activation='relu', regularizer="L2")
network = max_pool_2d(network, 2)
network = local_response_normalization(network)
network = conv_2d(network, 64, 3, activation='relu', regularizer="L2")
network = max_pool_2d(network, 2)
network = local_response_normalization(network)
network = fully_connected(network, 128, activation='tanh')
network = dropout(network, 0.8)
network = fully_connected(network, 256, activation='tanh')
network = dropout(network, 0.8)
network = fully_connected(network, 720, activation='softmax')
network = regression(network, optimizer='adam', learning_rate=0.01, loss='categorical_crossentropy', name='target')

# Training
model = tflearn.DNN(network, tensorboard_verbose=1)
model.fit({'input': X_train}, {'target': y_train}, n_epoch=20,
           validation_set=({'input': X_test}, {'target': y_test}),
           snapshot_step=100, show_metric=True, run_id='cnn_clocker')