from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

from tflearn.data_utils import image_preloader

import numpy as np

# X, Y = image_preloader('/Users/abhiagarwal/Desktop/cnn/dataset.txt', image_shape=(28, 28), mode='file', categorical_labels=True, normalize=True, grayscale=True)

import imageio
X = [
    imageio.imread('clocks/%s.jpg' % (i + 1))[:, :, :3]
    for i in range(719)
]

X = np.array(X)

Y = [
    [ 1 if i == j else 0 for i in range(720) ]
    for j in range(720)
]

Y = np.array(Y)

# testX, testY = image_preloader('/Users/abhiagarwal/Desktop/cnn/dataset_test.txt', image_shape=(28, 28), mode='file', categorical_labels=True, normalize=True, grayscale=True)

testX = [
    imageio.imread('test/240.jpg')[:, :, :3],
    imageio.imread('test/600.jpg')[:, :, :3],
]

testX = np.array(testX)

testY = [
    [ 1 if i == 240 else 0 for i in range(720) ],
    [ 1 if i == 600 else 0 for i in range(720) ],
]

testY = np.array(testY)

# Building convolutional network
network = input_data(shape=[None, 128, 128, 3], name='input')
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
network = regression(network, optimizer='adam', learning_rate=0.01,
                     loss='categorical_crossentropy', name='target')

# Training
model = tflearn.DNN(network, tensorboard_verbose=0)
model.fit({'input': X}, {'target': Y}, n_epoch=20,
           validation_set=({'input': testX}, {'target': testY}),
           snapshot_step=100, show_metric=True, run_id='cnn_clocker')