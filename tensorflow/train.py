from __future__ import absolute_import, division, print_function

import argparse
import os

import numpy as np
import tensorflow as tf
from torch.utils.data import Dataset

from utilx import *


class ModelNet(Dataset):
    def __init__(self, data_path, split, voxel_size):
        self.data_path = data_path
        self.split = split
        self.voxel_size = voxel_size
        self.data = self.load_data(os.path.join(data_path, '{0}-{1}.txt'.format(split, voxel_size)))

    def load_data(self, data_path):
        data = []
        for line in open(data_path, 'r'):
            model_path, category = line.strip().split()
            data.append((os.path.join(self.data_path, model_path), np.int(category)))
        return data

    def __getitem__(self, index):
        model_path, target = self.data[index]
        data = np.load(model_path).astype(int)
        input = np.zeros((1, self.voxel_size, self.voxel_size, self.voxel_size))
        input[0, data[:, 0], data[:, 1], data[:, 2]] = 1
        return input, target

    def __len__(self):
        return len(self.data)


tf.logging.set_verbosity(tf.logging.INFO)


def cnn_model_fn(features, labels, mode):
    _ = tf.reshape(features['x'], [-1, 28, 28, 1])

    _ = tf.layers.conv2d(
        inputs = _,
        filters = 32,
        kernel_size = [5, 5],
        padding = 'same',
        activation = tf.nn.relu)

    _ = tf.layers.max_pooling2d(inputs = _, pool_size = [2, 2], strides = 2)

    _ = tf.layers.conv2d(
        inputs = _,
        filters = 64,
        kernel_size = [5, 5],
        padding = 'same',
        activation = tf.nn.relu)

    _ = tf.layers.max_pooling2d(inputs = _, pool_size = [2, 2], strides = 2)

    _ = tf.reshape(_, [-1, 7 * 7 * 64])

    _ = tf.layers.dense(inputs = _, units = 1024, activation = tf.nn.relu)

    _ = tf.layers.dropout(
        inputs = _, rate = 0.4, training = mode == tf.estimator.ModeKeys.TRAIN
    )

    _ = tf.layers.dense(inputs = _, units = 10)

    predictions = {
        'classes': tf.argmax(input = _, axis = 1),
        'probabilities': tf.nn.softmax(_, name = 'softmax_tensor')
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode = mode, predictions = predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels = labels, logits = _)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.001)
        train_op = optimizer.minimize(
            loss = loss,
            global_step = tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode = mode, loss = loss, train_op = train_op)

    eval_metric_ops = {
        'accuracy': tf.metrics.accuracy(
            labels = labels, predictions = predictions['classes'])
    }
    return tf.estimator.EstimatorSpec(
        mode = mode, loss = loss, eval_metric_ops = eval_metric_ops)


def main(unused_argv):
    mnist = tf.contrib.learn.datasets.load_dataset('mnist')
    train_data = mnist.train.images  # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype = np.int32)
    eval_data = mnist.test.images  # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype = np.int32)

    mnist_classifier = tf.estimator.Estimator(
        model_fn = cnn_model_fn,
        model_dir = '/tmp/mnist_convnet_model'
    )

    tensors_to_log = {'probabilities': 'softmax_tensor'}
    logging_hook = tf.train.LoggingTensorHook(
        tensors = tensors_to_log,
        every_n_iter = 50
    )

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x = {'x': train_data},
        y = train_labels,
        batch_size = 100,
        num_epochs = None,
        shuffle = True
    )
    mnist_classifier.train(
        input_fn = train_input_fn,
        steps = 20000,
        hooks = [logging_hook]
    )

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x = {'x': eval_data},
        y = eval_labels,
        num_epochs = 1,
        shuffle = False
    )
    eval_results = mnist_classifier.evaluate(input_fn = eval_input_fn)
    print(eval_results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp', default = 'default')
    parser.add_argument('--resume', default = None)
    parser.add_argument('--gpu', default = '0')

    parser.add_argument('--data_path', default = './data/')
    parser.add_argument('--voxel_size', default = 32, type = int)
    parser.add_argument('--workers', default = 8, type = int)
    parser.add_argument('--batch', default = 64, type = int)

    parser.add_argument('--epochs', default = 64, type = int)
    parser.add_argument('--snapshot', default = 1, type = int)
    parser.add_argument('--learning_rate', default = 1e-4, type = float)
    parser.add_argument('--weight_decay', default = 1e-3, type = float)
    parser.add_argument('--step_size', default = 8, type = int)
    parser.add_argument('--gamma', default = 4e-1, type = float)

    args = parser.parse_args()
    print('==> arguments parsed')
    for key in vars(args):
        print('[{0}] = {1}'.format(key, getattr(args, key)))

    args.gpu = set_cuda_visible_devices(args.gpu)

    tf.app.run()
