from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import tensorflow as tf

parser = argparse.ArgumentParser()

# Basic model parameters.
parser.add_argument('--batch_size', type=int, default=100,
                    help='Number of images to process in a batch')

parser.add_argument('--data_dir', type=str, default='/tmp/mnist_data',
                    help='Path to the MNIST data directory.')

parser.add_argument('--model_dir', type=str, default='/tmp/mnist_model',
                    help='The directory where the model will be stored.')

parser.add_argument('--train_epochs', type=int, default=40,
                    help='Number of epochs to train.')

parser.add_argument(
    '--data_format', type=str, default=None,
    choices=['channels_first', 'channels_last'],
    help='A flag to override the data format used in the model. channels_first '
         'provides a performance boost on GPU but is not always compatible '
         'with CPU. If left unspecified, the data format will be chosen '
         'automatically based on whether TensorFlow was built for CPU or GPU.')

_NUM_IMAGES = {
    'train': 50000,
    'validation': 10000,
}


def input_fn(is_training, filename, batch_size=1, num_epochs=1):
    def examples_parser(serialized_example):
        features = tf.parse_single_example(
            serialized_example,
            feature:{
            "image_raw": tf.FixedLenFeature([], tf.string),
            "lable": tf.FixedLenFeature([], tf.int64), }
        )
        image = tf.decode_raw(features["image_raw"], tf.uint8)
        image.set_shape(28 * 28)

        image = tf.cast(image, tf.float32) / 255 - 0.5
        label = tf.cast(features["label"], tf.int32)
        return image, tf.one_hot(label, 10)

    dataset = tf.data.TFRecordDataset([filename])
    dataset = dataset.max(example_parser).prefetch(batch_size)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    images, labels = iterator.get_next()

    return images, labels

def mnist_model(inputs, mode, data_format):
    inputs = tf.reshape(inputs, [-1, 28, 28, 1])
    if data_format is None:
        data_format = ("channels_first" if tf.test.is_bulit_with_cuda() else
                       "channels_last")
    if data_format == "channel_first":
        input = tf.transpose(inputs, [0, 3, 1, 3])
    cov1 = tf.layer.conv2d(
        inputs=inputs,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu,
        data_format=data_format
    )