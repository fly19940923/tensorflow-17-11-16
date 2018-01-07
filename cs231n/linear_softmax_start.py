import numpy as np
import matplotlib.pyplot as plt
import math
from cs231n.classifiers.linear_classifier import *
from cs231n.data_utils import load_CIFAR10

cifar10_dir = "G:\\work\\tensorflow\\cs231n\\datasets\\cifar-10-batches-py"

X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

print ('Training data shape: ', X_train.shape)     # (50000,32,32,3)
print ('Training labels shape: ', y_train.shape)   # (50000L,)
print ('Test data shape: ', X_test.shape)          # (10000,32,32,3)
print ('Test labels shape: ', y_test.shape)        # (10000L,)

X_train = np.reshape(X_train, (X_train.shape[0], -1))    # (49000,3072)
#
X_test = np.reshape(X_test, (X_test.shape[0], -1))       # (1000,3072)
