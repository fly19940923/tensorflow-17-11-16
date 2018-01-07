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

classes = ['plane', 'car', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck']
num_classes = len(classes)
samples_per_class = 7




# Preprocessing1: reshape the image data into rows
X_train = np.reshape(X_train, (X_train.shape[0], -1))    # (49000,3072)
#
X_test = np.reshape(X_test, (X_test.shape[0], -1))       # (10000,3072)
print(X_test.shape,'Xtest')

# Preprocessing2: subtract the mean image
mean_image = np.mean(X_train, axis=0)       # (1,3072)
X_train -= mean_image

X_test -= mean_image




# Use the validation set to tune hyperparameters (regularization strength
# and learning rate).
learning_rates = [1e-7, 5e-5]
regularization_strengths = [5e4, 1e5]
results = {}
best_val = -1    # The highest validation accuracy that we have seen so far.
best_svm = None   # The LinearSVM object that achieved the highest validation rate.
iters = 15000
losshistory = np.inf
for lr in learning_rates:
    for rs in regularization_strengths:
        svm = Softmax()
        losshistory1 = svm.train(X_train, y_train, learning_rate=lr, reg=rs, num_iters=iters)


        print('loss',losshistory1[-1])
        if losshistory1[-1] < losshistory:
            losshistory = losshistory1[-1]
            best_svm = svm

# print results



# Evaluate the best svm on test set
Ts_pred = best_svm.predict(X_test)
test_accuracy = np.mean(y_test == Ts_pred)     # around 37.1%
print ('LinearSVM on raw pixels of CIFAR-10 final test set accuracy: %f' % test_accuracy)

