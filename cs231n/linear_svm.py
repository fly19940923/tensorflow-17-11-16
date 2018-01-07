import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:, y[i]] += X[i, :]
        dW[:, j] += X[i, :]
  loss /= num_train
  dW /= num_train
  dW += 2 * reg * W
  loss += reg * np.sum(W * W)
  return loss, dW

def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  m,n = X.shape
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  scores = X.dot(W)
  scores_y = scores[xrange(m),y].reshape(m,1)
  margins = scores - scores_y + 1
  #np.argmax(margins,1)
  margins[xrange(m),y] = 0
  margins[margins<=0] = 0.0
  loss += np.sum(margins,1) / m
  loss += reg * np.sum(W * W)

  margins[margins > 0] = 1.0  # 示性函数的意义
  row_sum = np.sum(margins, axis=1)  # 1 by N
  margins[np.arange(m), y] = -row_sum
  dW += np.dot(X.T, margins) / m + reg * W  # D by C

  return loss, dW