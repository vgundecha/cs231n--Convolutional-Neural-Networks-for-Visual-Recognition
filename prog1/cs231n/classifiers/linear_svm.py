import numpy as np
from random import shuffle
import inspect

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
          
        dW[:,j] += X[i]
        dW[:,y[i]] += -X[i]
        loss += margin
        
  # Right now the loss and gradient is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train
  

  # Add regularization to the loss and gradient.
  loss += 0.5 * reg * np.sum(W * W)
  dW += 2*reg*W
  
  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  num_train = X.shape[0]
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  
  scores = X.dot(W)
  correct_class_score  = scores[range(X.shape[0]),y]
  scores_svm = scores - correct_class_score[:,np.newaxis] + 1
  mask = scores_svm > 0
  loss = np.sum(scores_svm*mask) - num_train
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  
  """Compute gradient, vectorized implementation"""
  
  mask[range(X.shape[0]),y] = False
  dW_not_equalto_yi = np.dot(np.transpose(mask),X) 
  
  sum_margins = np.sum(mask,1)
  scale_inputs = -(sum_margins[:,np.newaxis]*X)
  digit_to_vector = np.zeros((num_train,W.shape[1]))
  digit_to_vector[range(num_train),y] = 1
  dW_equalto_yi  = np.dot(np.transpose(digit_to_vector),scale_inputs)
  
  dW = dW_equalto_yi + dW_not_equalto_yi
  dW = np.transpose(dW)
  
  dW /= num_train
  dW += 2*reg*W #regularization term
  
  return  loss, dW
