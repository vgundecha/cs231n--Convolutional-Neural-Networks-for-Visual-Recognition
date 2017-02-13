import numpy as np
from random import shuffle


def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

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
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_classes = W.shape[1]
  num_train = X.shape[0]    
  raw_scores = X.dot(W) - np.max(X.dot(W),axis=1)[:,np.newaxis] #normalized
  scores = np.zeros_like(raw_scores)

  #Compute the loss
  for i in xrange(num_train):
      for j in xrange(num_classes):
          scores[i,j] = np.exp(raw_scores[i,j])
      scores[i,:] = scores[i,:]/np.sum(scores[i,:])
      #loss += -np.log(scores[i,y[i]]/np.sum(scores[i,:]))
      loss += -np.log(scores[i,y[i]])
  loss /= num_train   
  loss += 0.5*reg*np.sum(W*W) #regularization
  
  #Compute the gradient
  for i in xrange(num_train):
      for j in xrange(num_classes):
          if j==y[i]:
              dW[:,j] += (scores[i,j] - 1)*X[i,:]
          else:
              dW[:,j] += scores[i,j]*X[i,:]
              
  dW /= num_train
  dW += 2*reg*W
  
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)  
  num_train = X.shape[0]
  num_classes = W.shape[1]  
  
 
  # Compute loss
  raw_scores = X.dot(W) - np.max(X.dot(W),axis=1)[:,np.newaxis] #normalized
  scores = np.exp(raw_scores)/(np.sum(np.exp(raw_scores),axis=1)[:,np.newaxis])
  loss = np.sum(-np.log(scores[range(num_train),y]))
  loss /= num_train
  
  # Add regularization ot the loss
  loss += 0.5*reg*np.sum(W**2)
  
  # Compute gradient
  class_vector = digit_to_vector(y, num_classes)
  dW_equalto_yi = (class_vector.T).dot((scores[range(num_train),y] - 1)[:,np.newaxis]*X)
  scores_mask = scores
  scores_mask[range(num_train),y] = 0;
  dW_notequalto_yi = (scores_mask.T).dot(X)
  dW = ((dW_equalto_yi + dW_notequalto_yi)/num_train).T
  
  #Add regularization ot the gradient
  dW += 2*reg*W
  
  return loss, dW

def digit_to_vector(y, num_classes):
    v = np.zeros((y.shape[0],num_classes))
    v[range(y.shape[0]),y] = 1
    
    return v