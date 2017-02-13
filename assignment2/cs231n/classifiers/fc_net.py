import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network with ReLU nonlinearity and
  softmax loss that uses a modular layer design. We assume an input dimension
  of D, a hidden dimension of H, and perform classification over C classes.
  
  The architecure should be affine - relu - affine - softmax.

  Note that this class does not implement gradient descent; instead, it
  will interact with a separate Solver object that is responsible for running
  optimization.

  The learnable parameters of the model are stored in the dictionary
  self.params that maps parameter names to numpy arrays.
  """
  
  def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
               weight_scale=1e-3, reg=0.0):
    """
    Initialize a new network.

    Inputs:
    - input_dim: An integer giving the size of the input
    - hidden_dim: An integer giving the size of the hidden layer
    - num_classes: An integer giving the number of classes to classify
    - dropout: Scalar between 0 and 1 giving dropout strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - reg: Scalar giving L2 regularization strength.
    """
    self.params = {}
    self.reg = reg
    
    self.params['W1'] = weight_scale*np.random.randn(input_dim, hidden_dim)
    self.params['b1'] = np.zeros(hidden_dim)
    self.params['W2'] = weight_scale*np.random.randn(hidden_dim, num_classes)
    self.params['b2'] = np.zeros(num_classes)

  def loss(self, X, y=None):
    """
    Compute loss and gradient for a minibatch of data.

    Inputs:
    - X: Array of input data of shape (N, d_1, ..., d_k)
    - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

    Returns:
    If y is None, then run a test-time forward pass of the model and return:
    - scores: Array of shape (N, C) giving classification scores, where
      scores[i, c] is the classification score for X[i] and class c.

    If y is not None, then run a training-time forward and backward pass and
    return a tuple of:
    - loss: Scalar value giving the loss
    - grads: Dictionary with the same keys as self.params, mapping parameter
      names to gradients of the loss with respect to those parameters.
    """  
    scores = None
    activations, cache_1 = affine_relu_forward(X, self.params['W1'], self.params['b1'])
    scores, cache_2 = affine_forward(activations, self.params['W2'], self.params['b2'])
    
    # If y is None then we are in test mode so just return scores
    if y is None:
      return scores
    
    loss, grads = 0, {}
    data_loss, grad = softmax_loss(scores,y)
    loss = data_loss  + 0.5*self.reg*(np.sum(self.params['W1']**2) + np.sum(self.params['W2']**2))
    dout, grads['W2'],  grads['b2'] = affine_backward(grad, cache_2)
    _, grads['W1'], grads['b1'] = affine_relu_backward(dout, cache_1)
    
    grads['W2'] += self.reg*self.params['W2']
    grads['W1'] += self.reg*self.params['W1']
    return loss, grads

#%%
class FullyConnectedNet(object):
  """
  A fully-connected neural network with an arbitrary number of hidden layers,
  ReLU nonlinearities, and a softmax loss function. This will also implement
  dropout and batch normalization as options. For a network with L layers,
  the architecture will be
  
  {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax
  
  where batch normalization and dropout are optional, and the {...} block is
  repeated L - 1 times.
  
  Similar to the TwoLayerNet above, learnable parameters are stored in the
  self.params dictionary and will be learned using the Solver class.
  """

  def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
               dropout=0, use_batchnorm=False, reg=0.0,
               weight_scale=1e-2, dtype=np.float32, seed=None):
    """
    Initialize a new FullyConnectedNet.
    
    Inputs:
    - hidden_dims: A list of integers giving the size of each hidden layer.
    - input_dim: An integer giving the size of the input.
    - num_classes: An integer giving the number of classes to classify.
    - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
      the network should not use dropout at all.
    - use_batchnorm: Whether or not the network should use batch normalization.
    - reg: Scalar giving L2 regularization strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - dtype: A numpy datatype object; all computations will be performed using
      this datatype. float32 is faster but less accurate, so you should use
      float64 for numeric gradient checking.
    - seed: If not None, then pass this random seed to the dropout layers. This
      will make the dropout layers deteriminstic so we can gradient check the
      model.
    """
    self.use_batchnorm = use_batchnorm
    self.use_dropout = dropout > 0
    self.reg = reg
    self.num_layers = 1 + len(hidden_dims)
    self.dtype = dtype
    self.params = {}
    layer_dimensions = [input_dim]
    layer_dimensions.extend(hidden_dims)
    layer_dimensions.extend([num_classes])               
                                                                           
    # When using batch normalization, store scale and shift parameters for the #
    # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
    # beta2, etc. Scale parameters should be initialized to one and shift      #
    # parameters should be initialized to zero.     

    #Initialize the parameters of the network
    for i in range(0,self.num_layers):
       self.params['W'+str(i+1)] = weight_scale*np.random.randn(layer_dimensions[i], layer_dimensions[i+1])
       self.params['b'+str(i+1)] = np.zeros(layer_dimensions[i+1])
       if self.use_batchnorm == True and i<self.num_layers-1:
           self.params['gamma'+str(i+1)] = np.ones((1,layer_dimensions[i+1]))
           self.params['beta'+str(i+1)] = np.zeros((1,layer_dimensions[i+1]))
    
    # When using dropout we need to pass a dropout_param dictionary to each
    # dropout layer so that the layer knows the dropout probability and the mode
    # (train / test). You can pass the same dropout_param to each dropout layer.
    self.dropout_param = {}
    if self.use_dropout:
      self.dropout_param = {'mode': 'train', 'p': dropout}
      if seed is not None:
        self.dropout_param['seed'] = seed
    
    # With batch normalization we need to keep track of running means and
    # variances, so we need to pass a special bn_param object to each batch
    # normalization layer. You should pass self.bn_params[0] to the forward pass
    # of the first batch normalization layer, self.bn_params[1] to the forward
    # pass of the second batch normalization layer, etc.
    self.bn_params = []
    if self.use_batchnorm:
      self.bn_params = [{'mode': 'train'} for i in xrange(self.num_layers - 1)]
    
    # Cast all parameters to the correct datatype
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Compute loss and gradient for the fully-connected net.

    Input / output: Same as TwoLayerNet above.
    """
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'

    # Set train/test mode for batchnorm params and dropout param since they
    # behave differently during training and testing.
    if self.dropout_param is not None:
      self.dropout_param['mode'] = mode   
    if self.use_batchnorm:
      for bn_param in self.bn_params:
        bn_param[mode] = mode

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the fully-connected net, computing  #
    # the class scores for X and storing them in the scores variable.          #
    #                                                                          #
    # When using dropout, you'll need to pass self.dropout_param to each       #
    # dropout forward pass.                                                    #
    #                                                                          #
    # When using batch normalization, you'll need to pass self.bn_params[0] to #
    # the forward pass for the first batch normalization layer, pass           #
    # self.bn_params[1] to the forward pass for the second batch normalization #
    # layer, etc.                                                              #
    
    cache = [None]*self.num_layers
    dropout_cache = [None]*(self.num_layers-1)
    activations = X
    
    #Forward pass 
    for i in range(self.num_layers-1):
       if self.use_batchnorm==False:
           activations, cache[i] = affine_relu_forward(activations, 
                                                        self.params['W'+str(i+1)],
                                                         self.params['b'+str(i+1)])
                                                         
           
           
       else:
           activations, cache[i] = affine_bnorm_relu_forward(activations, 
                                                              self.params['W'+str(i+1)], self.params['b'+str(i+1)],
                                                               self.params['gamma'+str(i+1)],
                                                                self.params['beta'+str(i+1)], self.bn_params[i])
                                                                
       if self.use_dropout==True:
           activations, dropout_cache[i] = dropout_forward(activations, self.dropout_param)
                                                             
           
    #Forward pass for affine layer before Softmax   
    scores, cache[self.num_layers-1] = affine_forward(activations, 
                                                       self.params['W'+str(self.num_layers)],
                                                        self.params['b'+str(self.num_layers)]) 
    
    # If test mode return early
    if mode == 'test':
      return scores
     
    #Compute loss 
    loss, grads = 0.0, {}
    data_loss, grad = softmax_loss(scores,y)
    regularization_loss = 0
    for i in range(self.num_layers):
       regularization_loss += 0.5*self.reg*np.sum(self.params['W'+str(i+1)]**2)
       
    loss = data_loss + regularization_loss
    
    ############################################################################
    # TODO: Implement the backward pass for the fully-connected net. Store the #
    # loss in the loss variable and gradients in the grads dictionary. Compute #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    #                                                                          #
    # When using batch normalization, you don't need to regularize the scale   #
    # and shift parameters.                                                    #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    
    #Backward pass for layer before Softmax
    dout, grads['W'+str(self.num_layers)], grads['b'+str(self.num_layers)] = affine_backward(grad, cache[self.num_layers-1])    
    grads['W'+str(self.num_layers)] += self.reg*self.params['W'+str(self.num_layers)]
          
    #Backward pass for remaining layers      
    for i in range(self.num_layers-1, 0, -1):
       
       if self.use_dropout==True:
           dout = dropout_backward(dout, dropout_cache[i-1])
           
       if self.use_batchnorm==False:
           dout, grads['W'+str(i)], grads['b'+str(i)] = affine_relu_backward(dout, cache[i-1])
           grads['W'+str(i)] += self.reg*self.params['W'+str(i)]
           
       else:
           dout, grads['W'+str(i)], grads['b'+str(i)], grads['gamma' + str(i)], grads['beta' + str(i)] = affine_bnorm_relu_backward(dout, cache[i-1])
           grads['W'+str(i)] += self.reg*self.params['W'+str(i)]
           

        
               
    
    return loss, grads
     
    
def affine_bnorm_relu_forward(x, w, b, gamma, beta, bn_param):
  """
  Convenience layer that performs affine-bnorm-relu 
  forward pass
  """
  out_affine, fc_cache = affine_forward(x, w, b)
  out_bnorm, bnorm_cache = batchnorm_forward(out_affine, gamma, beta, bn_param)
  out, relu_cache = relu_forward(out_bnorm)      
  cache = (fc_cache, bnorm_cache, relu_cache)
  return out, cache
  
def affine_bnorm_relu_backward(dout, cache):
  """
  Convenience layer that performs affine-bnorm-relu
  backward pass
  """
  
  fc_cache, bnorm_cache, relu_cache = cache
  dout_relu = relu_backward(dout, relu_cache)
  dout_bnorm, dgamma, dbeta = batchnorm_backward(dout_relu, bnorm_cache)
  dout_affine, dw, db = affine_backward(dout_bnorm, fc_cache)
  return dout_affine, dw, db, dgamma, dbeta
    
      
      
      
      
     
