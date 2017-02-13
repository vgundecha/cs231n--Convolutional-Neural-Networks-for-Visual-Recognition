"Image Gradients"

# As usual, a bit of setup

import time, os, json
import numpy as np
import skimage.io
import matplotlib.pyplot as plt

from cs231n.classifiers.pretrained_cnn import PretrainedCNN
from cs231n.data_utils import load_tiny_imagenet
from cs231n.image_utils import blur_image, deprocess_image

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

#Load tiny Imagenet
data = load_tiny_imagenet('cs231n/datasets/tiny-imagenet-100-A', subtract_mean=True)

for i, names in enumerate(data['class_names']):
  print i, ' '.join('"%s"' % name for name in names)
  

#%% Load pre-trained model

model = PretrainedCNN(h5_file='cs231n/datasets/pretrained_model.h5') 

#%% Pretrained model performance

batch_size = 100

# Test the model on training data
mask = np.random.randint(data['X_train'].shape[0], size=batch_size)
X, y = data['X_train'][mask], data['y_train'][mask]
y_pred = model.loss(X).argmax(axis=1)
print 'Training accuracy: ', (y_pred == y).mean()

# Test the model on validation data
mask = np.random.randint(data['X_val'].shape[0], size=batch_size)
X, y = data['X_val'][mask], data['y_val'][mask]
y_pred = model.loss(X).argmax(axis=1)
print 'Validation accuracy: ', (y_pred == y).mean()

#%%

def compute_saliency_maps(X, y, model):
  """
  Compute a class saliency map using the model for images X and labels y.
  
  Input:
  - X: Input images, of shape (N, 3, H, W)
  - y: Labels for X, of shape (N,)
  - model: A PretrainedCNN that will be used to compute the saliency map.
  
  Returns:
  - saliency: An array of shape (N, H, W) giving the saliency maps for the input
    images.
  """
  saliency = None
  ##############################################################################
  # TODO: Implement this function. You should use the forward and backward     #
  # methods of the PretrainedCNN class, and compute gradients with respect to  #
  # the unnormalized class score of the ground-truth classes in y.             #
  ##############################################################################
  N = X.shape[0]  
  out, cache = model.forward(X)
  dout = np.zeros((N, 100))
  dout[range(N), y] = 1
  saliency = np.max(model.backward(dout, cache)[0], axis=1)
  
  return saliency
  
#%%

def show_saliency_maps(mask):
  mask = np.asarray(mask)
  X = data['X_val'][mask]
  y = data['y_val'][mask]

  saliency = compute_saliency_maps(X, y, model)

  for i in xrange(mask.size):
    plt.subplot(2, mask.size, i + 1)
    plt.imshow(deprocess_image(X[i], data['mean_image']))
    plt.axis('off')
    plt.title(data['class_names'][y[i]][0])
    plt.subplot(2, mask.size, mask.size + i + 1)
    plt.title(mask[i])
    plt.imshow(saliency[i])
    plt.axis('off')
  plt.gcf().set_size_inches(10, 4)
  plt.show()

# Show some random images
mask = np.random.randint(data['X_val'].shape[0], size=5)
show_saliency_maps(mask)
  
# These are some cherry-picked images that should give good results
show_saliency_maps([128, 3225, 2417, 1640, 4619])  

#%%

def make_fooling_image(X, target_y, model):
  """
  Generate a fooling image that is close to X, but that the model classifies
  as target_y.
  
  Inputs:
  - X: Input image, of shape (1, 3, 64, 64)
  - target_y: An integer in the range [0, 100)
  - model: A PretrainedCNN
  
  Returns:
  - X_fooling: An image that is close to X, but that is classifed as target_y
    by the model.
  """
  X_fooling = X.copy()
  ##############################################################################
  # TODO: Generate a fooling image X_fooling that the model will classify as   #
  # the class target_y. Use gradient ascent on the target class score, using   #
  # the model.forward method to compute scores and the model.backward method   #
  # to compute image gradients.                                                #
  #                                                                            #
  # HINT: For most examples, you should be able to generate a fooling image    #
  # in fewer than 100 iterations of gradient ascent.                           #
  ##############################################################################
  
  dout = np.zeros((1, 100))
  dout[0,target_y] = 1  
  
  for it in range(100):
      out, cache = model.forward(X_fooling)
      dX, _ = model.backward(dout, cache)
      X_fooling += 1000*dX
  
  return X_fooling
  
  
#%%

# Find a correctly classified validation image
while True:
  i = np.random.randint(data['X_val'].shape[0])
  X = data['X_val'][i:i+1]
  y = data['y_val'][i:i+1]
  y_pred = model.loss(X)[0].argmax()
  if y_pred == y: break

target_y = 67
X_fooling = make_fooling_image(X, target_y, model)

# Make sure that X_fooling is classified as y_target
scores = model.loss(X_fooling)
assert scores[0].argmax() == target_y, 'The network is not fooled!'

# Show original image, fooling image, and difference
plt.subplot(1, 3, 1)
plt.imshow(deprocess_image(X, data['mean_image']))
plt.axis('off')
plt.title(data['class_names'][y][0])
plt.subplot(1, 3, 2)
plt.imshow(deprocess_image(X_fooling, data['mean_image'], renorm=True))
plt.title(data['class_names'][target_y][0])
plt.axis('off')
plt.subplot(1, 3, 3)
plt.title('Difference')
plt.imshow(deprocess_image(X - X_fooling, data['mean_image']))
plt.axis('off')
plt.show()  