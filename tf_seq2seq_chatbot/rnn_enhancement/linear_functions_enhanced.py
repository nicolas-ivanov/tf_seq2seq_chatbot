"""Library of Linear Algebraic Functions"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

print('Linear functions enhaned has been imported')

def identity_initializer():

    print('Warning -- You have opted to use the identity_initializer for your identity matrix!!!!!!!!!!!!!!!!@@@@@@')
    def _initializer(shape, dtype=tf.float32):
        if len(shape) == 1:
            return tf.constant(0., dtype=dtype, shape=shape)
        elif len(shape) == 2 and shape[0] == shape[1]:
            return tf.constant(np.identity(shape[0], dtype))
        elif len(shape) == 4 and shape[2] == shape[3]:
            array = np.zeros(shape, dtype=float)
            cx, cy = shape[0]/2, shape[1]/2
            for i in range(shape[2]):
                array[cx, cy, i, i] = 1
            return tf.constant(array, dtype=dtype)
        else:
            raise
    return _initializer

def orthogonal_initializer(scale = 1.1):
    ''' From Lasagne and Keras. Reference: Saxe et al., http://arxiv.org/abs/1312.6120
    '''
    print('Warning -- You have opted to use the orthogonal_initializer!!!!!!!!!!!!!!!!@@@@@@')
    def _initializer(shape, dtype=tf.float32):
      flat_shape = (shape[0], np.prod(shape[1:]))
      a = np.random.normal(0.0, 1.0, flat_shape)
      u, _, v = np.linalg.svd(a, full_matrices=False)
      # pick the one with the correct shape
      q = u if u.shape == flat_shape else v
      q = q.reshape(shape)

      return tf.constant(scale * q[:shape[0], :shape[1]])
    return _initializer

def enhanced_linear(args, output_size, bias, bias_start=0.0, weight_initializer = "uniform_unit", orthogonal_scale = 1.1, scope=None):
  """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

  Args:
    args: a 2D Tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    bias: boolean, whether to add a bias term or not.
    bias_start: starting value to initialize the bias; 0 by default.
    scope: VariableScope for the created subgraph; defaults to "Enhanced_Linear".

  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.

  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """


  print('Warning -- you have opted to use enhanced_linear function!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')

  assert args
  if not isinstance(args, (list, tuple)):
    args = [args]

  # Calculate the total size of arguments on dimension 1.
  total_arg_size = 0
  shapes = [a.get_shape().as_list() for a in args]
  for shape in shapes:
    if len(shape) != 2:
      raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
    if not shape[1]:
      raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
    else:
      total_arg_size += shape[1]

  # Now the computation.
  with tf.variable_scope(scope or "Enhanced_Linear"): #in this linear scope, the library that you're retriving is Linear
  #this will make a class for these variables so you can reference them in the future. 

    '''initialize weight matrix properly'''
    if weight_initializer == "uniform_unit":
      matrix = tf.get_variable("Identity_Matrix", [total_arg_size, output_size]) #i think this is retrieving the weight matrix
    elif weight_initializer == "identity":
      matrix = tf.get_variable("Enhanced_Matrix", [total_arg_size, output_size], initializer = identity_initializer()) #fix this when you get a chance for identity?
    elif weight_initializer == "orthogonal":
      matrix = tf.get_variable("Orthogonal_Matrix", [total_arg_size, output_size], initializer = orthogonal_initializer(scale = orthogonal_scale)) #fix this when you get a chance for identity?
    else:
      raise ValueError("weight_initializer not set correctly: %s Initializers: uniform_unit, identity, orthogonal" % weight_initializer)


    #this will create a variable if it hasn't been created yet! we need to make it an identiy matrix?
    if len(args) == 1:
      res = tf.matmul(args[0], matrix) #this is just one matrix to multiply by 
    else:
      res = tf.matmul(tf.concat(1, args), matrix)
    if not bias:
      return res
    bias_term = tf.get_variable("Enhanced_Bias", [output_size],
                                initializer=tf.constant_initializer(bias_start)) #this is retrieving the bias term that you would use
    '''the tf.constant_initializer is used because it makes all one value'''
  return res + bias_term


  '''Nick, the matrix variable tf I think is your weight matrix'''

def euclidean_norm(tensor, reduction_indices = None, name = None):
	with tf.op_scope(tensor + reduction_indices, name, "euclidean_norm"): #need to have this for tf to work
		squareroot_tensor = tf.square(tensor)
		euclidean_norm = tf.sum(squareroot_tensor, reduction_indices =  reduction_indices)
		return euclidean_norm

def frobenius_norm(tensor, reduction_indices = None, name = None):
	with tf.op_scope(tensor + reduction_indices, name, "frobenius_norm"): #need to have this for tf to work
		squareroot_tensor = tf.square(tensor)
		tensor_sum = tf.sum(squareroot_tensor, reduction_indices =  reduction_indices)
		frobenius_norm = tf.sqrt(tensor_sum)
		return frobenius_norm

