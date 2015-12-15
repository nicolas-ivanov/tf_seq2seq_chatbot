
"""Linear Algebraic Functions for Unitary Matrices.
These equations come from http://arxiv.org/pdf/1511.06464v2.pdf
This paper is constantly referenced throughout this library"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

print('Unitary Linear has been imported')


#ORIGINAL FUNCTION
def times_diag(input, n_hidden, diag): #convert this to tensorflow....
    input_re = input[:, :n_hidden]
    input_im = input[:, n_hidden:]
    Re = T.nlinalg.AllocDiag()(T.cos(diag)) #I think this will allocate a square matrix mtrix
    Im = T.nlinalg.AllocDiag()(T.sin(diag))
    input_re_times_Re = T.dot(input_re, Re)
    input_re_times_Im = T.dot(input_re, Im)
    input_im_times_Re = T.dot(input_im, Re)
    input_im_times_Im = T.dot(input_im, Im)

    return T.concatenate([input_re_times_Re - input_im_times_Im,
                          input_re_times_Im + input_im_times_Re], axis=1)


'''Note, the equation for this comes from Section 3 of complex paper, first bullet point

Dj,j = e^iw_j 

To accomplish this task, they use Euler's formula instead of the exponent

e^ix = cos(x) + i sin(x)'''



#CONVERTED TENSORFLOW
def times_diag_tf(input_matrix, n_hidden, diag):
    input_re = input_matrix[:, :n_hidden] #okay so the first left half of the matrix is real numbers
    input_im = input_matrix[:, n_hidden:] #the right half is the imaginary numbers that correspond
    Re = tf.diag(tf.cos(diag))
    Im = tf.diag(tf.sin(diag))
    input_re_times_Re = tf.matmul(input_re, Re) #matmul is the equivalent of dot
    input_re_times_Im = tf.matmul(input_re, Im)
    input_im_times_Re = tf.matmul(input_im, Re)
    input_im_times_Im = tf.matmul(input_im, Im)

    return tf.concat(1, [input_re_times_Re - input_im_times_Im,
                          input_re_times_Im + input_im_times_Re]) #this will combine two matrixes

    #nick, this concatenate at the end is the equation number 7 i think....
    #in the future, see if these can be done in one step and skip the concatenation



'''-----------------------------------NEXT FUNCTION TO WORK ON------------------'''
#ORIGINAL FUNCTION
def times_reflection(input, n_hidden, reflection):
    input_re = input[:, :n_hidden]
    input_im = input[:, n_hidden:]
    reflect_re = reflection[:n_hidden]
    reflect_im = reflection[n_hidden:]

    vstarv = (reflect_re**2 + reflect_im**2).sum()


    input_re_reflect = input_re - 2 / vstarv * (T.outer(T.dot(input_re, reflect_re), reflect_re) 
                                                + T.outer(T.dot(input_re, reflect_im), reflect_im) 
                                                - T.outer(T.dot(input_im, reflect_im), reflect_re) 
                                                + T.outer(T.dot(input_im, reflect_re), reflect_im))
    input_im_reflect = input_im - 2 / vstarv * (T.outer(T.dot(input_im, reflect_re), reflect_re) 
                                                + T.outer(T.dot(input_im, reflect_im), reflect_im) 
                                                + T.outer(T.dot(input_re, reflect_im), reflect_re) 
                                                - T.outer(T.dot(input_re, reflect_re), reflect_im))

    return T.concatenate([input_re_reflect, input_im_reflect], axis=1)   



#TF CONVERTED

'''Note, the equation for this comes from Section 3, second bullet point

R = I - (2vv*/||v||2)'''



def times_reflection_tf(input, n_hidden, reflection):
    input_re = input[:, :n_hidden]
    input_im = input[:, n_hidden:]
    reflect_re = reflection[:n_hidden]
    reflect_im = reflection[n_hidden:]

    vstarv = (reflect_re**2 + reflect_im**2).sum() #not sure where all of this is coming from

    vstarv = tf.add(tf.square(reflect_re) + tf.square(reflect_im)) #this might need to be add_n...i don't know

    #i think this mkaes a unitary matrix -- the vstarv
    input_re_reflect = input_re - 2 / vstarv * (T.outer(T.dot(input_re, reflect_re), reflect_re) 
                                                + T.outer(T.dot(input_re, reflect_im), reflect_im) 
                                                - T.outer(T.dot(input_im, reflect_im), reflect_re) 
                                                + T.outer(T.dot(input_im, reflect_re), reflect_im))
    input_im_reflect = input_im - 2 / vstarv * (T.outer(T.dot(input_im, reflect_re), reflect_re) 
                                                + T.outer(T.dot(input_im, reflect_im), reflect_im) 
                                                + T.outer(T.dot(input_re, reflect_im), reflect_re) 
                                                - T.outer(T.dot(input_re, reflect_re), reflect_im))

    return tf.concat(1, [input_re_reflect, input_im_reflect])   





#ORIGINAL FUNCTION
def vec_permutation(input, n_hidden, index_permute):
    re = input[:, :n_hidden]
    im = input[:, n_hidden:]
    re_permute = re[:, index_permute]
    im_permute = im[:, index_permute]

    return T.concatenate([re_permute, im_permute], axis=1)   

'''section three bullet 3 -- 
II, a fixed random index permutation matrix

A permutation matrix consists of one's and zero's. 
http://mathworld.wolfram.com/PermutationMatrix.html

'''

#TF CONVERTED FUNCTION
def vec_permutation_tf(input, n_hidden, index_permute): #I don't get this...why do we do this?
    re = input[:, :n_hidden]
    im = input[:, n_hidden:]
    re_permute = re[:, index_permute] #this part means you keep the batch size and choose one index to permuate? 
    im_permute = im[:, index_permute]

    return tf.concat(1, [re_permute, im_permute])   







def unitary_linear(args, output_size, bias, bias_start=0.0, scope=None):
  """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

  Args:
    args: a 2D Tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    bias: boolean, whether to add a bias term or not.
    bias_start: starting value to initialize the bias; 0 by default.
    scope: VariableScope for the created subgraph; defaults to "Linear".

  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.

  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """
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
  with tf.variable_scope(scope or "Unitary_Linear"):
    matrix = tf.get_variable("Unitary_Matrix", [total_arg_size, output_size])
    if len(args) == 1:
      res = tf.matmul(args[0], matrix)
    else:
      res = tf.matmul(tf.concat(1, args), matrix)
    if not bias:
      return res
    bias_term = tf.get_variable("Unitary_Bias", [output_size],
                                initializer=tf.constant_initializer(bias_start))
  return res + bias_term


  '''Nick, the matrix variable tf I think is your weight matrix'''