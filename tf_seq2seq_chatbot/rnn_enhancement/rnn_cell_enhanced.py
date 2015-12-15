
"""Module for constructing RNN Cells. -- Mostly taken from tensorflow"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf


from tf_seq2seq_chatbot.rnn_enhancement import linear_enhanced as linear


#Warning commenting the two lines below allows it to work!
from tf_seq2seq_chatbot.rnn_enhancement import linear_functions_enhanced as lfe
from tf_seq2seq_chatbot.rnn_enhancement import unitary_linear

with tf.variable_scope("Skip_Connections"):
  timestep_counter = tf.Variable(1, trainable = False, name = "timestep_counter")
  previous_inputs = tf.Variable(0, trainable = False, name = "previous_inputs")
  previous_hidden_states = tf.Variable(0, trainable = False, name = "previous_inputs")


class RNNCell(object):
  """Abstract object representing an RNN cell.

  An RNN cell, in the most abstract setting, is anything that has
  a state -- a vector of floats of size self.state_size -- and performs some
  operation that takes inputs of size self.input_size. This operation
  results in an output of size self.output_size and a new state.

  This module provides a number of basic commonly used RNN cells, such as
  LSTM (Long Short Term Memory) or GRU (Gated Recurrent Unit), and a number
  of operators that allow add dropouts, projections, or embeddings for inputs.
  Constructing multi-layer cells is supported by a super-class, MultiRNNCell,
  defined later. Every RNNCell must have the properties below and and
  implement __call__ with the following signature.
  """

  def __call__(self, inputs, state, scope=None):
    """Run this RNN cell on inputs, starting from the given state.

    Args:
      inputs: 2D Tensor with shape [batch_size x self.input_size].
      state: 2D Tensor with shape [batch_size x self.state_size].
      scope: VariableScope for the created subgraph; defaults to class name.

    Returns:
      A pair containing:
      - Output: A 2D Tensor with shape [batch_size x self.output_size] #output is the first variable?
      - New state: A 2D Tensor with shape [batch_size x self.state_size] #new state is the second variable?.
    """
    raise NotImplementedError("Abstract method")

  @property
  def input_size(self):
    """Integer: size of inputs accepted by this cell."""
    raise NotImplementedError("Abstract method")

  @property
  def output_size(self):
    """Integer: size of outputs produced by this cell."""
    raise NotImplementedError("Abstract method")

  @property
  def state_size(self):
    """Integer: size of state used by this cell."""
    raise NotImplementedError("Abstract method")

  def zero_state(self, batch_size, dtype): #this might be really useful for the idenity rnn...
    """Return state tensor (shape [batch_size x state_size]) filled with 0.

    Args:
      batch_size: int, float, or unit Tensor representing the batch size.
      dtype: the data type to use for the state.

    Returns:
      A 2D Tensor of shape [batch_size x state_size] filled with zeros.
    """
    zeros = tf.zeros(tf.pack([batch_size, self.state_size]), dtype=dtype) #tf.pack converts list to numpy matrix
    zeros.set_shape([None, self.state_size])
    return zeros


class BasicRNNCell(RNNCell):
  """The most basic RNN cell. Tanh activation"""

  def __init__(self, num_units, gpu_for_layer, weight_initializer = "uniform_unit"):
    self._num_units = num_units
    self._gpu_for_layer = gpu_for_layer 
    self._weight_initializer = weight_initializer


  @property
  def input_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  @property
  def state_size(self):
    return self._num_units

  def __call__(self, inputs, state, scope=None):
    """Most basic RNN: output = new_state = tanh(W * input + U * state + B)."""
    with tf.device("/gpu:"+str(self._gpu_for_layer)):
      with tf.variable_scope(scope or type(self).__name__):  # "BasicRNNCell"
        output = tf.tanh(lfe.linear_enhanced([inputs, state], self._num_units, True, weight_initializer = self._weight_initializerF ))
      return output, output


class UnitaryRNNCell(RNNCell):
  """Unitary RNN from Paper: http://arxiv.org/pdf/1511.06464v1.pdf"""

  def __init__(self, num_units, gpu_for_layer = 0, weight_initializer = "uniform_unit"):
    self._num_units = num_units
    self._gpu_for_layer = gpu_for_layer 
    self._weight_initializer = weight_initializer
  

  @property #why is the input and output size the num_units? 
  def input_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  @property
  def state_size(self):
    return self._num_units


  '''Design Structure of Unitary RNN

  The Unitary RNN consists of the following matrices (or parameters):
  1. V_re = A Real Matrix for Hidden weights -- n_input x n_hidden
  2. V_im = An imaginary matrix for hidden weights -- n_input x n_hidden -- V is the input matrix
  3. reflection = A single reflection matrix 
  4. U = Unitary matrix -- 2*n_hidden x n_output -- This is the Output Matrix
  5. Hidden bias -- these are initialized at 0
  6. Out_bias -- these are also initialized at 0 
  7. theta = phase -- 3 x n_hidden (i think its three, one for U, V_re, V_im)
  8. h_0 = bucket uniform initilzation -- 1 x 2*n_hidden
  9. scale = matrix of ones for scaling -- n_hidden


  ------------Initialization-----------
  U and V are initialized with glorot uniform

  Notice here that R1 and R2 are reflections. Uniform initialized from -1 to 1. 

  D1, D2, D3 (diagonals) are sampled from a uniform [-pi,pi]

  h_0 -- buket uniform initialized

  The Weight matrix is Unitary, meaning UU* = U*U = I where U* is the complex conjugate
  Keep in mind for matmul, you must have the same number columns in m1 as rows in m2

  Most memory consumption with RNNs is O(MNT) for hidden unit activations -- so as long as they are the same size, you're cool

  Typical batch_size for seq2seq is 128 x 120 timesteps = ~15000

  In terms of wall time, uRNN step is a bit slower than the regular LSTM. They have some optimization changes that make it 4x faster???
  but they are not on github. 

  In the paper, they used RMSprop as an optimizer with 0.001 learing rate'''



  def __call__(self, inputs, state, scope=None):
    with tf.device("/gpu:"+str(self._gpu_for_layer)):
      print('testing')
      with tf.variable_scope(scope or type(self).__name__):  # "UnitaryRNNCell"
        with tf.variable_scope("UnitaryGates"):  # Reset gate and update gate.


          '''just for sake of consistency, we'll keep some var names the same as authors'''

          n_hidden = self._num_units
          h_prev = state


          '''development nick version here'''
          step1 = unitary_linear.times_diag_tf(h_prev, n_hidden) #this will create a diagonal tensor with given diagonal values


          #work on times_reflection next



          modulus = T.sqrt(lin_output_re ** 2 + lin_output_im ** 2)
          rescale = T.maximum(modulus + hidden_bias.dimshuffle('x',0), 0.) / (modulus + 1e-5)
          nonlin_output_re = lin_output_re * rescale
          nonlin_output_im = lin_output_im * rescale

          h_t = tf.concat(1, [nonlin_output_re, 
                             nonlin_output_im]) 

          #keep in mind that you can use tf.complex to convert two numbers into a complex number -- this works for tensors!

          return h_t, h_t #check if h_t is the same as the output?????


          '''list of complex number functions in tf

          1. tf.complex -- makes complex number
          2. complex_abs -- finds the absolute value of the tensor
          3. tf.conj -- makes conjugate
          4. tf.imag -- returns imaginary part -- go back and forth between complex and imag
          5. tf.real -- returns real part'''

          #keep in mind that identity matricies are a form of diagonal matricies, but they just have ones.


          '''----------------------------end of unitary rnn cell--------------------------'''


          # We start with bias of 1.0 to not reset and not update.
          '''First, we will start with the hidden linear transform
          W = D3R2F-1D2PermR1FD1

          Keep in mind that originally the equation would be W = VDV*, but it leads to too much computation/memory o(n^2)'''
          step1 = times_diag(h_prev, n_hidden, theta[0,:])
          step2 = step1
  #        step2 = do_fft(step1, n_hidden)
          step3 = times_reflection(step2, n_hidden, reflection[0,:])
          step4 = vec_permutation(step3, n_hidden, index_permute)
          step5 = times_diag(step4, n_hidden, theta[1,:])
          step6 = step5
  #        step6 = do_ifft(step5, n_hidden)
          step7 = times_reflection(step6, n_hidden, reflection[1,:])
          step8 = times_diag(step7, n_hidden, theta[2,:])     
          step9 = scale_diag(step8, n_hidden, scale)

          hidden_lin_output = step9

          z = tf.sigmoid(linear.linear([inputs], 
                            self._num_units, True, 1.0))

          '''equation 2 r = sigm(WxrXt+Whrht+Br), h_t is the previous state'''

          r = tf.sigmoid((linear.linear([inputs,state],
                            self._num_units, True, 1.0)))
          '''equation 3'''
        with tf.variable_scope("Candidate"):
          component_0 = linear.linear([r*state],
                            self._num_units, True)
          component_1 = tf.tanh(tf.tanh(inputs) + component_0)
          component_2 = component_1*z
          component_3 = state*(1 - z)

          h_t = component_2 + component_3

          h_t = tf.concat(concat_dim = 1, value =[nonlin_output_re, nonlin_output_im]) #I know here you need to concatenate the real and imaginary parts


        return h_t, h_t #there is only one hidden state output to keep track of. 
        #This makes it more mem efficient than LSTM


class JZS1Cell(RNNCell):
  """Mutant 1 of the following paper: http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf"""

  def __init__(self, num_units, gpu_for_layer = 0, weight_initializer = "uniform_unit"):
    self._num_units = num_units
    self._gpu_for_layer = gpu_for_layer 
    self._weight_initializer = weight_initializer

  @property
  def input_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  @property
  def state_size(self):
    return self._num_units

  def __call__(self, inputs, state, scope=None):
    with tf.device("/gpu:"+str(self._gpu_for_layer)):
      """JZS1, mutant 1 with n units cells."""
      with tf.variable_scope(scope or type(self).__name__):  # "JZS1Cell"
        with tf.variable_scope("Zinput"):  # Reset gate and update gate.
          # We start with bias of 1.0 to not reset and not update.
          '''equation 1 z = sigm(WxzXt+Bz), x_t is inputs'''

          z = tf.sigmoid(lfe.enhanced_linear([inputs], 
                            self._num_units, True, 1.0, weight_initializer = self._weight_initializer)) 

        with tf.variable_scope("Rinput"):
          '''equation 2 r = sigm(WxrXt+Whrht+Br), h_t is the previous state'''

          r = tf.sigmoid(lfe.enhanced_linear([inputs,state],
                            self._num_units, True, 1.0, weight_initializer = self._weight_initializer))
          '''equation 3'''
        with tf.variable_scope("Candidate"):
          component_0 = linear.linear([r*state], 
                            self._num_units, True) 
          component_1 = tf.tanh(tf.tanh(inputs) + component_0)
          component_2 = component_1*z
          component_3 = state*(1 - z)

        h_t = component_2 + component_3

      return h_t, h_t #there is only one hidden state output to keep track of. 
      #This makes it more mem efficient than LSTM


class JZS2Cell(RNNCell):
  """Mutant 2 of the following paper: http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf"""

  def __init__(self, num_units, gpu_for_layer = 0, weight_initializer = "uniform_unit"):
    self._num_units = num_units
    self._gpu_for_layer = gpu_for_layer 
    self._weight_initializer = weight_initializer

  @property
  def input_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  @property
  def state_size(self):
    return self._num_units

  def __call__(self, inputs, state, scope=None):
    with tf.device("/gpu:"+str(self._gpu_for_layer)):
      """JZS2, mutant 2 with n units cells."""
      with tf.variable_scope(scope or type(self).__name__):  # "JZS1Cell"
        with tf.variable_scope("Zinput"):  # Reset gate and update gate.
          '''equation 1'''

          z = tf.sigmoid(lfe.enhanced_linear([inputs, state], 
                            self._num_units, True, 1.0, weight_initializer = self._weight_initializer))

          '''equation 2 '''
        with tf.variable_scope("Rinput"):
          r = tf.sigmoid(inputs+(lfe.enhanced_linear([state],
                            self._num_units, True, 1.0, weight_initializer = self._weight_initializer)))
          '''equation 3'''

        with tf.variable_scope("Candidate"):

          component_0 = linear.linear([state*r,inputs],
                            self._num_units, True)
          
          component_2 = (tf.tanh(component_0))*z
          component_3 = state*(1 - z)

        h_t = component_2 + component_3

      return h_t, h_t #there is only one hidden state output to keep track of. 
        #This makes it more mem efficient than LSTM

class JZS3Cell(RNNCell):
  """Mutant 3 of the following paper: http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf"""

  def __init__(self, num_units, gpu_for_layer = 0, weight_initializer = "uniform_unit"):
    self._num_units = num_units
    self._gpu_for_layer = gpu_for_layer 
    self._weight_initializer = weight_initializer

  @property
  def input_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  @property
  def state_size(self):
    return self._num_units

  def __call__(self, inputs, state, scope=None):
    with tf.device("/gpu:"+str(self._gpu_for_layer)):
      """JZS3, mutant 2 with n units cells."""
      with tf.variable_scope(scope or type(self).__name__):  # "JZS1Cell"
        with tf.variable_scope("Zinput"):  # Reset gate and update gate.
          # We start with bias of 1.0 to not reset and not update.
          '''equation 1'''

          z = tf.sigmoid(lfe.enhanced_linear([inputs, tf.tanh(state)], 
                            self._num_units, True, 1.0, weight_initializer = self._weight_initializer))

          '''equation 2'''
        with tf.variable_scope("Rinput"):
          r = tf.sigmoid(lfe.enhanced_linear([inputs, state],
                            self._num_units, True, 1.0, weight_initializer = self._weight_initializer))
          '''equation 3'''
        with tf.variable_scope("Candidate"):
          component_0 = linear.linear([state*r,inputs],
                            self._num_units, True)
          
          component_2 = (tf.tanh(component_0))*z
          component_3 = state*(1 - z)

        h_t = component_2 + component_3

      return h_t, h_t #there is only one hidden state output to keep track of. 
      #This makes it more mem efficient than LSTM



class GRUCell(RNNCell):
  """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078)."""

  def __init__(self, num_units, gpu_for_layer = 0, weight_initializer = "uniform_unit",
    skip_connections = False, skip_neuron_number = 4):
    self._num_units = num_units
    self._gpu_for_layer = gpu_for_layer 
    self._weight_initializer = weight_initializer
    self._skip_connections = skip_connections
    self._skip_neuron_number = skip_neuron_number


  @property
  def input_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  @property
  def state_size(self):
    return self._num_units

  def __call__(self, inputs, state,scope=None):
    with tf.device("/gpu:"+str(self._gpu_for_layer)):

      '''Modifying skip connections part -- Added Additional Input

      Nick, in the future, you can also add an additional hidden value input as well!'''
      if self._skip_connections:
        with tf.variable_scope("Skip_Connections"):
          timestep_counter.assign(timestep_counter+1) #add one to timestep counter
          print('for testing, you added one to the timestep_counter')
          if tf.add_n(previous_inputs) == 0:
            if previous_inputs.shape == 1:
              previous_inputs.assign(tf.zeros(tf.shape(inputs)))

          '''you have modified the gru network to incorporate the previous inputs'''
          with tf.variable_scope(scope or type(self).__name__):  # "GRUCell"
            with tf.variable_scope("Gates"):  # Reset gate and update gate.
              # We start with bias of 1.0 to not reset and not udpate.
              r, u = tf.split(1, 3, lfe.enhanced_linear([inputs, state, previous_inputs],
                                                  3 * self._num_units, True, 1.0, weight_initializer = self._weight_initializer))
              r, u, pr = tf.sigmoid(r), tf.sigmoid(u), tf.sigmoid(pr)
            with tf.variable_scope("Candidate"): #you need a different one because you're doing a new linear
              #notice they have the activation/non-linear step right here! 
              c = tf.tanh(linear.linear([inputs, r * state, pr*state], self._num_units, True))
            new_h = u * state + (1 - u) * c

          '''need to update inputs if they are available'''  
          if timestep_counter/skip_neuron_number == 0:
            previous_inputs.assign(inputs)
            print('you changed the previous inputs')
            # previous_hidden_states.assign(new_h) #only activate if you need this 

          return new_h, new_h

          

      else:
        """Normal Gated recurrent unit (GRU) with nunits cells."""
        with tf.variable_scope(scope or type(self).__name__):  # "GRUCell"
          with tf.variable_scope("Gates"):  # Reset gate and update gate.
            # We start with bias of 1.0 to not reset and not udpate.
            r, u = tf.split(1, 2, lfe.enhanced_linear([inputs, state],
                                                2 * self._num_units, True, 1.0, weight_initializer = self._weight_initializer))
            r, u = tf.sigmoid(r), tf.sigmoid(u)
          with tf.variable_scope("Candidate"): #you need a different one because you're doing a new linear
            #notice they have the activation/non-linear step right here! 
            c = tf.tanh(linear.linear([inputs, r * state], self._num_units, True))
          new_h = u * state + (1 - u) * c
        return new_h, new_h



class BasicLSTMCell(RNNCell):
  """Basic LSTM recurrent network cell.

  The implementation is based on: http://arxiv.org/pdf/1409.2329v5.pdf.

  It does not allow cell clipping, a projection layer, and does not
  use peep-hole connections: it is the basic baseline.

  Biases of the forget gate are initialized by default to 1 in order to reduce
  the scale of forgetting in the beginning of the training.
  """

  def __init__(self, num_units, gpu_for_layer = 0, weight_initializer = "uniform_unit"):
    self._num_units = num_units
    self._gpu_for_layer = gpu_for_layer 
    self._weight_initializer = weight_initializer


  @property
  def input_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  @property
  def state_size(self):
    return 2 * self._num_units

  def __call__(self, inputs, state, scope=None):
    with tf.device("/gpu:"+str(self._gpu_for_layer)):
      """Long short-term memory cell (LSTM)."""
      with tf.variable_scope(scope or type(self).__name__):  # "BasicLSTMCell"
        # Parameters of gates are concatenated into one multiply for efficiency.
        c, h = tf.split(1, 2, state)
        concat = lfe.enhanced_linear([inputs, h], 4 * self._num_units, True, weight_initializer = self._weight_initializer)

        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        i, j, f, o = tf.split(1, 4, concat)

        new_c = c * tf.sigmoid(f + self._forget_bias) + tf.sigmoid(i) * tf.tanh(j)
        new_h = tf.tanh(new_c) * tf.sigmoid(o)

      return new_h, tf.concat(1, [new_c, new_h])

      '''important, the second part is the hidden state!, thus a lstm with n cells had a hidden state of dimenson 2n'''

      #in the basic lstm, the output and the hidden state are different!


class LSTMCell(RNNCell):
  """Long short-term memory unit (LSTM) recurrent network cell.

  This implementation is based on:

    https://research.google.com/pubs/archive/43905.pdf

  Hasim Sak, Andrew Senior, and Francoise Beaufays.
  "Long short-term memory recurrent neural network architectures for
   large scale acoustic modeling." INTERSPEECH, 2014.

  It uses peep-hole connections, optional cell clipping, and an optional
  projection layer.
  """

  def __init__(self, num_units, input_size,
               use_peepholes=False, cell_clip=None,
               initializer=None, num_proj=None,
               num_unit_shards=1, num_proj_shards=1,
               gpu_for_layer = 0, weight_initializer = "uniform_unit"):
    """Initialize the parameters for an LSTM cell.

    Args:
      num_units: int, The number of units in the LSTM cell
      input_size: int, The dimensionality of the inputs into the LSTM cell
      use_peepholes: bool, set True to enable diagonal/peephole connections.
      cell_clip: (optional) A float value, if provided the cell state is clipped
        by this value prior to the cell output activation.
      initializer: (optional) The initializer to use for the weight and
        projection matrices.
      num_proj: (optional) int, The output dimensionality for the projection
        matrices.  If None, no projection is performed.
      num_unit_shards: How to split the weight matrix.  If >1, the weight
        matrix is stored across num_unit_shards.
        Note that num_unit_shards must evenly divide num_units * 4.
      num_proj_shards: How to split the projection matrix.  If >1, the
        projection matrix is stored across num_proj_shards.
        Note that num_proj_shards must evenly divide num_proj
              (if num_proj is not None).

    Raises:
      ValueError: if num_unit_shards doesn't divide 4 * num_units or
        num_proj_shards doesn't divide num_proj
    """
    self._num_units = num_units
    self._input_size = input_size
    self._use_peepholes = use_peepholes
    self._cell_clip = cell_clip
    self._initializer = initializer
    self._num_proj = num_proj
    self._num_unit_shards = num_unit_shards
    self._num_proj_shards = num_proj_shards
    self._gpu_for_layer = gpu_for_layer 
    self._weight_initializer = weight_initializer

    if (num_units * 4) % num_unit_shards != 0:
      raise ValueError("num_unit_shards must evently divide 4 * num_units")
    if num_proj and num_proj % num_proj_shards != 0:
      raise ValueError("num_proj_shards must evently divide num_proj")

    if num_proj:
      self._state_size = num_units + num_proj
      self._output_size = num_proj
    else:
      self._state_size = 2 * num_units
      self._output_size = num_units

  @property
  def input_size(self):
    return self._input_size

  @property
  def output_size(self):
    return self._output_size

  @property
  def state_size(self):
    return self._state_size

  def __call__(self, input_, state, scope=None):
    
      with tf.device("/gpu:"+str(self._gpu_for_layer)):

        """Run one step of LSTM.

        Args:
          input_: input Tensor, 2D, batch x num_units.
          state: state Tensor, 2D, batch x state_size.
          scope: VariableScope for the created subgraph; defaults to "LSTMCell".

        Returns:
          A tuple containing:
          - A 2D, batch x output_dim, Tensor representing the output of the LSTM
            after reading "input_" when previous state was "state".
            Here output_dim is:
               num_proj if num_proj was set,
               num_units otherwise.
          - A 2D, batch x state_size, Tensor representing the new state of LSTM
            after reading "input_" when previous state was "state".
        """
        num_proj = self._num_units if self._num_proj is None else self._num_proj

        c_prev = tf.slice(state, [0, 0], [-1, self._num_units])
        m_prev = tf.slice(state, [0, self._num_units], [-1, num_proj])

        dtype = input_.dtype

        unit_shard_size = (4 * self._num_units) // self._num_unit_shards

        with tf.variable_scope(scope or type(self).__name__):  # "LSTMCell"
          w = tf.concat(
              1,
              [tf.get_variable("W_%d" % i,
                               shape=[self.input_size + num_proj, unit_shard_size],
                               initializer=self._initializer,
                               dtype=dtype) for i in xrange(self._num_unit_shards)])

          b = tf.get_variable(
              "B", shape=[4 * self._num_units],
              initializer=tf.zeros_initializer, dtype=dtype)

          # i = input_gate, j = new_input, f = forget_gate, o = output_gate
          cell_inputs = tf.concat(1, [input_, m_prev])
          i, j, f, o = tf.split(1, 4, tf.nn.bias_add(tf.matmul(cell_inputs, w), b))

          # Diagonal connections
          if self._use_peepholes:
            w_f_diag = tf.get_variable(
                "W_F_diag", shape=[self._num_units], dtype=dtype)
            w_i_diag = tf.get_variable(
                "W_I_diag", shape=[self._num_units], dtype=dtype)
            w_o_diag = tf.get_variable(
                "W_O_diag", shape=[self._num_units], dtype=dtype)

          if self._use_peepholes:
            c = (tf.sigmoid(f + 1 + w_f_diag * c_prev) * c_prev +
                 tf.sigmoid(i + w_i_diag * c_prev) * tf.tanh(j))
          else:
            c = (tf.sigmoid(f + 1) * c_prev + tf.sigmoid(i) * tf.tanh(j))

          if self._cell_clip is not None:
            c = tf.clip_by_value(c, -self._cell_clip, self._cell_clip)

          if self._use_peepholes:
            m = tf.sigmoid(o + w_o_diag * c) * tf.tanh(c)
          else:
            m = tf.sigmoid(o) * tf.tanh(c)

          if self._num_proj is not None:
            proj_shard_size = self._num_proj // self._num_proj_shards
            w_proj = tf.concat(
                1,
                [tf.get_variable("W_P_%d" % i,
                                 shape=[self._num_units, proj_shard_size],
                                 initializer=self._initializer,
                                 dtype=dtype)
                 for i in xrange(self._num_proj_shards)])
            # TODO(ebrevdo), use matmulsum
            m = tf.matmul(m, w_proj)

      return m, tf.concat(1, [c, m])      




class IdentityRNNCell(RNNCell):
  """Identity RNN from http://arxiv.org/pdf/1504.00941v2.pdf"""

  '''if you want only short term memory, you can use a small scalar in the initialization of the identity matrix'''

  def __init__(self, num_units, gpu_for_layer):
    self._num_units = num_units
    self._gpu_for_layer = gpu_for_layer 
    self._weight_initializer = weight_initializer


  @property
  def input_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  @property
  def state_size(self):
    return self._num_units

  def __call__(self, inputs, state, scope=None):
    """Most basic RNN: output = new_state = tanh(W * input + U * state + B)."""

    '''we need to separate the matmul's because of the identity matrix configuration'''
    with tf.device("/gpu:"+str(self._gpu_for_layer)):
      with tf.variable_scope(scope or type(self).__name__):  # "IdentityRNNCell"
        with tf.variable_scope("inputs_weights"):
          input_weight_matrix_updated = lfe.linear(lfe.linear([inputs], self._num_units, True, weight_initializer = "constant",
            bias_start = 0.0))
        with tf.variable_scope("state_weights"): #notice that we make an identity matrix for the weights.
          state_weight_matrix_updated = lfe.linear(lfe.linear([state], self._num_units, True, weight_initializer = "identity",
            bias_start = 0.0))

        output = tf.nn.relu(tf.add(input_weight_matrix_updated, state_weight_matrix_updated)) #add them together. 
        
      return output, output



class OutputProjectionWrapper(RNNCell):
  """Operator adding an output projection to the given cell.

  Note: in many cases it may be more efficient to not use this wrapper,
  but instead concatenate the whole sequence of your outputs in time,
  do the projection on this batch-concated sequence, then split it
  if needed or directly feed into a softmax.
  """

  def __init__(self, cell, output_size):
    """Create a cell with output projection.

    Args:
      cell: an RNNCell, a projection to output_size is added to it.
      output_size: integer, the size of the output after projection.

    Raises:
      TypeError: if cell is not an RNNCell.
      ValueError: if output_size is not positive.
    """
    if not isinstance(cell, RNNCell):
      raise TypeError("The parameter cell is not RNNCell.")
    if output_size < 1:
      raise ValueError("Parameter output_size must be > 0: %d." % output_size)
    self._cell = cell
    self._output_size = output_size

  @property
  def input_size(self):
    return self._cell.input_size

  @property
  def output_size(self):
    return self._output_size

  @property
  def state_size(self):
    return self._cell.state_size

  def __call__(self, inputs, state, scope=None):
    """Run the cell and output projection on inputs, starting from state."""
    output, res_state = self._cell(inputs, state)
    # Default scope: "OutputProjectionWrapper"
    with tf.variable_scope(scope or type(self).__name__):
      projected = linear.linear(output, self._output_size, True)
    return projected, res_state


class InputProjectionWrapper(RNNCell):
  """Operator adding an input projection to the given cell.

  Note: in many cases it may be more efficient to not use this wrapper,
  but instead concatenate the whole sequence of your inputs in time,
  do the projection on this batch-concated sequence, then split it.
  """

  def __init__(self, cell, input_size):
    """Create a cell with input projection.

    Args:
      cell: an RNNCell, a projection of inputs is added before it.
      input_size: integer, the size of the inputs before projection.

    Raises:
      TypeError: if cell is not an RNNCell.
      ValueError: if input_size is not positive.
    """
    if not isinstance(cell, RNNCell):
      raise TypeError("The parameter cell is not RNNCell.")
    if input_size < 1:
      raise ValueError("Parameter input_size must be > 0: %d." % input_size)
    self._cell = cell
    self._input_size = input_size

  @property
  def input_size(self):
    return self._input_size

  @property
  def output_size(self):
    return self._cell.output_size

  @property
  def state_size(self):
    return self._cell.state_size

  def __call__(self, inputs, state, scope=None):
    """Run the input projection and then the cell."""
    # Default scope: "InputProjectionWrapper"
    with tf.variable_scope(scope or type(self).__name__):
      projected = linear.linear(inputs, self._cell.input_size, True)
    return self._cell(projected, state)


class DropoutWrapper(RNNCell):
  """Operator adding dropout to inputs and outputs of the given cell."""

  def __init__(self, cell, input_keep_prob=1.0, output_keep_prob=1.0,
               seed=None, gpu_for_layer = 0, weight_initializer = "uniform_unit"):
    """Create a cell with added input and/or output dropout.

    Dropout is never used on the state.

    Args:
      cell: an RNNCell, a projection to output_size is added to it.
      input_keep_prob: unit Tensor or float between 0 and 1, input keep
        probability; if it is float and 1, no input dropout will be added.
      output_keep_prob: unit Tensor or float between 0 and 1, output keep
        probability; if it is float and 1, no output dropout will be added.
      seed: (optional) integer, the randomness seed.

    Raises:
      TypeError: if cell is not an RNNCell.
      ValueError: if keep_prob is not between 0 and 1.
    """
    if not isinstance(cell, RNNCell):
      raise TypeError("The parameter cell is not a RNNCell.")
    if (isinstance(input_keep_prob, float) and
        not (input_keep_prob >= 0.0 and input_keep_prob <= 1.0)):
      raise ValueError("Parameter input_keep_prob must be between 0 and 1: %d"
                       % input_keep_prob)
    if (isinstance(output_keep_prob, float) and
        not (output_keep_prob >= 0.0 and output_keep_prob <= 1.0)):
      raise ValueError("Parameter input_keep_prob must be between 0 and 1: %d"
                       % output_keep_prob)
    self._cell = cell
    self._input_keep_prob = input_keep_prob
    self._output_keep_prob = output_keep_prob
    self._seed = seed
    self._gpu_for_layer = gpu_for_layer 
    self._weight_initializer = weight_initializer

  @property
  def input_size(self):
    return self._cell.input_size

  @property
  def output_size(self):
    return self._cell.output_size

  @property
  def state_size(self):
    return self._cell.state_size

  def __call__(self, inputs, state):
    """Run the cell with the declared dropouts."""
    with tf.device("/gpu:"+str(self._gpu_for_layer)):
      if (not isinstance(self._input_keep_prob, float) or
          self._input_keep_prob < 1):
        inputs = tf.nn.dropout(inputs, self._input_keep_prob, seed=self._seed)
      output, new_state = self._cell(inputs, state)
      if (not isinstance(self._output_keep_prob, float) or
          self._output_keep_prob < 1):
        output = tf.nn.dropout(output, self._output_keep_prob, seed=self._seed)
      return output, new_state


class EmbeddingWrapper(RNNCell):
  """Operator adding input embedding to the given cell.

  Note: in many cases it may be more efficient to not use this wrapper,
  but instead concatenate the whole sequence of your inputs in time,
  do the embedding on this batch-concated sequence, then split it and
  feed into your RNN.
  """

  def __init__(self, cell, embedding_classes=0, embedding=None,
               initializer=None):
    """Create a cell with an added input embedding.

    Args:
      cell: an RNNCell, an embedding will be put before its inputs.
      embedding_classes: integer, how many symbols will be embedded.
      embedding: Variable, the embedding to use; if None, a new embedding
        will be created; if set, then embedding_classes is not required.
      initializer: an initializer to use when creating the embedding;
        if None, the initializer from variable scope or a default one is used.

    Raises:
      TypeError: if cell is not an RNNCell.
      ValueError: if embedding_classes is not positive.
    """
    if not isinstance(cell, RNNCell):
      raise TypeError("The parameter cell is not RNNCell.")
    if embedding_classes < 1 and embedding is None:
      raise ValueError("Pass embedding or embedding_classes must be > 0: %d."
                       % embedding_classes)
    if embedding_classes > 0 and embedding is not None:
      if embedding.size[0] != embedding_classes:
        raise ValueError("You declared embedding_classes=%d but passed an "
                         "embedding for %d classes." % (embedding.size[0],
                                                        embedding_classes))
      if embedding.size[1] != cell.input_size:
        raise ValueError("You passed embedding with output size %d and a cell"
                         " that accepts size %d." % (embedding.size[1],
                                                     cell.input_size))
    self._cell = cell
    self._embedding_classes = embedding_classes
    self._embedding = embedding
    self._initializer = initializer

  @property
  def input_size(self):
    return 1

  @property
  def output_size(self):
    return self._cell.output_size

  @property
  def state_size(self):
    return self._cell.state_size

  def __call__(self, inputs, state, scope=None):
    """Run the cell on embedded inputs."""
    with tf.variable_scope(scope or type(self).__name__):  # "EmbeddingWrapper"
      with tf.device("/cpu:0"):
        if self._embedding:
          embedding = self._embedding
        else:
          if self._initializer:
            initializer = self._initializer
          elif tf.get_variable_scope().initializer:
            initializer = tf.get_variable_scope().initializer
          else:
            # Default initializer for embeddings should have variance=1.
            sqrt3 = math.sqrt(3)  # Uniform(-sqrt(3), sqrt(3)) has variance=1.
            initializer = tf.random_uniform_initializer(-sqrt3, sqrt3)
          embedding = tf.get_variable("embedding", [self._embedding_classes,
                                                    self._cell.input_size],
                                      initializer=initializer)
        embedded = tf.nn.embedding_lookup(embedding, tf.reshape(inputs, [-1]))
    return self._cell(embedded, state)


class MultiRNNCell(RNNCell):
  """RNN cell composed sequentially of multiple simple cells."""

  def __init__(self, cells):
    """Create a RNN cell composed sequentially of a number of RNNCells.

    Args:
      cells: list of RNNCells that will be composed in this order.

    Raises:
      ValueError: if cells is empty (not allowed) or if their sizes don't match.
    """
    if not cells:
      raise ValueError("Must specify at least one cell for MultiRNNCell.")
    for i in xrange(len(cells) - 1):
      if cells[i + 1].input_size != cells[i].output_size:
        raise ValueError("In MultiRNNCell, the input size of each next"
                         " cell must match the output size of the previous one."
                         " Mismatched output size in cell %d." % i)
    self._cells = cells

  @property
  def input_size(self):
    return self._cells[0].input_size

  @property
  def output_size(self):
    return self._cells[-1].output_size

  @property
  def state_size(self):
    return sum([cell.state_size for cell in self._cells])

  def __call__(self, inputs, state, scope=None):
    """Run this multi-layer cell on inputs, starting from state."""
    with tf.variable_scope(scope or type(self).__name__):  # "MultiRNNCell"
      cur_state_pos = 0
      cur_inp = inputs
      new_states = []
      for i, cell in enumerate(self._cells):
        with tf.variable_scope("Cell%d" % i):
          cur_state = tf.slice(state, [0, cur_state_pos], [-1, cell.state_size])
          cur_state_pos += cell.state_size
          cur_inp, new_state = cell(cur_inp, cur_state)
          new_states.append(new_state)
    return cur_inp, tf.concat(1, new_states)