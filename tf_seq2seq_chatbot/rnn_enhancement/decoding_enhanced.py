
"""Library for decoding functions"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from six.moves import xrange  # pylint: disable=redefined-builtin


import tensorflow as tf

print('You are using the decoding_enhanced library!')

def sample_with_temperature(a, temperature=1.0):
	'''this function takes logits input, and produces a specific number from the array.
	As you increase the temperature, you will get more diversified output but with more errors

	args: 
	Logits -- this must be a 1d array
	Temperature -- how much variance you want in output

	returns:
	Selected number from distribution
	'''

	'''
	Equation can be found here: https://en.wikipedia.org/wiki/Softmax_function (under reinforcement learning)

        Karpathy did it here as well: https://github.com/karpathy/char-rnn/blob/4297a9bf69726823d944ad971555e91204f12ca8/sample.lua
        '''
	with tf.op_scope(a+temperature, "sample_with_temperature"):
		a = np.squeeze(a)/temperature #start by reduction of temperature
		
		exponent_raised = np.exp(a) #this makes the temperature much more effective and gets rid of negative numbers. 

		probs = exponent_raised / np.sum(exponent_raised) #this will make everything add up to 100% 

		#get rid of any negative numbers in the probabilities -- they shouldn't be in here anyways
		probs = probs.clip(0)

		#reduce the sum for rounding errors
		subtracting_factor = 0.002/probs.shape[0]

		probs = probs - subtracting_factor

		multinomial_part = np.random.multinomial(1, probs, 1)

		final_number = int(np.argmax(multinomial_part))

	return final_number

def batch_sample_with_temperature(a, temperature=1.0):
	'''this function is like sample_with_temperature except it can handle batch input a of [batch_size x logits] 
		this function takes logits input, and produces a specific number from the array. This is all done on the gpu
		because this function uses tensorflow
		As you increase the temperature, you will get more diversified output but with more errors (usually gramatical if you're 
			doing text)
	args: 
		Logits -- this must be a 2d array [batch_size x logits]
		Temperature -- how much variance you want in output
	returns:
		Selected number from distribution
	'''

	'''
	Equation can be found here: https://en.wikipedia.org/wiki/Softmax_function (under reinforcement learning)
        Karpathy did it here as well: https://github.com/karpathy/char-rnn/blob/4297a9bf69726823d944ad971555e91204f12ca8/sample.lua'''
	'''a is [batch_size x logits]'''
	with tf.op_scope(a+temperature, "batch_sample_with_temperature"):
		
		exponent_raised = tf.exp(tf.div(a, temperature)) #start by reduction of temperature, and get rid of negative numbers with exponent
		
		matrix_X = tf.div(exponent_raised, tf.reduce_sum(exponent_raised, reduction_indices = 1)) #this will yield probabilities!

		matrix_U = tf.random_uniform([batch_size, tf.shape(a)[1]], minval = 0, maxval = 1)

		final_number = tf.argmax(tf.sub(matrix_X - matrix_U), dimension = 1) #you want dimension = 1 because you are argmaxing across rows.

	return final_number







'''Cost Functions as defined by http://arxiv.org/pdf/1510.03055v1.pdf to encourage diversity within generated content'''

def G_i_piecewise_variance(batch_size, total_timesteps, gamma = 5):
	'''returns g(i) function for U(T) for diversity paper
	Input:
		total_timesteps: number of total timesteps made
		gamma: the number that you stop switch gamma to 0
	Output:
		Value of G which is either a 0 or a 1
		'''
	with tf.op_scope(batch_size+total_timesteps + gamma, "G_i_piecewise_variance"):
		
		#make a list of the tensor you want

		#convert this list into a tensor

		tf.concat(1, [ones_matrix, zeros_matrix])
		#it might be better to just concatenate two separate 0's and 1's matrices
		tf.constant


def U_t_variance(timestep_outputs_matrix, total_timesteps, gamma = 5):

	with tf.op_scope(timestep_outputs_matrix + total_timesteps + gamma, "U_t_variance"):

		G_i_matrix = G_i_piecewise_variance(timestep_outputs_matrix, total_timesteps)
		tf.mul(timestep_outputs_matrix, )
		tf.reduce_prod(timestep_outputs_matrix_with_g)



def pTS_minus_LamdaUT_cost():

	'''This function can be implemented at each timestep! 
	You do not have to wait for your whole sequence to be generated, so lets do this one first.


	Score(T) = p(T|S) - Lamda*U(T) + Gamma*L_t

	where:
	L_t is the length of the sequence generated
	Gamma is the amount that length affects the Score

	note that L_t is linearly combined with ultimate score for the target T

	Then there is the function g(x) that is piecewise'''



	print('testing')

#need to make sure you take sentence length into account. 

'''This cost function is linearly combined with the length penalization, gamma, and L_t is the length of the sentence

The softmax cross_entropy cost function will be substituted with the maximum mutual information

these pairs are chosne betwen p(S,T)/p(S)p(T)

Normally a softmax is calculated by the multiplication of series of wrong numbers -- this is denoted by the cross_entropy cost. 

the | in p(a|b) means A given b -- conditional probability

P(a|b) = P(a ^ b) / P(b)

if p(b) is undefined, then the p(a|b) is undefined

lamda is a hyperparameter that controls how much to penalize generic responses


'''



def pTS_plus_LamdapST_cost():
	print('testing')