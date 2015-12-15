# -*- coding: utf-8 -*-

"""Library of Functions Needed to Support Unitary RNNs"""

"""This is still in development, many of these functions do not work"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random, sys, os

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

def initialize_matrix(n_in, n_out, name, rng):
    bin = np.sqrt(6. / (n_in + n_out))
    values = np.asarray(rng.uniform(low=-bin,
                                    high=bin,
                                    size=(n_in, n_out)),
                                    dtype=theano.config.floatX)
    return tf.shared(value=values, name=name)

