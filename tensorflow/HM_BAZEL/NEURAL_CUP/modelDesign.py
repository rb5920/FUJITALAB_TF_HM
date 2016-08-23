from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
from operator import itemgetter
from numpy import*
import os
import tensorflow as tf
import datetime
import csv,sys

def welcomemymodel(trainingdata, hidden1_units, hidden2_units, hidden3_units, hidden4_units, hidden5_units, inputnum, NUM_CLASSES):
  # Hidden 1
  with tf.name_scope('hidden1'):
    weights = tf.Variable(tf.truncated_normal([inputnum, hidden1_units],
                            stddev=1.0 / math.sqrt(float(inputnum))),name='weights')
    biases = tf.Variable(tf.zeros([hidden1_units]),name='biases')
    hidden1 = tf.nn.relu(tf.matmul(trainingdata, weights) + biases)
  # Hidden 2
  with tf.name_scope('hidden2'):
    weights = tf.Variable(tf.truncated_normal([hidden1_units, hidden2_units],
                            stddev=1.0 / math.sqrt(float(hidden1_units))),name='weights')
    biases = tf.Variable(tf.zeros([hidden2_units]),name='biases')
    hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
  # Hidden 3
  with tf.name_scope('hidden3'):
    weights = tf.Variable(tf.truncated_normal([hidden2_units, hidden3_units],
                            stddev=1.0 / math.sqrt(float(hidden2_units))),name='weights')
    biases = tf.Variable(tf.zeros([hidden3_units]),name='biases')
    hidden3 = tf.nn.relu(tf.matmul(hidden2, weights) + biases)
  # Hidden 4
  with tf.name_scope('hidden4'):
    weights = tf.Variable(tf.truncated_normal([hidden3_units, hidden4_units],
                            stddev=1.0 / math.sqrt(float(hidden3_units))),name='weights')
    biases = tf.Variable(tf.zeros([hidden4_units]),name='biases')
    hidden4 = tf.nn.relu(tf.matmul(hidden3, weights) + biases)
  # Hidden 5
  with tf.name_scope('hidden5'):
    weights = tf.Variable(tf.truncated_normal([hidden4_units, hidden5_units],
                            stddev=1.0 / math.sqrt(float(hidden4_units))),name='weights')
    biases = tf.Variable(tf.zeros([hidden5_units]),name='biases')
    hidden5 = tf.nn.relu(tf.matmul(hidden4, weights) + biases)
  # Linear
  with tf.name_scope('softmax_linear'):
    weights = tf.Variable(tf.truncated_normal([hidden5_units, NUM_CLASSES],
                            stddev=1.0 / math.sqrt(float(hidden5_units))),name='weights')
    biases = tf.Variable(tf.zeros([NUM_CLASSES]),name='biases')
    logits = tf.add(tf.matmul(hidden5, weights) , biases, name='logits')
    tf.add_to_collection("logits",logits)
  return logits
def loss(logits, labels):
  labels = tf.to_int64(labels)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels, name='xentropy')
  loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
  return loss
def training(loss, learning_rate):
  # Add a scalar summary for the snapshot loss.
  tf.scalar_summary(loss.op.name, loss)
  # Create the gradient descent optimizer with the given learning rate.
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  # Create a variable to track the global step.
  global_step = tf.Variable(0, name='global_step', trainable=False)
  # Use the optimizer to apply the gradients that minimize the loss
  # (and also increment the global step counter) as a single training step.
  train_op = optimizer.minimize(loss, global_step=global_step)
  return train_op
def evaluation(logits, labels):
  # For a classifier model, we can use the in_top_k Op.
  # It returns a bool tensor with shape [batch_size] that is true for
  # the examples where the label is in the top k (here k=1)
  # of all logits for that example.
  correct = tf.nn.in_top_k(logits, labels, 1)
  # Return the number of true entries.
  return tf.reduce_sum(tf.cast(correct, tf.int32))
