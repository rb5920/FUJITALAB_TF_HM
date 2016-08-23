# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from operator import itemgetter
from numpy import*
import os
import tensorflow as tf
import datetime
import csv,sys
import DataIMPORT
import modelDesign
#===============================================
# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.0001, 'Initial learning rate.')
flags.DEFINE_integer('batch_size', 20, 'Number of a training set.')
flags.DEFINE_integer('max_steps', 30000000, 'Number of a training set.')
flags.DEFINE_integer('inputnum', 5, 'Number of a training set.')
flags.DEFINE_integer('NUM_CLASSES', 2, 'Number of classes.')
flags.DEFINE_integer('hidden1', 64, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 128, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('hidden3', 128, 'Number of units in hidden layer 3.')
flags.DEFINE_integer('hidden4', 128, 'Number of units in hidden layer 4.')
flags.DEFINE_integer('hidden5', 64, 'Number of units in hidden layer 5.')
flags.DEFINE_string('foldername', 'PUSIZE/', 'Directory to put the training data.')
flags.DEFINE_string('EXPNAME', 'TUPartition_', 'Directory to put the training data.')
flags.DEFINE_string('recordNAME', 'record', 'Directory to put the training data.')
flags.DEFINE_string('testDNAME', 'trainingdata.csv', 'Directory to put the training data.')
#===============================================
def placeholder_inputs(input_num):  
  trainingD_placeholder = tf.placeholder(tf.float32, [None, input_num],name='inputx')
  teachingD_placeholder = tf.placeholder(tf.int32, [None])
  return trainingD_placeholder, teachingD_placeholder
#===============================================
trainingD,teachingD,trainingdata_count=DataIMPORT.read_testdata('','', FLAGS.testDNAME, FLAGS.inputnum)
test_trainingD,test_teachingD,data_count=DataIMPORT.read_testdata('','', FLAGS.testDNAME, FLAGS.inputnum)
#===============================================
eachdatanumtotal=trainingdata_count
index = 0
end_temp = 0
count = 0
listacc=[]
#===============================================
from tensorflow.python.platform import gfile

with tf.Session() as persisted_sess:
	print("load graph")
	with gfile.FastGFile("models/ret.pb",'rb') as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())
		persisted_sess.graph.as_default()
		tf.import_graph_def(graph_def, name='')
	print("map variables")
	
	persisted_result = persisted_sess.graph.get_tensor_by_name("saved_result:0")
	tf.add_to_collection(tf.GraphKeys.VARIABLES,persisted_result)
	saver = tf.train.Saver(tf.all_variables())
	print("load data")
	
	#tf.train.write_graph(persisted_sess.graph_def, 'models/', 'ret.pb', as_text=False)
	
	TimeLabel1 = datetime.datetime.now()
	
	saver.restore(sess, FLAGS.recordNAME)
	TimeLabel2 = datetime.datetime.now()
	#new_y = sess.run([logits], feed_dict={x: reshape(test_trainingD[0],[1,5])})
	new_y = sess.run([logits], feed_dict={x: test_trainingD[0:8]})
	TimeLabel3 = datetime.datetime.now()
	print(new_y)
	print("Accuracy:%g" % (argmax(new_y[0])))


