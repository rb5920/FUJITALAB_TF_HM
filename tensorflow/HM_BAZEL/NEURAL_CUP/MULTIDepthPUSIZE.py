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
with tf.Graph().as_default():
	x, y = placeholder_inputs(FLAGS.inputnum)
	tf.add_to_collection("inputx",x)
	# Build a Graph that computes predictions from the inference model.
	logits = modelDesign.welcomemymodel(x, FLAGS.hidden1, FLAGS.hidden2, FLAGS.hidden3, FLAGS.hidden4, FLAGS.hidden5, FLAGS.inputnum, FLAGS.NUM_CLASSES)
	
  	saved_result= tf.Variable(tf.zeros([1,FLAGS.NUM_CLASSES]), name="saved_result")
  	do_save=tf.assign(saved_result,logits)
	tf.add_to_collection("saved_result",saved_result)
    # Add to the Graph the Ops for loss calculation.
	loss = modelDesign.loss(logits, y)

    # Add to the Graph the Ops that calculate and apply gradients.
	train_op = modelDesign.training(loss, FLAGS.learning_rate)

    # Add the Op to compare the logits to the labels during evaluation.
	eval_correct = modelDesign.evaluation(logits, y)

    # Build the summary operation based on the TF collection of Summaries.
	summary_op = tf.merge_all_summaries()

    # Add the variable initializer Op.
	init = tf.initialize_all_variables()

    # Create a saver for writing training checkpoints.
	saver = tf.train.Saver()
	
    # Create a session for running Ops on the Graph.
	sess = tf.Session()
	
	tf.train.write_graph(sess.graph_def, 'models/', 'ret.pb', as_text=False)
    # Instantiate a SummaryWriter to output summaries and the Graph.
	summary_writer = tf.train.SummaryWriter('summ/', sess.graph)

    # And then after everything is built:

    # Run the Op to initialize the variables.
	sess.run(init)
	
	TimeLabel1 = datetime.datetime.now()
	
	for step in range(FLAGS.max_steps):
		start = index
		end_temp += FLAGS.batch_size
		if end_temp > eachdatanumtotal-FLAGS.batch_size:
			perm = arange(eachdatanumtotal)
			random.shuffle(perm)
			trainingD=trainingD[perm]
			teachingD=teachingD[perm]
      		# Start next epoch
			start = 0
			end_temp = FLAGS.batch_size
			assert FLAGS.batch_size <= eachdatanumtotal
		end = end_temp - 1
		feedD = {x: trainingD[start:end], y: teachingD[start:end]}
		_, loss_value = sess.run([train_op, loss], feed_dict=feedD)
		index = end + 1 
		if step % 100 == 0:
			train_accuracy = sess.run(eval_correct, feed_dict={x:trainingD, y: teachingD})   
			train_accuracy = train_accuracy / eachdatanumtotal
			listacc.append(train_accuracy)
 			print('Step %d: loss = %.2f (accuracy = %.4f)' % (step, loss_value, train_accuracy))
        	# Update the events file.
			summary_str = sess.run(summary_op, feed_dict=feedD)
			summary_writer.add_summary(summary_str, step)
			summary_writer.flush()
			if train_accuracy > 0.991:
				count=count+1
				if count > 4:
					break
			else:
				count=0
	saver.save(sess, FLAGS.recordNAME, write_meta_graph=True)
	tf.train.write_graph(sess.graph_def, 'models/', 'ret2.pb', as_text=False)
	TimeLabel2 = datetime.datetime.now()
	acc,new_y = sess.run([eval_correct,logits], feed_dict={x: test_trainingD, y: test_teachingD})
	TimeLabel3 = datetime.datetime.now()
	print("Accuracy:%g" % (acc/data_count))
labeltrue=transpose(array([test_teachingD,argmax(new_y,1)]))
print(labeltrue)
fout=open("ansrecord_data.csv","w")
wout=csv.writer(fout)
wout.writerows(labeltrue)
fout.close()
faccout=open("accrecord_data.csv","w")
waccout=csv.writer(faccout)
dataout=reshape(array(listacc),[-1,1])
#dataout=array(listacc)
print(dataout)
waccout.writerows(dataout)
faccout.close()
print("Training Time=",(TimeLabel2-TimeLabel1).seconds,"Testing Time=",(TimeLabel3-TimeLabel2).seconds)


