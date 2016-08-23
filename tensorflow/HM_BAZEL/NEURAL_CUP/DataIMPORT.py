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
def transform_data(dataarray,datanum,inputnum):
	trainingD=empty([datanum,inputnum])
	teachingD=empty([datanum])
	for i in range(datanum):
		for j in range(inputnum+1):
			if j!= inputnum :
				trainingD[i][j]=int(dataarray[i][j])
			else:
				if int(dataarray[i][j])==0: 
					teachingD[i]=0	
				elif int(dataarray[i][j])==1: 
					teachingD[i]=1
	return trainingD,teachingD
	
def read_testdata(foldername,EXPNAME,FILEN,inputnum):
	testdata_count = 0
	list_test=[]
	filename = foldername+EXPNAME+FILEN
	fpData = open(filename,'rb')
	reader = csv.reader(fpData)
	try:
		for line in reader:
			list_test.append(line)
			testdata_count=testdata_count+1
	except csv.Error, e:
		print('EOF')
	fpData.close()
	OutputA,OutputB=transform_data(array(list_test),testdata_count,inputnum)
	return OutputA,OutputB,testdata_count
def read_trainingdata(foldername,EXPNAME,inputnum):
	trainingdata_count = 0
	list_training=[]
	filename = foldername+EXPNAME+'bus.csv'
	fpData = open(filename,'rb')
	reader = csv.reader(fpData)
	try:
		for line in reader:
			list_training.append(line)
			trainingdata_count=trainingdata_count+1
	except csv.Error, e:
		print('EOF')
	fpData.close()
	filename = foldername+EXPNAME+'container.csv'
	fpData = open(filename,'rb')
	reader = csv.reader(fpData)
	try:
		for line in reader:
			list_training.append(line)
			trainingdata_count=trainingdata_count+1
	except csv.Error, e:
		print('EOF')
	fpData.close()
	filename = foldername+EXPNAME+'foreman.csv'
	fpData = open(filename,'rb')
	reader = csv.reader(fpData)
	try:
		for line in reader:
			list_training.append(line)
			trainingdata_count=trainingdata_count+1
	except csv.Error, e:
		print('EOF')
	fpData.close()
	filename = foldername+EXPNAME+'s201_1920x.csv'
	fpData = open(filename,'rb')
	reader = csv.reader(fpData)
	try:
		for line in reader:
			list_training.append(line)
			trainingdata_count=trainingdata_count+1
	except csv.Error, e:
		print('EOF')
	fpData.close()
	filename = foldername+EXPNAME+'s202_1920x.csv'
	fpData = open(filename,'rb')
	reader = csv.reader(fpData)
	try:
		for line in reader:
			list_training.append(line)
			trainingdata_count=trainingdata_count+1
	except csv.Error, e:
		print('EOF')
	fpData.close()
	filename = foldername+EXPNAME+'s203_1920x.csv'
	fpData = open(filename,'rb')
	reader = csv.reader(fpData)
	try:
		for line in reader:
			list_training.append(line)
			trainingdata_count=trainingdata_count+1
	except csv.Error, e:
		print('EOF')
	fpData.close()
	filename = foldername+EXPNAME+'s204_1920x.csv'
	fpData = open(filename,'rb')
	reader = csv.reader(fpData)
	try:
		for line in reader:
			list_training.append(line)
			trainingdata_count=trainingdata_count+1
	except csv.Error, e:
		print('EOF')
	fpData.close()
	OutputA,OutputB=transform_data(array(list_training),trainingdata_count,inputnum)
	return OutputA,OutputB,testdata_count
