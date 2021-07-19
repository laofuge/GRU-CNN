from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import tensorflow.contrib.slim as slim
import numpy as np
import tensorflow as tf
import time
import random
import math
import sys
import pandas as pd
import torch
class Model(object): 
	"""docstring for model"""
	def __init__(self, data_name, num_class, size = 300, batch_size = 64, dropout = 0.5, max_grad_norm = 5.0, L2reg = 0.000001, 
				 rnn_cell = 'lstm', optimize = 'Adagrad',lr=0.1):
		self.data_name = data_name
		self.num_class = num_class
		# self.shape_weight = shape_weight
		# self.shape_bias = shape_bias
		self.size = size
		self.lr = lr
		self.batch_size = batch_size
		self.dropout = dropout
		self.max_grad_norm = max_grad_norm
		self.bidirection = rnn_cell.startswith('bi')
		self.rnn_cell = rnn_cell
		self.optimize = optimize
		self.log_file = '../data/' + self.data_name + '/log' + '_' + rnn_cell + '_' + str(size)
		self.build()
		


	def padding(self, data):
		#print data[0]
		padded_batch_set = []
		max_docs1_len = max([len(doc1) for doc1,doc2,image,label in data])
		max_docs2_len = max([len(doc2) for doc1,doc2,image,label in data])

		max_words1_len = max([max([len(sentence) for sentence in doc1]) for doc1,doc2,image,label in data])
		max_words2_len = max([max([len(sentence) for sentence in doc2]) for doc1,doc2,image,label in data])
		for doc1,doc2,image,label in data:
			docs1_len = len(doc1)
			docs2_len = len(doc2)
			doc_pad1 = doc1 + [[0]] * (max_docs1_len - docs1_len)
			doc_pad2 = doc2 + [[0]] * (max_docs2_len - docs2_len)
			# words_len = [len(sentence) for sentence in doc_pad]
			doc_pad1 = [sentence + [0] * (max_words1_len - len(sentence)) for sentence in doc_pad1]
			doc_pad2 = [sentence + [0] * (max_words2_len - len(sentence)) for sentence in doc_pad2]
			padded_batch_set.append([doc_pad1,doc_pad2,image,label])
			# print(label)

		return padded_batch_set


	def get_batch_set(self, data, batch_size, shuffle = True):

		data_size = len(data)
		num_batches = int(data_size/batch_size) if data_size % batch_size == 0 else int(data_size/batch_size) + 1
		# Shuffle the data at each epoch
		if shuffle:
			random.shuffle(data) # shuffle：置乱

		for batch_num in range(num_batches):
			start_index = batch_num * batch_size
			end_index = min((batch_num + 1) * batch_size, data_size)
			
			yield self.padding(data[start_index:end_index])


	def get_cell(self, output_dim, dropout_keep_prob = 1.0):
		if self.rnn_cell.endswith('lstm'):
			cell = tf.contrib.rnn.LSTMCell(self.size, state_is_tuple=True)
		if self.rnn_cell.endswith('gru'):
			cell = tf.contrib.rnn.GRUCell(self.size)
		if self.rnn_cell.endswith('rnn'):
			cell = tf.contrib.rnn.BasicRNNCell(self.size)
		cell = tf.contrib.rnn.DropoutWrapper(cell, dropout_keep_prob, dropout_keep_prob)
		cell = tf.contrib.rnn.OutputProjectionWrapper(cell, output_dim)
		return cell


	def weight_variable(self, shape,scope=None):
		with tf.variable_scope(scope or "cnn5"):
			initial = tf.truncated_normal(shape, stddev = 0.1)
		return tf.Variable(initial)
	def bias_variable(self, shape):
		initial = tf.constant(0.1, shape = shape)
		return tf.Variable(initial)
	def conv2d(self, x, W):
		return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME')
	def max_pool_2x2(self, x):
		return tf.nn.max_pool(x, ksize = [1, 2, 1, 1], strides = [1, 1, 1, 1], padding = 'SAME')


	def variable_weight(self, shape, stddev = 0.05):
		init = tf.truncated_normal_initializer(stddev = stddev)
		return tf.get_variable(shape = shape, initializer = init, name = 'weight')
	def variable_bias(self, shape):
		init = tf.constant_initializer(0.1)
		return tf.get_variable(shape = shape, initializer = init, name = 'bias')		


	def convolution1(self, kernel, bias, x_image):
		W_conv1 = self.weight_variable(kernel,scope='cnn1')
		b_conv1 = self.bias_variable(bias)
		h_conv1 = tf.nn.relu(self.conv2d(x_image, W_conv1) + b_conv1)
		return self.max_pool_2x2(h_conv1)


	def convolution2(self, kernel, bias, x_image):
		W_conv2 = self.weight_variable(kernel,scope='cnn2')
		b_conv2 = self.bias_variable(bias)
		h_conv2 = tf.nn.relu(self.conv2d(x_image, W_conv2) + b_conv2)
		return self.max_pool_2x2(h_conv2)


	def fullconnect0(self, input_, output_size):
		shape = input_.get_shape().as_list() # 获取input的形状，将input_的形状转为list形式
		# print(input_.get_shape().as_list())
		input_size = shape[1] # 获取的这个形状的列数，即每个[]内的元素数
		W_fc1 = self.weight_variable([input_size, output_size])
		b_fc1 = self.bias_variable([output_size])
		input_flat = tf.reshape(input_, [-1, input_size])
		h_fc1 = tf.nn.relu(tf.matmul(input_flat, W_fc1) + b_fc1)
		return tf.nn.dropout(h_fc1, self.dropout)


	def fullconnect1(self, input_, output_size):
		shape = input_.get_shape().as_list() # 获取input的形状，将input_的形状转为list形式
		input_size = shape[1] # 获取的这个形状的列数，即每个[]内的元素数
		W_fc2 = self.weight_variable([input_size, output_size]) # 二维，第一维设置，第二维是输出维度
		b_fc2 = self.bias_variable([output_size])
		h_fc2 = tf.nn.relu(tf.matmul(input_, W_fc2) + b_fc2)
		return tf.nn.dropout(h_fc2, self.dropout)


	def fullconnect2(self, input_, output_size):
		shape = input_.get_shape().as_list() # 获取input的形状，将input_的形状转为list形式
		input_size = shape[1] # 获取的这个形状的列数，即每个[]内的元素数
		W_fc2 = self.weight_variable([input_size, output_size]) # 二维，第一维设置，第二维是输出维度
		b_fc2 = self.bias_variable([output_size])
		h_fc2 = tf.nn.relu(tf.matmul(input_, W_fc2) + b_fc2)
		return tf.nn.dropout(h_fc2, self.dropout)


	def fullconnect3(self, input_, output_size):
		shape = input_.get_shape().as_list() # 获取input的形状，将input_的形状转为list形式
		input_size = shape[1] # 获取的这个形状的列数，即每个[]内的元素数
		W_fc3 = self.weight_variable([input_size, output_size]) # 二维，第一维设置，第二维是输出维度
		b_fc3 = self.bias_variable([output_size])
		h_fc3 = tf.matmul(input_, W_fc3) + b_fc3
		return h_fc3


	def linear0(self, input_, output_size, scope=None):
	    shape = input_.get_shape().as_list() 
	    input_size = shape[1]
	    with tf.variable_scope(scope or "SimpleLinear1"):
	        matrix1 = tf.get_variable("Matrix1", [output_size, input_size], initializer=tf.constant_initializer(0.1),dtype=input_.dtype)
	        bias_term1 = tf.get_variable("Bias1", [output_size], dtype=input_.dtype)
	    return tf.matmul(input_, tf.transpose(matrix1)) + bias_term1


	def linear1(self, input_, output_size, scope=None):
	    shape = input_.get_shape().as_list() # 获取input的形状为元组，将该元组转为list形式
	    input_size = shape[1]

	    # Now the computation.
	    with tf.variable_scope(scope or "SimpleLinear1"):
	        matrix1 = tf.get_variable("Matrix2", [output_size, input_size], initializer=tf.constant_initializer(0.1),dtype=input_.dtype)
	        bias_term1 = tf.get_variable("Bias2", [output_size], dtype=input_.dtype)
	    return tf.matmul(input_, tf.transpose(matrix1)) + bias_term1


	def linear2(self, input_, output_size, scope=None):
	    shape = input_.get_shape().as_list() # 获取input的形状，将input_的形状转为list形式
	    input_size = shape[1] # 获取的这个形状的列数，即每个[]内的元素数

	    # Now the computation.
	    with tf.variable_scope(scope or "SimpleLinear2"):
	        matrix1 = tf.get_variable("Matrix3", [output_size, input_size], initializer=tf.constant_initializer(0.09),dtype=input_.dtype)
	        bias_term1 = tf.get_variable("Bias3", [output_size], dtype=input_.dtype)
	    return tf.matmul(input_, tf.transpose(matrix1)) + bias_term1


	def linear3(self, input_, output_size, scope=None):
	    shape = input_.get_shape().as_list() # 获取input的形状，将input_的形状转为list形式
	    input_size = shape[1] # 获取的这个形状的列数，即每个[]内的元素数

	    # Now the computation.
	    with tf.variable_scope(scope or "SimpleLinear2"):
	        matrix1 = tf.get_variable("Matrix4", [output_size, input_size], initializer=tf.constant_initializer(0.09),dtype=input_.dtype)
	        bias_term1 = tf.get_variable("Bias4", [output_size], dtype=input_.dtype)
	    return tf.matmul(input_, tf.transpose(matrix1)) + bias_term1


	def linear4(self, input_, output_size, act = tf.nn.relu, scope = 'fully_connect', reuse = None):
		shape = input_.get_shape().as_list() # 获取input的形状，将input_的形状转为list形式 [?, 29, 20]
		input_size = shape[-1] # 20
		print(shape)
		out = []
		with tf.variable_scope('weight'):
			weight = self.variable_weight([input_size, output_size]) # [20, outsize]
		with tf.variable_scope('bias'):
			bias = self.variable_bias([output_size])
		with tf.variable_scope('weight2333'):
			weight1 = self.variable_weight([1, 30])	
		# if shape[0] == self.batch_size:
		for r in range(self.batch_size):
			input_1 = tf.slice(input_, [r, 0, 0], [1, -1, -1]) # tf.slice(inputs, begin, size, name)
			input_1 = tf.reshape(input_1, [shape[1], shape[2]]) # [29, 20]

			# fc = tf.matmul(input_1, weight) # [29, outsize]
			
			fc1 = tf.nn.bias_add(tf.matmul(weight1, input_1), bias, name = 'fc') # [1, outsize]
			out.append(fc1)
		out = tf.reshape(out, [self.batch_size, output_size])
		return out	


	def linear5(self, input_, output_size, act = tf.nn.relu, scope = 'fully_connect', reuse = None):
		shape = input_.get_shape().as_list() # 获取input的形状，将input_的形状转为list形式 [?, 29, 20]
		input_size = shape[-1] # 20
		print(shape)
		out = []
		with tf.variable_scope('weight'):
			weight = self.variable_weight([input_size, output_size]) # [20, outsize]
		with tf.variable_scope('bias'):
			bias = self.variable_bias([output_size])
		with tf.variable_scope('weight2333'):
			# weight1 = np.array([[0.06, 0.06, 0.06, 0.06, 0.06, 0.05, 0.05, 0.05, 0.05, 0.05, 0.04, 0.04, 0.04, 0.04, 0.04, 0.03, 0.03, 0.03, 0.03, 0.03, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]])
			
			weight1 = self.variable_weight([1, 30, 1])
		# for r in range(self.batch_size):
		# 	input_1 = tf.slice(input_, [r, 0, 0], [1, -1, -1]) # tf.slice(inputs, begin, size, name)
		# 	input_1 = tf.reshape(input_1, [shape[1], shape[2]]) # [29, 20]

		fc = tf.multiply(input_, weight1) # [29, outsize]

		print(fc.get_shape().as_list()) # 获取input的形状，将input_的形状转为list形式 [?, 30, 20]

		arlist = []
		for r in range(30):
			fc_ = tf.slice(fc, [0, r, 0], [-1, 1, -1])
			fc_ = tf.reshape(fc_, [-1, self.batch_size])
			arlist.append(fc_)
		for g in range(29):
			out = tf.reshape(arlist[0], [-1, self.batch_size])
			out += arlist[g+1]
		out = tf.reshape(out, [-1, self.batch_size])
		return out


# VGG
	def variable_weight(self, shape, stddev = 0.05):
		init = tf.truncated_normal_initializer(stddev = stddev)
		return tf.get_variable(shape = shape, initializer = init, name = 'weight')
	def variable_bias(self, shape):
		init = tf.constant_initializer(0.1)
		return tf.get_variable(shape = shape, initializer = init, name = 'bias')


	def conv(self, x, ksize, out_depth, strides, padding = 'SAME', act = tf.nn.sigmoid, scope = 'conv_layer', reuse = None):
		in_depth = x.get_shape().as_list()[-1]
		with tf.variable_scope(scope, reuse = reuse):
			shape = ksize + [in_depth, out_depth]
			with tf.variable_scope('kernel'):
				kernel = self.variable_weight(shape)
			strides = [1, strides[0], strides[1], 1]
			conv = tf.nn.conv2d(x, kernel, strides, padding, name = 'conv')
			with tf.variable_scope('bias'):
				bias = self.variable_bias([out_depth])
			preact = tf.nn.bias_add(conv, bias)
			out = act(preact)
			return out


	def max_pool(self, x, ksize, strides, padding = 'SAME', name = 'pool_layer'):
		return tf.nn.max_pool(x, [1, ksize[0], ksize[1], 1], [1, strides[0], strides[1], 1], padding, name = name)


	def fc(self, x, out_depth, act = tf.nn.sigmoid, scope = 'fully_connect', reuse = None):
		in_depth = x.get_shape().as_list()[-1]
		with tf.variable_scope(scope, reuse = reuse):
			with tf.variable_scope('weight'):
				weight = self.variable_weight([in_depth, out_depth])
			with tf.variable_scope('bias'):
				bias = self.variable_bias([out_depth])
			fc = tf.nn.bias_add(tf.matmul(x, weight), bias, name = 'fc')
			out = act(fc)
			return out


	def vgg_block(self, inputs, num_convs, out_depth, scope = 'vgg_block', reuse = None):
		in_depth = inputs.get_shape().as_list()[-1]
		with tf.variable_scope(scope, reuse = reuse) as sc:
			net = inputs
			for i in range(num_convs):
				net = self.conv(net, ksize = [5, 5], out_depth = out_depth, strides = [1, 1], padding = 'SAME', scope = 'conv%d' % i, reuse = reuse)
			net = self.max_pool(net, [2, 2], [2, 2], name = 'pool')
			return net


	def vgg_stack(self, inputs, num_convs, out_depths, scope = 'vgg_stack', reuse = None):
		with tf.variable_scope(scope, reuse = reuse) as sc:
			net = inputs
			for i, (n, d) in enumerate(zip(num_convs, out_depths)):
				net = self.vgg_block(net, n, d, scope = 'block%d' % i)
			return net


	def vgg(self, inputs, num_convs, out_depths, num_outputs, scope = 'vgg', reuse = None):
		with tf.variable_scope('vgg', reuse = reuse) as sc:
			net = self.vgg_stack(inputs, num_convs, out_depths)
			print(net.get_shape().as_list())

			with tf.variable_scope('regression'):
				net = tf.reshape(net, [-1, 256*7*10])
				net = self.fc(net, 20)
				net = self.fc(net, num_outputs, act = tf.identity, scope = 'regression')
			return net


# GoogleNet
	def inception(self, x, d0_1, d1_1, d1_3, d2_1, d2_5, d3_1, scope = 'inception', reuse = None):
		with tf.variable_scope(scope, reuse = reuse):
			with slim.arg_scope([slim.conv2d, slim.max_pool2d], stride = 1, padding = 'SAME'):
				with tf.variable_scope('branch0'):
					branch_0 = slim.conv2d(x, d0_1, [1, 1], scope = 'conv_1x1')
				with tf.variable_scope('branch1'):
					branch_1 = slim.conv2d(x, d1_1, [1, 1], scope = 'conv_1x1')
					branch_1 = slim.conv2d(branch_1, d1_3, [3, 3], scope = 'conv_3x1')
				with tf.variable_scope('branch2'):
					branch_2 = slim.conv2d(x, d2_1, [1, 1], scope = 'conv_1x1')
					branch_2 = slim.conv2d(branch_2, d2_5, [5, 5], scope = 'conv_5x1')
				with tf.variable_scope('branch3'):
					branch_3 = slim.max_pool2d(x, [3, 3], scope = 'conv_1x1')
					branch_3 = slim.conv2d(branch_3, d3_1, [1, 1], scope = 'conv_1x1')					
				net = tf.concat([branch_0, branch_1, branch_2, branch_3], axis = -1)
				return net

	
	def googlenet(self, inputs, num_classes, reuse = None, is_training = None, verbose = False):
		with tf.variable_scope('googlenet', reuse = tf.AUTO_REUSE): # ?
			with slim.arg_scope([slim.batch_norm], is_training = is_training):
				with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], padding = 'SAME', stride = 1):
					net = inputs
					# print(net)
					with tf.variable_scope('block1'):
						net = slim.conv2d(net, 16, [5, 5], stride = 2, scope = 'conv_5x1')
						if verbose:
							print('block1 output: ', net.get_shape().as_list())
					with tf.variable_scope('block2'):
						net = slim.conv2d(net, 16, [1, 1], stride = 1, scope = 'conv_1x1')
						net = slim.conv2d(net, 48, [3, 3], stride = 1, scope = 'conv_3x1')
						net = slim.max_pool2d(net, [3, 3], stride = 2, scope = 'maxpool')
						if verbose:
							print('block2 output: ', net.get_shape().as_list())
					with tf.variable_scope('block3'):
						net = self.inception(net, 16, 24, 32, 4, 8, 8, scope = 'inception_1')
						net = self.inception(net, 32, 32, 48, 8, 24, 16, scope = 'inception_2')
						net = slim.max_pool2d(net, [3, 3], stride = 1, scope = 'max_pool')
						if verbose:
							print('block3 output: ', net.get_shape().as_list())
					with tf.variable_scope('block4'):
						net = self.inception(net, 48, 24, 52, 4, 12, 16, scope = 'inception_1')
						net = self.inception(net, 40, 28, 56, 6, 16, 16, scope = 'inception_2')
						net = self.inception(net, 32, 32, 64, 4, 16, 16, scope = 'inception_3')
						# net = self.inception(net, 112, 144, 288, 24, 64, 64, scope = 'inception_4')
						# net = self.inception(net, 256, 160, 320, 32, 128, 128, scope = 'inception_5')
						net = slim.max_pool2d(net, [3, 3], stride = 2, scope = 'max_pool')
						if verbose:
							print('block4 output: ', net.get_shape().as_list())
					with tf.variable_scope('block5'):
						net = self.inception(net, 64, 40, 80, 8, 32, 32, scope = 'inception_1')
						# net = self.inception(net, 384, 182, 384, 48, 128, 128, scope = 'inception_2')
						net = slim.avg_pool2d(net, [2, 2], stride = 2, scope = 'avg_pool')
						if verbose:
							print('block5 output: ', net.get_shape().as_list())
					with tf.variable_scope('regression'):
						net = slim.flatten(net)
						net = slim.fully_connected(net, num_classes, activation_fn = None, normalizer_fn = None, scope = 'logit')
						# 因子分解机
						if verbose:
							print('regression output: ', net.get_shape().as_list())
					return net


# ResNet
	def subsample(self, x, factor, scope = None):
		if factor == 1:
			return x
		return slim.max_pool2d(x, [1, 1], factor, scope = scope)


	def residual_block(self, x, bottleneck_depth, out_depth, stride = 1, scope = 'residual_block'):
		in_depth = x.get_shape().as_list()[-1]
		with tf.variable_scope(scope):
			if in_depth == out_depth:
				shortcut = self.subsample(x, stride, 'shortcut')
			else:
				shortcut = slim.conv2d(x, out_depth, [1, 1], stride = stride, activation_fn = None, scope = 'shortcut')
			residual = slim.conv2d(x, bottleneck_depth, [1, 1], stride = stride, scope = 'conv1')
			residual = slim.conv2d(residual, bottleneck_depth, [3, 3], 1, scope = 'conv2')
			residual = slim.conv2d(residual, out_depth, [1, 1], stride = 1, activation_fn = None, scope = 'conv3')
			output = tf.nn.relu(shortcut + residual)
			return output


	def resnet(self, inputs, num_classes, reuse = None, is_training = None, verbose = False):
		with tf.variable_scope('resnet', reuse = reuse):
			net = inputs
			if verbose:
				print('input: ', net.get_shape().as_list())
			with slim.arg_scope([slim.batch_norm], is_training = is_training):
				with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], padding = 'SAME'):
					with tf.variable_scope('block1'):
						net = slim.conv2d(net, 32, [7, 7], stride = 2, scope = 'conv_5x5')
						if verbose:
							print('block1:', net.get_shape().as_list())
					with tf.variable_scope('block2'):
						net = slim.max_pool2d(net, [3, 3], 2, scope = 'max_pool')
						net = self.residual_block(net, 32, 128, scope = 'residual_block1')
						# net = self.residual_block(net, 32, 128, scope = 'residual_block2')
						if verbose:
							print('block2:', net.get_shape().as_list())
					with tf.variable_scope('block3'):
						net = self.residual_block(net, 64, 256, stride = 2, scope = 'residual_block1')
						# net = self.residual_block(net, 64, 256, scope = 'residual_block2')
						if verbose:
							print('block3:', net.get_shape().as_list())
					with tf.variable_scope('block4'):
						net = self.residual_block(net, 128, 512, stride = 2, scope = 'residual_block1')
						net = self.residual_block(net, 128, 512, scope = 'residual_block2')
						net = slim.avg_pool2d(net, [7, 7], 7, scope = 'max_pool')
						if verbose:
							print('block4:', net.get_shape().as_list())
					with tf.variable_scope('regression'):
						# net = tf.reduce_mean(net, [1, 2], name = 'global_pool', keep_dims = True)
						net = slim.flatten(net, scope = 'flatten')
						# net = tf.nn.dropout(slim.fully_connected(net, 500, activation_fn = None, normalizer_fn = None, scope = 'logit1'), self.dropout)	
						# net = tf.nn.dropout(slim.fully_connected(net, 100, activation_fn = None, normalizer_fn = None, scope = 'logit2'), self.dropout)
						# # net = slim.fully_connected(net, 25, activation_fn = None, normalizer_fn = None, scope = 'logit3')
						# net = tf.nn.dropout(slim.fully_connected(net, num_classes, activation_fn = None, normalizer_fn = None, scope = 'logit4'), self.dropout)

						if verbose:
							print('regression:', net.get_shape().as_list())					
					return net





# 
	def build(self):
		self.dropout_keep_prob = tf.placeholder(tf.float32)
		self.imagesforCNN = tf.placeholder(tf.float32, [None, 112, 112, 3])
		self.labelsforCNN = tf.placeholder(tf.float32, [None, 1])
		self.docs1 = tf.placeholder(tf.float32, [None, 30, 6]) 
		# self.docs2 = tf.placeholder(tf.float32, [None, 30, 6]) # 三维，分别是batch size（每个batch多少个样本）、一个样本多少天、一天多少个数据。None处也可改为真实的已知数值。
		self.labels = tf.placeholder(tf.float32, [None])
	

		# cnn_output2 = self.vgg(self.imagesforCNN, (1, 1, 2, 2, 2), (32, 64, 128, 256, 256), 10)	
# googlenet
		# self.reuse = tf.placeholder(tf.bool, name = 'reuses')		
		# self.is_training = tf.placeholder(tf.bool, name = 'is_training')
		# with slim.arg_scope([slim.conv2d], activation_fn = tf.nn.relu, normalizer_fn = slim.batch_norm) as sc:
		# 	conv_scope = sc
		# with slim.arg_scope(conv_scope):
		# 	cnn_output2 = self.googlenet(self.imagesforCNN, 10, is_training = self.is_training, verbose = True)

# resnet
		self.reuse = tf.placeholder(tf.bool, name = 'reuses')		
		self.is_training = tf.placeholder(tf.bool, name = 'is_training')
		with slim.arg_scope([slim.conv2d], activation_fn = tf.nn.relu, normalizer_fn = slim.batch_norm) as sc:
			conv_scope = sc
		with slim.arg_scope(conv_scope):
			cnn_output2 = self.resnet(self.imagesforCNN, 25, is_training = self.is_training, verbose = True)




		with tf.variable_scope("doc_rnn0"):
			cell = self.get_cell(self.size, self.dropout_keep_prob) # self.dropout_keep_prob指train中的dropout值
			outputs0, state = tf.nn.dynamic_rnn(cell, inputs = self.docs1, dtype = tf.float32)
			# print(outputs0.get_shape().as_list()) # shape的第三个数是size of each layer

		# outputs0 = self.linear5(outputs0, 100, scope = 'fc111')
		# outputs0 = tf.reshape(outputs0, [self.batch_size, 30, 100])

		with tf.variable_scope("doc_rnn1"):
			cell = self.get_cell(self.size, self.dropout_keep_prob)
			outputs1, state = tf.nn.dynamic_rnn(cell, inputs = outputs0, dtype = tf.float32)


		# with tf.variable_scope("doc_rnn2"):
		# 	cell = self.get_cell(self.size, self.dropout_keep_prob) # self.dropout_keep_prob指train中的dropout值
		# 	outputs2, state = tf.nn.dynamic_rnn(cell, inputs = self.docs2, dtype = tf.float32)
		# 	# print(outputs0.get_shape().as_list()) # shape的第三个数是size of each layer

		# # outputs0 = self.linear5(outputs0, 100, scope = 'fc111')
		# # outputs0 = tf.reshape(outputs0, [self.batch_size, 30, 100])

		# with tf.variable_scope("doc_rnn3"):
		# 	cell = self.get_cell(self.size, self.dropout_keep_prob)
		# 	outputs3, state = tf.nn.dynamic_rnn(cell, inputs = outputs2, dtype = tf.float32)

		# rnn_output1 = tf.concat([outputs1, outputs3], axis = -1)
		print(outputs1.get_shape().as_list())		
		rnn_output2 = outputs1[:,-1]
		# rnn_output2 = rnn_output1
		# rnn_output2 = self.linear5(rnn_output1, 20, scope = 'fc14')
		# print(rnn_output2.get_shape().as_list()) # shape的第三个数是size of each layer

		fusion_output0 = tf.concat([rnn_output2, cnn_output2], axis = -1)
		# print(fusion_output0.get_shape().as_list()) # shape的第三个数是size of each layer
		fusion_output1 = self.linear0(fusion_output0, 500, scope = 'fc114') # 勿动

		# fusion_output1 = self.fullconnect0(fusion_output0, 512)
		# cnn_output = self.CNNfullconnect2(cnn_output, 32)
		# print(cnn_output.get_shape().as_list())
		# fusion_output2 = self.fullconnect1(fusion_output0, 16)
		fusion_output3 = self.linear1(fusion_output0, 100)

		# print(cnn_output.get_shape().as_list())
		fusion_output4 = self.linear2(fusion_output3, 25)
		fusion_output5 = self.linear3(fusion_output4, 1)

		self.pred_fusion = fusion_output5
		# loss_CNN=tf.square(self.labelsforCNN - cnn_output3)

		# loss_fusion=tf.abs(self.labels - fusion_output4)
		# self.mean_loss_fusion = tf.reduce_mean(loss_fusion)
		self.mean_loss_fusion = tf.reduce_mean(tf.square(self.labels - self.pred_fusion))

		# tvars_fusion = tf.trainable_variables()
		# grads_fusion, _ = tf.clip_by_global_norm(tf.gradients(self.mean_loss_fusion, tvars_fusion), self.max_grad_norm)
		
		# if self.optimize == 'Adagrad':
		# 	optimizer = tf.train.AdagradOptimizer(self.lr)
		# else:
		# 	optimizer = tf.train.AdamOptimizer(self.lr)
		
		# self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = self.labelsforCNN, logits = self.pred_CNN))
		# self.train_op_CNN = tf.train.AdamOptimizer(self.lr).minimize(self.cross_entropy)
		self.train_op_fusion = tf.train.RMSPropOptimizer(self.lr).minimize(self.mean_loss_fusion)


	def fit(self, train_set, test_set, epoch = 40):		
		pred_train, alltrainloss = [], []
		config = tf.ConfigProto(allow_soft_placement=True) # 自动分配CPU或GPU来运行
		config.gpu_options.allow_growth=True
		# config.gpu_options.per_process_gpu_memory_fraction = 0.4
		with tf.Session(config=config) as sess:
			# sess = tf.InteractiveSession()
			sess.run(tf.global_variables_initializer())
			for t in range(epoch):
				batch_set = self.get_batch_set(train_set, self.batch_size, True)
				loss, start_time = 0.0, time.time()
				num_batches = int(len(train_set)/self.batch_size) if len(train_set) % self.batch_size == 0 else int(len(train_set)/self.batch_size) + 1
				for i, batch_sample in enumerate(batch_set):
					docs1, docs2, images, labels = zip(*batch_sample)
					# for j in range(len(docs)):
					# 	for k in range(len(docs[j])):
					# 		for l in range(len(docs[j][k])):
					# 			docs[j][k][l] = [docs[j][k][l]]
					# docsforCNN = docs			
					labelsforCNN = []
					labels = list(labels)
					for j in range(len(labels)):
						labels[j] = [labels[j]]
					labelsforCNN = labels

					# docsforRNN = np.array([docsforCNN]).reshape(-1, 30, 6)
					labelsforRNN = np.array([labelsforCNN]).reshape(-1)
					#print X, '\n', Y, '\n', length, '\n', mask, '\n', self.dropout, '\n'
					#return
					batch_loss, _,pred, dummy_loss = sess.run([self.mean_loss_fusion, self.train_op_fusion, self.pred_fusion, self.mean_loss_fusion],
											 {self.imagesforCNN: images, self.labelsforCNN: labelsforCNN, self.docs1: docs1, self.labels: labelsforRNN, self.dropout_keep_prob: 0.5, self.is_training: 1})
					loss += dummy_loss # 不是真正的loss，只是用来观察loss是否随迭代缩小。真正的loss在build函数中变量loss体现（没有输出这个损失值，只去拟合来使其最小）。
					print ('training epoch %d, %.2f ...' % (t+1, ((i+1) * 100.0 / num_batches)))
					sys.stdout.write("\033[F")
					for m in pred:
						if t == epoch-1:
							pred_train.append(m)
					# print(self.pred_CNN)
				print("%d : loss = %.6f, time = %.3f" % (t+1, loss, time.time() - start_time), end='')		
				alltrainloss.append(loss)
				print('\n')
			np.save('../data/trainpredsFNN.npy', pred_train)
			self.evaluate(test_set, sess)
			print(alltrainloss)
			print("##### THE END OF FUSION FITTING #####")


	def evaluate(self, test_set, sess): # 此函数处理测试集
		preds = []
		batch_set = self.get_batch_set(test_set, self.batch_size, False) # 98275 一个batch的数据数量（必须大于测试集样本个数） False：不打散		
		num_batches = int(len(test_set)/self.batch_size) if len(test_set) % self.batch_size == 0 else int(len(test_set)/self.batch_size) + 1
		for i, batch_sample in enumerate(batch_set):
			docs1, docs2, images, labels = zip(*batch_sample)
			# docsforRNN = docs
			# labelsforRNN = labels
			# for j in range(len(docs)):
			# 	for k in range(len(docs[j])):
			# 		for l in range(len(docs[j][k])):
			# 			docs[j][k][l] = [float(docs[j][k][l])]
			# docsforCNN = docs
			# print(docsforCNN)
			labelsforCNN = []
			labels = list(labels)
			for j in range(len(labels)):
				labels[j] = [float(labels[j])]
			labelsforCNN = labels
			# docsforRNN = np.array([docsforCNN]).reshape(-1, 30, 6)
			labelsforRNN = np.array([labelsforCNN]).reshape(-1)
			# print(labelsforCNN)
			batch_loss, pred = sess.run([self.mean_loss_fusion, self.pred_fusion],
			 								 {self.imagesforCNN: images, self.labelsforCNN: labelsforCNN, self.docs1: docs1, self.labels: labelsforRNN, self.dropout_keep_prob: 1.0, self.is_training: 1, self.reuse: True})
			print ('testing %.2f ...' % ((i+1) * 100.0 / num_batches))
			for m in pred:
				preds.append(m)
		np.save('../data/testpredsFNN.npy',preds)			
