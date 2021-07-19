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
		max_images_len = max([len(image) for image, label in data])
		max_words_len = max([max([len(sentence) for sentence in image]) for image, label in data])
		for image, label in data:
			# images_len = len(image)
			# image_pad = image + [[0]] * (max_images_len - images_len)
			# words_len = [len(sentence) for sentence in image_pad]
			# image_pad = [sentence + [0] * (max_words_len - len(sentence)) for sentence in image_pad]
			padded_batch_set.append([image, label])
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


	def conv1(self, kernel, bias, x_image):
		W_conv1 = self.weight_variable(kernel,scope='cnn1')
		b_conv1 = self.bias_variable(bias)
		h_conv1 = tf.nn.sigmoid(self.conv2d(x_image, W_conv1) + b_conv1)
		return self.max_pool_2x2(h_conv1)


	def conv2(self, kernel, bias, x_image):
		W_conv2 = self.weight_variable(kernel,scope='cnn2')
		b_conv2 = self.bias_variable(bias)
		h_conv2 = tf.nn.sigmoid(self.conv2d(x_image, W_conv2) + b_conv2)
		return self.max_pool_2x2(h_conv2)


	def CNNfullconnect1(self, input_, output_size):
		shape = input_.get_shape().as_list() # 获取input的形状，将input_的形状转为list形式
		# print(input_.get_shape().as_list())
		input_size = shape[1] * shape [2]* shape [3] # 获取的这个形状的列数，即每个[]内的元素数：30*6 注意：这个维度会被消掉，所以设为最匹配两边矩阵的数
		W_fc1 = self.weight_variable([input_size, output_size])
		b_fc1 = self.bias_variable([output_size])
		input_flat = tf.reshape(input_, [-1, input_size])
		h_fc1 = tf.nn.sigmoid(tf.matmul(input_flat, W_fc1) + b_fc1)
		# keep_prob = tf.placeholder(tf.float32)
		return tf.nn.dropout(h_fc1, self.dropout)


	def CNNfullconnect2(self, input_, output_size):
		shape = input_.get_shape().as_list() # 获取input的形状，将input_的形状转为list形式
		input_size = shape[1] # 获取的这个形状的列数，即每个[]内的元素数
		W_fc2 = self.weight_variable([input_size, output_size]) # 二维，第一维设置，第二维是输出维度
		b_fc2 = self.bias_variable([output_size])
		h_fc2 = tf.nn.sigmoid(tf.matmul(input_, W_fc2) + b_fc2)
		return tf.nn.dropout(h_fc2, self.dropout)


	def CNNfullconnect3(self, input_, output_size):
		shape = input_.get_shape().as_list() # 获取input的形状，将input_的形状转为list形式
		input_size = shape[1] # 获取的这个形状的列数，即每个[]内的元素数
		W_fc3 = self.weight_variable([input_size, output_size]) # 二维，第一维设置，第二维是输出维度
		b_fc3 = self.bias_variable([output_size])
		h_fc3 = tf.matmul(input_, W_fc3) + b_fc3
		return h_fc3


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
				net = tf.reshape(net, [-1, 512*7*10])
				net = self.fc(net, 6)
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
						net = slim.conv2d(net, 64, [5, 5], stride = 2, scope = 'conv_5x1')
						if verbose:
							print('block1 output: ', net.get_shape().as_list())
					with tf.variable_scope('block2'):
						net = slim.conv2d(net, 64, [1, 1], stride = 1, scope = 'conv_1x1')
						net = slim.conv2d(net, 192, [3, 3], stride = 1, scope = 'conv_3x1')
						net = slim.max_pool2d(net, [3, 3], stride = 2, scope = 'maxpool')
						if verbose:
							print('block2 output: ', net.get_shape().as_list())
					with tf.variable_scope('block3'):
						net = self.inception(net, 64, 96, 128, 16, 32, 32, scope = 'inception_1')
						net = self.inception(net, 128, 128, 192, 32, 96, 64, scope = 'inception_2')
						net = slim.max_pool2d(net, [3, 3], stride = 1, scope = 'max_pool')
						if verbose:
							print('block3 output: ', net.get_shape().as_list())
					with tf.variable_scope('block4'):
						net = self.inception(net, 192, 96, 208, 16, 48, 64, scope = 'inception_1')
						net = self.inception(net, 160, 112, 224, 24, 64, 64, scope = 'inception_2')
						net = self.inception(net, 128, 128, 256, 24, 64, 64, scope = 'inception_3')
						# net = self.inception(net, 112, 144, 288, 24, 64, 64, scope = 'inception_4')
						# net = self.inception(net, 256, 160, 320, 32, 128, 128, scope = 'inception_5')
						net = slim.max_pool2d(net, [3, 3], stride = 2, scope = 'max_pool')
						if verbose:
							print('block4 output: ', net.get_shape().as_list())
					with tf.variable_scope('block5'):
						net = self.inception(net, 256, 160, 320, 32, 128, 128, scope = 'inception_1')
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
			residual = tf.nn.relu(slim.conv2d(x, bottleneck_depth, [1, 1], stride = stride, scope = 'conv1'))
			residual = tf.nn.relu(slim.conv2d(residual, bottleneck_depth, [3, 3], 1, scope = 'conv2'))
			residual = tf.nn.relu(slim.conv2d(residual, out_depth, [1, 1], stride = 1, activation_fn = None, scope = 'conv3'))
			output = shortcut + residual
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
						# net = self.residual_block(net, 128, 512, scope = 'residual_block2')
						net = slim.avg_pool2d(net, [7, 7], 7, scope = 'max_pool')
						if verbose:
							print('block4:', net.get_shape().as_list())
					with tf.variable_scope('regression'):
						# net = tf.reduce_mean(net, [1, 2], name = 'global_pool', keep_dims = True)
						net = slim.flatten(net, scope = 'flatten')
						net = tf.nn.dropout(slim.fully_connected(net, 500, activation_fn = None, normalizer_fn = None, scope = 'logit1'), self.dropout)	
						net = tf.nn.dropout(slim.fully_connected(net, 100, activation_fn = None, normalizer_fn = None, scope = 'logit2'), self.dropout)
						net = tf.nn.dropout(slim.fully_connected(net, 25, activation_fn = None, normalizer_fn = None, scope = 'logit3'), self.dropout)
						net = slim.fully_connected(net, num_classes, activation_fn = None, normalizer_fn = None, scope = 'logit4')

						if verbose:
							print('regression:', net.get_shape().as_list())					
					return net

# DenseNet
	def bn_relu_conv(self, x, out_depth, scope = 'dense_basic_conv', reuse = None):
		with tf.variable_scope(scope, reuse = reuse):
			net = slim.batch_norm(x, activation_fn = None, scope = 'bn')
			net = tf.nn.sigmoid(net, name = 'activation')
			net = slim.conv2d(net, out_depth, 3, activation_fn = None, normalizer_fn = None, biases_initializer = None, scope = 'conv', padding = 'SAME')
			return net


	def dense_block(self, x, growth_rate, num_layers, scope = 'dense_block', reuse = None):
		in_depth = x.get_shape().as_list()[-1]
		with tf.variable_scope(scope, reuse = reuse):
			net = x
			for i in range(num_layers):
				out = self.bn_relu_conv(net, growth_rate, scope = 'block%d' % i)
				net = tf.concat([net, out], axis = -1)
			return net


	def transition(self, x, out_depth, scope = 'transition', reuse = None):
		in_depth = x.get_shape().as_list()[-1]
		with tf.variable_scope(scope, reuse = reuse):
			net = slim.batch_norm(x, activation_fn = None, scope = 'bn')
			net = tf.nn.sigmoid(net, name = 'activation')
			net = slim.conv2d(net, out_depth, 1, activation_fn = None, normalizer_fn = None, biases_initializer = None, scope = 'conv', padding = 'SAME')	
			net = slim.avg_pool2d(net, [2, 1], 1, scope = 'avg_pool', padding = 'SAME')
			return net

	def densenet(self, x, num_classes, growth_rate = 16, block_layers = [6, 12, 12, 8], is_training = None, scope = 'densenet', reuse = None, verbose = False):
		with tf.variable_scope(scope, reuse = reuse):
			with slim.arg_scope([slim.batch_norm], is_training = is_training):
				if verbose:
					print('input: ', x.get_shape().as_list())
				with tf.variable_scope('block0'):
					net = slim.conv2d(x, 64, [7, 1], stride = 1, normalizer_fn = None, activation_fn = None, scope = 'conv7x1')
					net = slim.batch_norm(net, activation_fn = None, scope = 'bn')
					net = tf.nn.sigmoid(net, name = 'activation')
					net = slim.max_pool2d(net, [2, 1], stride = 1, scope = 'max_pool', padding = 'SAME')
					if verbose:
						print('block0: ', net.get_shape().as_list())
				for i, num_layers in enumerate(block_layers):
					with tf.variable_scope('block%d' % (i + 1)):
						net = self.dense_block(net, growth_rate, num_layers)
						if i != len(block_layers) - 1:
							current_depth = net.get_shape().as_list()[-1]
							net = self.transition(net, current_depth // 2)
					if verbose:
						print('block%d: '% (i+1), net.get_shape().as_list())
				with tf.variable_scope('block%d' % (len(block_layers) + 1)):
					net = slim.batch_norm(net, activation_fn = None, scope = 'bn')
					net = tf.nn.sigmoid(net, name = 'activation')
					net = tf.reduce_mean(net, [1, 2], name = 'global_pool', keep_dims = True)
					if verbose:
						print('block%d: '% (len(block_layers) + 1), net.get_shape().as_list())
				with tf.variable_scope('regression'):
					net = slim.flatten(net, scope = 'flatten')
					net = slim.fully_connected(net, 6, activation_fn = None, normalizer_fn = None, scope = 'logit1')					
					net = slim.fully_connected(net, num_classes, activation_fn = None, normalizer_fn = None, scope = 'logit2')
					if verbose:
						print('regression: ', net.get_shape().as_list())
					return net
			

#	
	def build(self, reuse = None):
	
		self.dropout_keep_prob = tf.placeholder(tf.float32)
		# self.docsforCNN = tf.placeholder(tf.float32, [None, 30, 6, 1])
		self.labelsforCNN = tf.placeholder(tf.float32, [None, 1])
		self.imagesforCNN = tf.placeholder(tf.float32, [None, 112, 112, 3])
# VGG
		# cnn_output3 = self.vgg(self.imagesforCNN, (1, 1, 2, 2, 2), (64, 128, 256, 512, 512), 1)	

# googlenet		
		# self.is_training = tf.placeholder(tf.bool, name = 'is_training')
		# with slim.arg_scope([slim.conv2d], activation_fn = tf.nn.sigmoid, normalizer_fn = slim.batch_norm) as sc:
		# 	conv_scope = sc
		# with slim.arg_scope(conv_scope):
		# 	cnn_output3 = self.googlenet(self.imagesforCNN, 1, is_training = self.is_training, verbose = True)
# resnet
		self.reuse = tf.placeholder(tf.bool, name = 'reuses')		
		self.is_training = tf.placeholder(tf.bool, name = 'is_training')
		with slim.arg_scope([slim.conv2d], activation_fn = tf.nn.relu, normalizer_fn = slim.batch_norm) as sc:
			conv_scope = sc
		with slim.arg_scope(conv_scope):
			cnn_output3 = self.resnet(self.imagesforCNN, 1, is_training = self.is_training, verbose = True)
# densenet
		# self.is_training = tf.placeholder(tf.bool, name = 'is_training')
		# with slim.arg_scope([slim.conv2d], activation_fn = tf.nn.sigmoid, normalizer_fn = slim.batch_norm) as sc:
		# 	conv_scope = sc
		# with slim.arg_scope(conv_scope):
		# 	cnn_output3 = self.densenet(self.docsforCNN, 1, is_training = self.is_training, verbose = True)
		# print(cnn_output.get_shape().as_list()) # [None, 6, 30, 1]
		# cnn_output3 = self.CNNfullconnect1(cnn_output3, 1)
# CNN
		# cnn_output = self.conv1([5, 5, 3, 16], [16], self.imagesforCNN)
		# print(cnn_output.get_shape().as_list()) # [None, 6, 30, 1]
		# cnn_output = self.conv2([3, 3, 16, 32], [32], cnn_output)
		# print(cnn_output.get_shape().as_list()) # [None, 6, 30, 1]
		# # cnn_output1 = self.CNNfullconnect1(cnn_output, 100)
		# cnn_output2 = self.CNNfullconnect1(cnn_output, 6)
		# cnn_output3 = self.CNNfullconnect3(cnn_output2, 1)
		# print(cnn_output.get_shape().as_list())

# loss
		self.pred_CNN = cnn_output3
		self.mean_loss_CNN = tf.reduce_mean(tf.square(self.labelsforCNN - self.pred_CNN)) ** 0.5
		self.train_op_CNN = tf.train.AdamOptimizer(self.lr).minimize(self.mean_loss_CNN)
		# self.train_op_CNN = tf.train.RMSPropOptimizer(self.lr).minimize(self.mean_loss_CNN)




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
					imgs, labels = zip(*batch_sample)
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
					#print X, '\n', Y, '\n', length, '\n', mask, '\n', self.dropout, '\n'
					#return
					batch_loss, _,pred, dummy_loss = sess.run([self.mean_loss_CNN, self.train_op_CNN, self.pred_CNN, self.mean_loss_CNN],
											 {self.imagesforCNN: imgs, self.labelsforCNN: labelsforCNN, self.dropout_keep_prob: 0.5, self.is_training: 1})
					loss += dummy_loss # 不是真正的loss，只是用来观察loss是否随迭代缩小。真正的loss在build函数中变量loss体现（没有输出这个损失值，只去拟合来使其最小）。
					print ('training epoch %d, %.2f ...' % (t+1, ((i+1) * 100.0 / num_batches)))
					sys.stdout.write("\033[F")
					for m in pred:
						if t == epoch-1:
							pred_train.append(m)
					# print(self.pred_CNN)
				print("%d : loss = %.6f, time = %.3f" % (t+1, loss, time.time() - start_time), end='')		
				print('\n')
				alltrainloss.append(loss)
			np.save('../data/trainpredsCNN.npy', pred_train)
			self.evaluate(test_set, sess)
			print(alltrainloss)
			print("##### THE END OF CNN FITTING #####")


	def evaluate(self, test_set, sess): # 此函数处理测试集
		preds = []
		batch_set = self.get_batch_set(test_set, self.batch_size, False) # 98275 一个batch的数据数量（必须大于测试集样本个数） False：不打散		
		num_batches = int(len(test_set)/self.batch_size) if len(test_set) % self.batch_size == 0 else int(len(test_set)/self.batch_size) + 1
		for i, batch_sample in enumerate(batch_set):
			imgs, labels = zip(*batch_sample)
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
			# print(labelsforCNN)
			batch_loss, pred = sess.run([self.mean_loss_CNN, self.pred_CNN],
											 {self.imagesforCNN: imgs, self.labelsforCNN: labelsforCNN,  self.dropout_keep_prob: 1.0, self.is_training: 1, self.reuse: True})
			print ('testing %.2f ...' % ((i+1) * 100.0 / num_batches))
			for m in pred:
				preds.append(m)
		np.save('../data/testpredsCNN.npy',preds)			
