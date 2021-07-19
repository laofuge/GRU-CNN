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
			doc_pad1 = [sentence + [0] * (max_words1_len - len(sentence)) for sentence in doc_pad1]
			doc_pad2 = [sentence + [0] * (max_words2_len - len(sentence)) for sentence in doc_pad2]
			padded_batch_set.append([doc_pad1,doc_pad2,image,label])

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
	def bias_variable(self, shape,scope=None):
		with tf.variable_scope(scope or "cnn5"):
			initial = tf.constant(0.1, shape = shape)
		return tf.Variable(initial)


	def variable_weight(self, shape, stddev = 0.05):
		init = tf.truncated_normal_initializer(stddev = stddev)
		return tf.get_variable(shape = shape, initializer = init, name = 'weight')
	def variable_bias(self, shape):
		init = tf.constant_initializer(0.1)
		return tf.get_variable(shape = shape, initializer = init, name = 'bias')		


	def fullconnect2(self, input_, output_size,scope):
		shape = input_.get_shape().as_list() # 获取input的形状，将input_的形状转为list形式
		input_size = shape[1] # 获取的这个形状的列数，即每个[]内的元素数
		W_fc2 = self.weight_variable([input_size, output_size],scope=scope+'w') # 二维，第一维设置，第二维是输出维度
		b_fc2 = self.bias_variable([output_size],scope=scope+'b')
		h_fc2 = tf.nn.relu(tf.matmul(input_, W_fc2) + b_fc2)
		return tf.nn.dropout(h_fc2, self.dropout)


	def fullconnect3(self, input_, output_size,scope):
		shape = input_.get_shape().as_list() # 获取input的形状，将input_的形状转为list形式
		input_size = shape[1] # 获取的这个形状的列数，即每个[]内的元素数
		W_fc3 = self.weight_variable([input_size, output_size],scope=scope+'w') # 二维，第一维设置，第二维是输出维度
		b_fc3 = self.bias_variable([output_size],scope=scope+'b')
		h_fc3 = tf.matmul(input_, W_fc3) + b_fc3
		return h_fc3


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
			fc_ = tf.reshape(fc_, [self.batch_size, -1])
			arlist.append(fc_)
		for g in range(29):
			out = tf.reshape(arlist[0], [self.batch_size, -1])
			out += arlist[g+1]
		out = tf.reshape(out, [self.batch_size, -1])
		return out


	def linear6(self, input_, output_size, act = tf.nn.relu, scope = 'fully_connect', reuse = None):
		shape = input_.get_shape().as_list() # 获取input的形状，将input_的形状转为list形式 [?, 29, 20]
		input_size = shape[-1] # 20
		print(shape)
		out = []
		with tf.variable_scope(scope+'weight'):
			weight = self.variable_weight([input_size, output_size]) # [20, outsize]
		with tf.variable_scope(scope+'bias'):
			bias = self.variable_bias([output_size])
		with tf.variable_scope(scope+'weight2333'):
			weight1 = self.variable_weight([1, 30, 1])


		fc = tf.multiply(input_, weight1) # [29, outsize]

		out = tf.reduce_sum(fc,1)
		print("########3")
		print('attention: ', out.get_shape().as_list()) # 获取input的形状，将input_的形状转为list形式 [?, 30, 20]
		print("########3")

		return out


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





	def fit(self, train_set, test_set, epoch = 40):		
		pred_train, alltrainloss = [], []
		config = tf.ConfigProto(allow_soft_placement=True) # 自动分配CPU或GPU来运行
		config.gpu_options.allow_growth=True
		with tf.Session(config=config) as sess:
			# sess = tf.InteractiveSession()
			sess.run(tf.global_variables_initializer())
			for t in range(epoch):
				batch_set = self.get_batch_set(train_set, self.batch_size, True)
				loss, start_time = 0.0, time.time()
				num_batches = int(len(train_set)/self.batch_size) if len(train_set) % self.batch_size == 0 else int(len(train_set)/self.batch_size) + 1
				for i, batch_sample in enumerate(batch_set):
					docs1, docs2, images, labels = zip(*batch_sample)
		
					labelsforCNN = []
					labels = list(labels)
					for j in range(len(labels)):
						labels[j] = [labels[j]]
					labelsforCNN = labels

					# docsforRNN = np.array([docsforCNN]).reshape(-1, 30, 6)
					labelsforRNN = np.array([labelsforCNN]).reshape(-1)

					batch_loss, _,pred, dummy_loss = sess.run([self.mean_loss_fusion, self.train_op_fusion, self.pred_fusion, self.mean_loss_fusion],
											 {self.imagesforCNN: images, self.docs1: docs1, self.docs2: docs2, self.labels: labelsforRNN, self.dropout_keep_prob: 0.5, self.is_training: 1})
					loss += dummy_loss # 不是真正的loss，只是用来观察loss是否随迭代缩小。真正的loss在build函数中变量loss体现（没有输出这个损失值，只去拟合来使其最小）。
					print ('training epoch %d, %.2f ...' % (t+1, ((i+1) * 100.0 / num_batches)))
					sys.stdout.write("\033[F")
					for m in pred:
						if t == epoch-1:
							pred_train.append(m)
					# print(self.pred_CNN)
				print("%d : loss = %.8f, time = %.3f" % (t+1, loss, time.time() - start_time), end='')		
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
			labelsforCNN = []
			labels = list(labels)
			for j in range(len(labels)):
				labels[j] = [float(labels[j])]
			labelsforCNN = labels
			labelsforRNN = np.array([labelsforCNN]).reshape(-1)
			# print(labelsforCNN)
			batch_loss, pred = sess.run([self.mean_loss_fusion, self.pred_fusion],
			 								 {self.imagesforCNN: images, self.labelsforCNN: labelsforCNN, self.docs1: docs1, self.docs2: docs2, self.labels: labelsforRNN, self.dropout_keep_prob: 1.0, self.is_training: 1, self.reuse: True})
			print ('testing %.2f ...' % ((i+1) * 100.0 / num_batches))
			for m in pred:
				preds.append(m)
		np.save('../data/testpredsFNN.npy',preds)			



# 
	def build(self):
		self.dropout_keep_prob = tf.placeholder(tf.float32)
		self.imagesforCNN = tf.placeholder(tf.float32, [None, 112, 112, 3])
		self.labelsforCNN = tf.placeholder(tf.float32, [None, 1])
		self.docs1 = tf.placeholder(tf.float32, [None, 30, 6]) 
		self.docs2 = tf.placeholder(tf.float32, [None, 30, 6]) # 三维，分别是batch size（每个batch多少个样本）、一个样本多少天、一天多少个数据。None处也可改为真实的已知数值。
		self.labels = tf.placeholder(tf.float32, [None])
	


# resnet
		self.reuse = tf.placeholder(tf.bool, name = 'reuses')		
		self.is_training = tf.placeholder(tf.bool, name = 'is_training')
		with slim.arg_scope([slim.conv2d], activation_fn = tf.nn.relu, normalizer_fn = slim.batch_norm) as sc:
			conv_scope = sc
		with slim.arg_scope(conv_scope):
			cnn_output2 = self.resnet(self.imagesforCNN, 10, is_training = self.is_training, verbose = True)




		with tf.variable_scope("doc_rnn0"):
			cell = self.get_cell(self.size, self.dropout_keep_prob) # self.dropout_keep_prob指train中的dropout值
			outputs0, state = tf.nn.dynamic_rnn(cell, inputs = self.docs1, dtype = tf.float32)


		with tf.variable_scope("doc_rnn1"):
			cell = self.get_cell(self.size, self.dropout_keep_prob)
			outputs1, state = tf.nn.dynamic_rnn(cell, inputs = outputs0, dtype = tf.float32)


		with tf.variable_scope("doc_rnn2"):
			cell = self.get_cell(self.size, self.dropout_keep_prob) # self.dropout_keep_prob指train中的dropout值
			outputs2, state = tf.nn.dynamic_rnn(cell, inputs = self.docs2, dtype = tf.float32)


		with tf.variable_scope("doc_rnn3"):
			cell = self.get_cell(self.size, self.dropout_keep_prob)
			outputs3, state = tf.nn.dynamic_rnn(cell, inputs = outputs2, dtype = tf.float32)
		# rnn_output1 = outputs1

		rnn_output1 = tf.concat([outputs1, outputs3], axis = -1)
		print(rnn_output1.get_shape().as_list())		

		rnn_output2 = self.linear6(rnn_output1, 80, scope = 'fc14')

		cnn_output31 =  self.fullconnect2(cnn_output2, 500,scope = 'co0')
		cnn_output34 =  self.fullconnect2(cnn_output31, 100,scope = 'co1')
		cnn_output32 = self.fullconnect2(cnn_output34, 25,scope = 'co2')
		cnn_output4 = self.fullconnect3(cnn_output32, 1,scope = 'co3')
		self.mean_loss_c = tf.reduce_mean(tf.square(self.labels - cnn_output4)) ** 0.5

		# rnn_output31 =  self.fullconnect2(rnn_output2, 500,scope = 'ro0')
		rnn_output34 =  self.fullconnect2(rnn_output2, 20,scope = 'ro1')
		rnn_output32 = self.fullconnect2(rnn_output34, 10,scope = 'ro2')
		rnn_output4 = self.fullconnect3(rnn_output32, 1,scope = 'ro3')
		self.mean_loss_r = tf.reduce_mean(tf.square(self.labels - rnn_output4)) ** 0.5


		fusion_output0 = tf.concat([rnn_output2, cnn_output2], axis = -1)
		fusion_output31 =  self.fullconnect2(fusion_output0, 500,scope = 'fo0')
		fusion_output34 =  self.fullconnect2(fusion_output31, 100,scope = 'fo1')
		fusion_output32 = self.fullconnect2(fusion_output34, 25,scope = 'fo2')
		fusion_output4 = self.fullconnect3(fusion_output32, 1,scope = 'fo3')
		self.mean_loss_f = tf.reduce_mean(tf.square(self.labels - fusion_output4)) ** 0.5


		self.pred_fusion = fusion_output4

		self.mean_loss_fusion = 0.1*self.mean_loss_c+0.1*self.mean_loss_r+0.8*self.mean_loss_f

		
		self.train_op_fusion = tf.train.AdamOptimizer(self.lr).minimize(self.mean_loss_fusion)
