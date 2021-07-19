from __future__ import print_function
import numpy as np
import tensorflow as tf
import time
import random
import math
import sys
import pandas as pd
import torch
# import tensorflow.contrib.slim as slim
# from utils.layers import lstm
class Model(object): 
	"""docstring for model"""
	def __init__(self, data_name, num_class, size = 300, batch_size = 64, dropout = 0.5, max_grad_norm = 5.0, L2reg = 0.000001, 
				 rnn_cell = 'lstm', optimize = 'Adagrad', lr = 0.1):
		self.data_name = data_name
		self.num_class = num_class
		# self.shape_weight = shape_weight
		# self.shape_bias = shape_bias
		self.lr = lr
		self.size = size
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
		max_docs_len = max([len(doc1) for doc1,_,_,_,doc2 in data])
		max_words_len = max([max([len(sentence) for sentence in doc1]) for doc1,_,_,_,doc2 in data])
		for doc1, label, _, _, doc2 in data:
			docs_len = len(doc1)
			doc_pad1 = doc1 + [[0]] * (max_docs_len - docs_len)
			words_len = [len(sentence) for sentence in doc_pad1]
			doc_pad1 = [sentence + [0] * (max_words_len - len(sentence)) for sentence in doc_pad1]

			doc_pad2 = doc2 + [[0]] * (max_docs_len - docs_len)
			words_len = [len(sentence) for sentence in doc_pad1]
			doc_pad2 = [sentence + [0] * (max_words_len - len(sentence)) for sentence in doc_pad2]
			padded_batch_set.append([doc_pad1, label, docs_len, words_len, doc_pad2])
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

	def variable_weight(self, shape, stddev = 0.05):
		init = tf.truncated_normal_initializer(stddev = stddev)
		return tf.get_variable(shape = shape, initializer = init, name = 'weight')
	def variable_bias(self, shape):
		init = tf.constant_initializer(0.1)
		return tf.get_variable(shape = shape, initializer = init, name = 'bias')

	# 全连接层
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

	def linear3(self, input_, output_size, act = tf.nn.relu, scope = 'fully_connect', reuse = None):
		shape = input_.get_shape().as_list() # 获取input的形状，将input_的形状转为list形式 [?, 30, 5]
		input_size = shape[-1] # 5
		out = []
		for r in range(self.batch_size):
			input_1 = tf.slice(input_, [r, 0, 0], [1, -1, -1]) # tf.slice(inputs, begin, size, name)
			input_1 = tf.reshape(input_1, [shape[1], shape[2]])		
			# with tf.variable_scope(scope, reuse = tf.AUTO_REUSE):
			with tf.variable_scope('weight' + str(r)):
				weight = self.variable_weight([input_size, output_size]) # [5, outsize]
			with tf.variable_scope('bias' + str(r)):
				bias = self.variable_bias([output_size])
			out.append(tf.nn.bias_add(tf.matmul(input_1, weight), bias, name = 'fc' + str(r)))
		out = tf.reshape(out, [self.batch_size, 30, output_size])
				# out = tf.nn.dropout(out, self.dropout_keep_prob)
		return out	    


	def linear4(self, input_, output_size, act = tf.nn.relu, scope = 'fully_connect', reuse = None):
		shape = input_.get_shape().as_list() # 获取input的形状，将input_的形状转为list形式 [?, 29, 20]
		input_size = shape[-1] # 20
		print(shape)
		out = []
		# with tf.variable_scope('weight'):
		# 	weight = self.variable_weight([input_size, output_size]) # [20, outsize]
		with tf.variable_scope('bias23'):
			bias = self.variable_bias([output_size])
		with tf.variable_scope('weight23313'):
			# weight1 = np.array([[0.06, 0.06, 0.06, 0.06, 0.06, 0.05, 0.05, 0.05, 0.05, 0.05, 0.04, 0.04, 0.04, 0.04, 0.04, 0.03, 0.03, 0.03, 0.03, 0.03, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]])
			
			weight1 = self.variable_weight([1, 30])
		# if shape[0] == self.batch_size:
		for r in range(self.batch_size):
			input_1 = tf.slice(input_, [r, 0, 0], [1, -1, -1]) # tf.slice(inputs, begin, size, name)
			input_1 = tf.reshape(input_1, [shape[1], shape[2]]) # [29, 20]

		# fc = tf.multiply(input_, weight1) # [29, outsize]
		# print(fc.get_shape().as_list()) # 获取input的形状，将input_的形状转为list形式 [?, 29, 20]
			
			fc1 = tf.nn.bias_add(tf.matmul(tf.cast(weight1, tf.float32), input_1), bias, name = 'fc') # [1, outsize]
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
			fc_ = tf.reshape(fc_, [self.batch_size, -1])
			arlist.append(fc_)
		for g in range(29):
			out = tf.reshape(arlist[0], [self.batch_size, -1])
			out += arlist[g+1]
		out = tf.reshape(out, [self.batch_size, -1])
		return out
		

	def build(self):
		self.docs1 = tf.placeholder(tf.float32, [None, 30, 6]) 
		# self.docs2 = tf.placeholder(tf.float32, [None, 30, 6]) # 三维，分别是batch size（每个batch多少个样本）、一个样本多少天、一天多少个数据。None处也可改为真实的已知数值。
		# self.docs_am = tf.placeholder(tf.float32, [None, 30, 2])
		# self.docs_ol = tf.placeholder(tf.float32, [None, 30, 2])

		self.labels = tf.placeholder(tf.float32, [None])
		self.dropout_keep_prob = tf.placeholder(tf.float32)
		# fcdocs = tf.reshape(self.docs,[-1,6])
		# weightfc = self.variable_weight([6, 6])
		# fcdocs = tf.matmul(fcdocs,weightfc)
		# fcdocs=tf.reshape(fcdocs,[-1,30,6])


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



		print('outputs1: ', outputs1.get_shape().as_list())
		# docs_embed = tf.reshape(outputs1, [-1, 30*self.batch_size])
		docs_embed = outputs1[:,-1] # 最后一列：最后一个时间节点 # [batchsize, cellnum]
		# docs_embed = tf.reduce_mean(outputs1, 1) # 取平均求和，1为计算每个[]内元素的均值再组合，0为计算每个[]内第i个元素均值再组合
		# docs_embed =  tf.concat([outputs1, outputs3], axis = -1)
		# docs_embed =  outputs1


		print('after concat: ', docs_embed.get_shape().as_list()) # [None, 30, 20]

		# docs_embed = self.linear5(docs_embed, 40, scope = 'fc14') # 加权则有这一层，30
		print(docs_embed.get_shape().as_list())
		docs_embed = self.linear1(docs_embed, 20, scope = 'fc11')
		docs_embed = self.linear0(docs_embed, 10, scope = 'fc12') # 调用全连接层
		docs_embed = self.linear2(docs_embed, self.num_class, scope = 'fc11')
		print(docs_embed.get_shape().as_list())
		self.pred = docs_embed
		# self.pred = tf.multiply(docs_embed[:, 0], docs_embed[:, 1])
		self.mean_loss = tf.reduce_mean(tf.square(self.labels - self.pred)) ** 0.5
		# self.mean_loss = tf.reduce_mean(tf.abs(self.labels - self.pred))
		# self.train_op = tf.train.AdamOptimizer(self.lr, 0.9, 0.999).minimize(self.mean_loss)
		self.train_op = tf.train.AdagradOptimizer(self.lr).minimize(self.mean_loss)

		# self.train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.mean_loss)

	def fit(self, train_set, test_set, epoch = 40): 
		pred_train, alltrainloss = [], []
		config = tf.ConfigProto(allow_soft_placement=True) # 自动分配CPU或GPU来运行
		config.gpu_options.allow_growth=True
		# config.gpu_options.per_process_gpu_memory_fraction = 0.4
		with tf.Session(config=config) as sess:
			sess.run(tf.global_variables_initializer())
			# f_log = open(self.log_file, 'w')
			for j in range(epoch):
				batch_set = self.get_batch_set(train_set, self.batch_size, True)
				loss, start_time = 0.0, time.time()
				num_batches = int(len(train_set)/self.batch_size) if len(train_set) % self.batch_size == 0 else int(len(train_set)/self.batch_size) + 1
				loss_list = []
				for i, batch_sample in enumerate(batch_set):
					docs1, labels, doc_len, word_len, docs2 = zip(*batch_sample)
					# docs11 = np.array(docs1)
					# docs = list((docs11[:, :,:4]).reshape(-1, 30, 4))
					# docs_am = list(docs11[:, :, 4:])
					# docs_ol = list((docs11[:, :,2:4]).reshape(-1, 30, 2))

					# print(docs)
					#print X, '\n', Y, '\n', length, '\n', mask, '\n', self.dropout, '\n'
					# print(labels)
					#return
					batch_loss, _,pred = sess.run([self.mean_loss, self.train_op, self.pred],
											 {self.docs1: docs1, self.labels: labels, self.dropout_keep_prob: self.dropout}) # 前面[]内是输出，后面{}内是输入
					loss += batch_loss # 不是真正的loss，只是用来观察loss是否随迭代缩小。真正的loss在build函数中变量loss体现（没有输出这个损失值，只去拟合来使其最小）。
					# print(pred)
					print ('training epoch %d, %.2f ...' % (j+1, ((i+1) * 100.0 / num_batches)))
					sys.stdout.write("\033[F")
					for m in pred:
						if j == epoch-1:
							pred_train.append(m)					
					# loss_list.append(loss)
				print("%d : loss = %.12f, time = %.3f" % (j+1, loss, time.time() - start_time), end='')
				# print(pred)
				print ('\n')
				alltrainloss.append(loss)
				# time.sleep(1)
			# pred = [i[0] for i in pred] # 去大列表中括号
			# print(pred)
			# print('loss: ', loss_list)
			np.save('../data/trainpredsRNN.npy', pred_train)
			self.evaluate(test_set, sess)
			print(alltrainloss)			
			print("##### THE END OF RNN FITTING #####")
			# print(labels)


	def evaluate(self, test_set, sess): # 此函数处理测试集
		batch_set = self.get_batch_set(test_set, self.batch_size, False) # 98275 一个batch的数据数量（必须大于测试集样本个数） False：不打散
		preds = []
		print ('testing...')
		for batch_sample in batch_set:
			docs1, labels, doc_len, word_len, docs2 = zip(*batch_sample)
			# docs11 = np.array(docs1)
			# docs = list((docs11[:, :, :4]).reshape(-1, 30, 4))
			# # docs_ol = list((docs11[:, :, 2:4]).reshape(-1, 30, 2))
			# docs_am = list(docs11[:, :, 4:])
			loss = 0.0
			batch_loss, pred = sess.run([self.mean_loss, self.pred],
			 								 {self.docs1: docs1, self.labels: labels, self.dropout_keep_prob: 1.0})
			loss += batch_loss
			for m in pred:
				preds.append(m)
		# print(pred)
		np.save('../data/testpredsRNN.npy',preds)
		print("loss = %.8f" % loss, end='')