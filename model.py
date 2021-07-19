from __future__ import print_function
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
		max_docs_len = max([len(doc) for doc,_,_,_ in data])
		max_words_len = max([max([len(sentence) for sentence in doc]) for doc,_,_,_ in data])
		for doc, label, _, _ in data:
			docs_len = len(doc)
			doc_pad = doc + [[0]] * (max_docs_len - docs_len)
			words_len = [len(sentence) for sentence in doc_pad]
			doc_pad = [sentence + [0] * (max_words_len - len(sentence)) for sentence in doc_pad]
			padded_batch_set.append([doc_pad, label, docs_len, words_len])
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


	# 全连接层
	def linear0(self, input_, output_size, scope=None):
	    shape = input_.get_shape().as_list() 
	    # if len(shape) != 2:
	    #     raise ValueError("Linear is expecting 2D arguments: %s" % str(shape))
	    # if not shape[1]:
	    #     raise ValueError("Linear expects shape[1] of arguments: %s" % str(shape))
	    input_size = shape[1]
	    with tf.variable_scope(scope or "SimpleLinear1"):
	        matrix1 = tf.get_variable("Matrix1", [output_size, input_size], initializer=tf.constant_initializer(0.1),dtype=input_.dtype)
	        bias_term1 = tf.get_variable("Bias1", [output_size], dtype=input_.dtype)
	    return tf.matmul(input_, tf.transpose(matrix1)) + bias_term1
	    # tf.multiply
	    # with tf.variable_scope(scope or "SimpleLinear0"):
	    #     matrix1 = tf.get_variable("Matrix1", [1,30], initializer=tf.constant_initializer(0.09),dtype=input_.dtype) # 初始化一般在0.01-0.1内。
	    #     bias_term1 = tf.get_variable("Bias1", [output_size], dtype=input_.dtype)
	    #     self.summulti = tf.placeholder(tf.float32, [None, None])
	    #     self.multi = []
	    #     for i in input_:
	    #     	i = tf.matmul(matrix1, input_) + bias_term1
	        # for i in range(2000):
	        # 	input_1 = tf.slice(input_, [i, 0, 0], [1, -1, -1]) # tf.slice(inputs, begin, size, name)
	        # 	input_1 = tf.reshape(input_1, [30, 32])
	        # 	self.multi.append(tf.matmul(matrix1, input_1)) # [1, 30] * [30, 32] = [1, 32]
	        # 	print(i) if i % 100 == 0
	        # self.summulti = tf.reshape(self.multi, [-1, 20]) # [None, 32]
	    # return input_


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
	    with tf.variable_scope(scope or "SimpleLinear3"):
	        matrix1 = tf.get_variable("Matrix3", [output_size, input_size], initializer=tf.constant_initializer(0.09),dtype=input_.dtype)
	        bias_term1 = tf.get_variable("Bias3", [output_size], dtype=input_.dtype)
	    return tf.matmul(input_, tf.transpose(matrix1)) + bias_term1	    


	def linear4(self, input_, output_size, scope=None):
	    shape = input_.get_shape().as_list() # 获取input的形状，将input_的形状转为list形式
	    input_size = shape[1] # 获取的这个形状的列数，即每个[]内的元素数

	    # Now the computation.
	    with tf.variable_scope(scope or "SimpleLinear4"):
	        matrix1 = tf.get_variable("Matrix3", [output_size, input_size], initializer=tf.constant_initializer(0.09),dtype=input_.dtype)
	        bias_term1 = tf.get_variable("Bias3", [output_size], dtype=input_.dtype)
	    return tf.matmul(input_, tf.transpose(matrix1)) + bias_term1


	def linear5(self, input_, output_size, scope=None):
	    shape = input_.get_shape().as_list() # 获取input的形状，将input_的形状转为list形式
	    input_size = shape[1] # 获取的这个形状的列数，即每个[]内的元素数

	    # Now the computation.
	    with tf.variable_scope(scope or "SimpleLinear5"):
	        matrix1 = tf.get_variable("Matrix3", [output_size, input_size], initializer=tf.constant_initializer(0.09),dtype=input_.dtype)
	        bias_term1 = tf.get_variable("Bias3", [output_size], dtype=input_.dtype)
	    return tf.matmul(input_, tf.transpose(matrix1)) + bias_term1


	def weight_variable(self, shape):
		initial = tf.truncated_normal(shape, stddev = 0.1)
		return tf.Variable(initial)
	def bias_variable(self, shape):
		initial = tf.constant(0.1, shape = shape)
		return tf.Variable(initial)
	def conv2d(self, x, W):
		return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME')
	def max_pool_2x2(self, x):
		return tf.nn.max_pool(x, ksize = [1, 5, 1, 1], strides = [1, 1, 1, 1], padding = 'SAME')


	def convolution1(self, kernel, bias, x_image):
		W_conv1 = self.weight_variable(kernel)
		b_conv1 = self.bias_variable(bias)
		h_conv1 = tf.nn.relu(self.conv2d(x_image, W_conv1) + b_conv1)
		return self.max_pool_2x2(h_conv1)


	def convolution2(self, kernel, bias, x_image):
		W_conv2 = self.weight_variable(kernel)
		b_conv2 = self.bias_variable(bias)
		h_conv2 = tf.nn.relu(self.conv2d(x_image, W_conv2) + b_conv2)
		return self.max_pool_2x2(h_conv2)


	def CNNfullconnect1(self, input_, output_size):
		shape = input_.get_shape().as_list() # 获取input的形状，将input_的形状转为list形式
		# print(input_.get_shape().as_list())
		input_size = shape[1] * shape [2] # 获取的这个形状的列数，即每个[]内的元素数：30*6 注意：这个维度会被消掉，所以设为最匹配两边矩阵的数
		W_fc1 = self.weight_variable([input_size, output_size])
		b_fc1 = self.bias_variable([output_size])
		input_flat = tf.reshape(input_, [-1, input_size])
		h_fc1 = tf.nn.relu(tf.matmul(input_flat, W_fc1) + b_fc1)
		# keep_prob = tf.placeholder(tf.float32)
		return tf.nn.dropout(h_fc1, self.dropout)


	def CNNfullconnect2(self, input_, output_size):
		shape = input_.get_shape().as_list() # 获取input的形状，将input_的形状转为list形式
		input_size = shape[1] # 获取的这个形状的列数，即每个[]内的元素数
		W_fc2 = self.weight_variable([input_size, output_size]) # 二维，第一维设置，第二维是输出维度
		b_fc2 = self.bias_variable([output_size])
		h_fc2 = tf.nn.relu(tf.matmul(input_, W_fc2) + b_fc2)
		return tf.nn.dropout(h_fc2, self.dropout)


	def CNNfullconnect3(self, input_, output_size):
		shape = input_.get_shape().as_list() # 获取input的形状，将input_的形状转为list形式
		input_size = shape[1] # 获取的这个形状的列数，即每个[]内的元素数
		W_fc3 = self.weight_variable([input_size, output_size]) # 二维，第一维设置，第二维是输出维度
		b_fc3 = self.bias_variable([output_size])
		h_fc3 = tf.nn.relu(tf.matmul(input_, W_fc3) + b_fc3)
		return tf.nn.dropout(h_fc3, self.dropout)

		
	def build(self):
		self.docs = tf.placeholder(tf.float32, [None, 30, 6]) # 三维，分别是batch size（每个batch多少个样本）、一个样本多少天、一天多少个数据。None处也可改为真实的已知数值。
		self.labels = tf.placeholder(tf.float32, [None])
		self.word_len = tf.placeholder(tf.int32, [None, None])
		self.doc_len = tf.placeholder(tf.int32, [None])
		self.dropout_keep_prob = tf.placeholder(tf.float32)
		self.docsforCNN = tf.placeholder(tf.float32, [None, 30, 6, 1])
		self.labelsforCNN = tf.placeholder(tf.float32, [None, 1])

		# self.docsforCNN = tf.reshape(self.docs, [-1, 30, 6, 1])
		# self.labelsforCNN = tf.reshape(self.labels, [-1, 1])

# # '''///////////////////////////                   RNN ONLY                   ///////////////////////////'''
		
# 		with tf.variable_scope("doc_rnn0"):
# 			cell = self.get_cell(self.size, self.dropout_keep_prob) # self.dropout_keep_prob指train中的dropout值
# 			outputs0, state = tf.nn.dynamic_rnn(cell, inputs = self.docs , sequence_length = self.doc_len, dtype = tf.float32)
# 			# print(outputs0.get_shape().as_list()) # shape的第三个数是size of each layer

# 		with tf.variable_scope("doc_rnn1"):
# 			cell = self.get_cell(self.size, 0.6)
# 			outputs1, state = tf.nn.dynamic_rnn(cell, inputs = outputs0, sequence_length = self.doc_len, dtype = tf.float32)
		
# 		# print(outputs1.get_shape().as_list()) # [None, 30, 32]
		
# 		# with tf.variable_scope("doc_rnn2"):
# 		# 	cell = self.get_cell(self.size, 0.5)
# 		# 	outputs2, state = tf.nn.dynamic_rnn(cell, inputs = outputs1, sequence_length = self.doc_len, dtype = tf.float32)
# 			# print(outputs2.get_shape().as_list())

# 		docs_embed = outputs1[:,-1] # 最后一列：最后一个时间节点
# 		# docs_embed = tf.reduce_mean(outputs1, 1) # 取平均求和，1为计算每个[]内元素的均值再组合，0为计算每个[]内第i个元素均值再组合
# 		# print(docs_embed.get_shape().as_list())

# 		docs_embed = self.linear0(docs_embed, 16)
# 		# print(docs_embed.get_shape().as_list())
# 		docs_embed = self.linear1(docs_embed, 4)
# 		docs_embed = self.linear2(docs_embed, self.num_class) # 调用全连接层
# 		self.pred = docs_embed
# 		loss=tf.abs(self.labels-docs_embed)
# 		self.mean_loss = tf.reduce_mean(loss)

# 		tvars = tf.trainable_variables()
# 		grads, _ = tf.clip_by_global_norm(tf.gradients(self.mean_loss, tvars), self.max_grad_norm)
# 		if self.optimize == 'Adagrad':
# 			optimizer = tf.train.AdagradOptimizer(self.lr)
# 		else:
# 			optimizer = tf.train.AdamOptimizer(self.lr)
# 		self.train_op = optimizer.apply_gradients(zip(grads, tvars))

# '''///////////////////////////                    CNN ONLY                  ///////////////////////////'''

		cnn_output = self.convolution1([2, 2, 1, 32], [32], self.docsforCNN)
		cnn_output = self.convolution2([2, 2, 32, 1], [1], cnn_output)
		# print(cnn_output.get_shape().as_list()) # [None, 6, 30, 1]

		cnn_output1 = self.CNNfullconnect1(cnn_output, 30)
		# cnn_output = self.CNNfullconnect2(cnn_output, 32)
		# print(cnn_output.get_shape().as_list())
		cnn_output2 = self.CNNfullconnect2(cnn_output1, 6)
		cnn_output3 = self.CNNfullconnect2(cnn_output2, 1)
		# print(cnn_output.get_shape().as_list())

		self.pred_CNN = cnn_output3

		loss_CNN=tf.abs(self.labelsforCNN - cnn_output3)
		self.mean_loss_CNN = tf.reduce_mean(loss_CNN)

		tvars_CNN = tf.trainable_variables()
		grads_CNN, _ = tf.clip_by_global_norm(tf.gradients(self.mean_loss_CNN, tvars_CNN), self.max_grad_norm)
		
		if self.optimize == 'Adagrad':
			optimizer = tf.train.AdagradOptimizer(self.lr)
		else:
			optimizer = tf.train.AdamOptimizer(self.lr)
		# self.train_op_CNN = optimizer.apply_gradients(zip(grads_CNN, tvars_CNN))
		
		self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = self.labelsforCNN, logits = self.pred_CNN))
		self.train_op_CNN = tf.train.AdamOptimizer(self.lr).minimize(self.cross_entropy)

# '''///////////////////////////                    RNN & CNN                  ///////////////////////////'''
		
		# docs_fusion = tf.concat([outputs1[:,-1] + cnn_output], axis = 1) # 最后一列：最后一个时间节点
		# # print(docs_fusion.get_shape().as_list())
		# docs_fusion = self.linear3(docs_fusion, 16)
		# docs_fusion = self.linear4(docs_fusion, 4)
		# docs_fusion = self.linear5(docs_fusion, self.num_class)
		# self.pred_fusion = docs_fusion

		# loss_fusion = tf.abs(self.labels - docs_fusion)
		# self.mean_loss_fusion = tf.reduce_mean(loss_fusion)

		# # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = docs_embed, labels = self.labels)
		# # loss=tf.square(self.labels-docs_embed)
		# # loss=tf.losses.mean_squared_error(self.labels,docs_embed)
		
		# tvars = tf.trainable_variables()
		# grads_fusion, _ = tf.clip_by_global_norm(tf.gradients(self.mean_loss_fusion, tvars), self.max_grad_norm)
		# if self.optimize == 'Adagrad':
		# 	optimizer_fusion = tf.train.AdagradOptimizer(self.lr)
		# else:
		# 	optimizer_fusion = tf.train.AdamOptimizer(self.lr)
		# self.train_op_fusion = optimizer_fusion.apply_gradients(zip(grads_fusion, tvars))



# fit函数用来拟合训练集。注意：sess.run小括号内两个中括号，第一个中括号调用了train_op，对应build函数最末的self.train_op。当运行train时，直接导致运行fit和build，这两个函数中所用的包在其他函数中已被定义，实现模型。
	def fit_rnn(self, train_set, test_set, epoch = 40): 
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
					docs, labels, doc_len, word_len = zip(*batch_sample)
					#print X, '\n', Y, '\n', length, '\n', mask, '\n', self.dropout, '\n'
					#return
					batch_loss, _,pred = sess.run([self.mean_loss, self.train_op,self.pred],
											 {self.docs: docs, self.labels: labels, self.doc_len: doc_len, 
											 self.word_len: word_len, self.dropout_keep_prob: self.dropout}) # 前面[]内是输出，后面{}内是输入
					loss += batch_loss # 不是真正的loss，只是用来观察loss是否随迭代缩小。真正的loss在build函数中变量loss体现（没有输出这个损失值，只去拟合来使其最小）。
					# print(pred)
					print ('training epoch %d, %.2f ...' % (j+1, ((i+1) * 100.0 / num_batches)))
					sys.stdout.write("\033[F")
					# loss_list.append(loss)
				print("%d : loss = %.3f, time = %.3f" % (j+1, loss, time.time() - start_time), end='')
				# print(pred)
				print ('\n')
				time.sleep(1)
			pred = [i[0] for i in pred] # 去大列表中括号
			# print(pred)
			# print('loss: ', loss_list)
			self.evaluate_rnn(test_set, sess)
			print("##### THE END OF RNN FITTING #####")
			# print(labels)


	def fit_cnn(self, train_set, test_set, epoch = 40):		
		pred_train = []
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
					docs, labels, doc_len, word_len = zip(*batch_sample)
					for j in range(len(docs)):
						for k in range(len(docs[j])):
							for l in range(len(docs[j][k])):
								docs[j][k][l] = [docs[j][k][l]]
					docsforCNN = docs			
					labelsforCNN = []
					labels = list(labels)
					for j in range(len(labels)):
						labels[j] = [labels[j]]
					labelsforCNN = labels
					#print X, '\n', Y, '\n', length, '\n', mask, '\n', self.dropout, '\n'
					#return
					batch_loss, _,pred, dummy_loss = sess.run([self.cross_entropy, self.train_op_CNN, self.pred_CNN, self.mean_loss_CNN],
											 {self.docsforCNN: docsforCNN, self.labelsforCNN: labelsforCNN, self.doc_len: doc_len, 
											 self.word_len: word_len, self.dropout_keep_prob: 0.9})
					loss += dummy_loss # 不是真正的loss，只是用来观察loss是否随迭代缩小。真正的loss在build函数中变量loss体现（没有输出这个损失值，只去拟合来使其最小）。
					print ('training epoch %d, %.2f ...' % (t+1, ((i+1) * 100.0 / num_batches)))
					sys.stdout.write("\033[F")
					# print(self.pred_CNN)
				print("%d : loss = %.3f, time = %.3f" % (t+1, loss, time.time() - start_time), end='')		
				print('\n')
				time.sleep(1)
			self.evaluate_cnn(test_set, sess)
			print("##### THE END OF CNN FITTING #####")


	def fit_fusion(self, train_set, test_set, epoch = 40):		
		config = tf.ConfigProto(allow_soft_placement=True) # 自动分配CPU或GPU来运行
		config.gpu_options.allow_growth=True
		with tf.Session(config=config) as sess:
			# sess = tf.InteractiveSession()
			sess.run(tf.global_variables_initializer())
			for i in range(1):
				batch_set = self.get_batch_set(train_set, self.batch_size, True)
				loss, start_time = 0.0, time.time()
				num_batches = int(len(train_set)/self.batch_size) if len(train_set) % self.batch_size == 0 else int(len(train_set)/self.batch_size) + 1
				for i, batch_sample in enumerate(batch_set):
					docs, labels, doc_len, word_len = zip(*batch_sample)
					for j in range(len(docs)):
						for k in range(len(docs[j])):
							for l in range(len(docs[j][k])):
								docs[j][k][l] = [docs[j][k][l]]
					docsforCNN = docs
					# print(docsforCNN)			
					labelsforCNN = []
					labels = list(labels)
					for j in range(len(labels)):
						labels[j] = [labels[j]]
					labelsforCNN = labels
					# print(labelsforCNN)
					#print X, '\n', Y, '\n', length, '\n', mask, '\n', self.dropout, '\n'
					#return
					batch_loss, _,pred = sess.run([self.mean_loss_fusion, self.train_op_fusion, self.pred_fusion],
											 {self.docs: docs, self.labels: labels, self.docsforCNN: docsforCNN, self.labelsforCNN: labelsforCNN, self.doc_len: doc_len, 
											 self.word_len: word_len, self.dropout_keep_prob: self.dropout}) # 前面[]内是输出，后面[]内是输入
					loss += batch_loss # 不是真正的loss，只是用来观察loss是否随迭代缩小。真正的loss在build函数中变量loss体现（没有输出这个损失值，只去拟合来使其最小）。
					# print(pred)
					print ('training %.2f ...' % ((i+1) * 100.0 / num_batches))
					sys.stdout.write("\033[F")
				print("%d : loss = %.5f, time = %.3f" % (j+1, loss, time.time() - start_time), end='')
				# print(pred)
				print ('\n')
				time.sleep(5)				
			# print(pred)
			self.evaluate_fusion(test_set, sess)
			print("##### THE END OF FUSION FITTING #####")
			# print(labels)


	def evaluate_rnn(self, test_set, sess): # 此函数处理测试集
		batch_set = self.get_batch_set(test_set, 98275, False) # 98275 一个batch的数据数量（必须大于测试集样本个数） False：不打散
		total_correct = []
		for batch_sample in batch_set:
			docs, labels, doc_len, word_len = zip(*batch_sample)

			batch_loss, pred = sess.run([self.mean_loss, self.pred],
			 								 {self.docs: docs, self.labels: labels, self.doc_len: doc_len, 
			 								 self.word_len: word_len, self.dropout_keep_prob: 1.0})
			# print(pred)
			np.save('../data/testpredsRNN.npy',pred)
		# preds = []
		# batch_set = self.get_batch_set(test_set, self.batch_size, False) # 98275 一个batch的数据数量（必须大于测试集样本个数） False：不打散
		# num_batches = int(len(test_set)/self.batch_size) if len(test_set) % self.batch_size == 0 else int(len(test_set)/self.batch_size) + 1
		# for i, batch_sample in enumerate(batch_set):
		# 	docs, labels, doc_len, word_len = zip(*batch_sample)
		# 	batch_loss, pred = sess.run([self.mean_loss, self.pred],
		# 	 								 {self.docs: docs, self.labels: labels, self.doc_len: doc_len, 
		# 	 								 self.word_len: word_len, self.dropout_keep_prob: 1.0})
		# 	print ('testing %.2f ...' % ((i+1) * 100.0 / num_batches))
		# 	for m in pred:
		# 		preds.append(m)	
		# np.save('../data/testpredsRNN.npy', preds)	


	def evaluate_cnn(self, test_set, sess): # 此函数处理测试集
		preds = []
		batch_set = self.get_batch_set(test_set, self.batch_size, False) # 98275 一个batch的数据数量（必须大于测试集样本个数） False：不打散		
		num_batches = int(len(test_set)/self.batch_size) if len(test_set) % self.batch_size == 0 else int(len(test_set)/self.batch_size) + 1
		for i, batch_sample in enumerate(batch_set):
			docs, labels, doc_len, word_len = zip(*batch_sample)
			for j in range(len(docs)):
				for k in range(len(docs[j])):
					for l in range(len(docs[j][k])):
						docs[j][k][l] = [float(docs[j][k][l])]
			docsforCNN = docs
			# print(docsforCNN)			
			labelsforCNN = []
			labels = list(labels)
			for j in range(len(labels)):
				labels[j] = [float(labels[j])]
			labelsforCNN = labels
			# print(labelsforCNN)
			batch_loss, pred = sess.run([self.cross_entropy, self.pred_CNN],
			 								 {self.docsforCNN: docsforCNN, self.labelsforCNN: labelsforCNN, self.doc_len: doc_len, 
			 								 self.word_len: word_len, self.dropout_keep_prob: 1.0})
			print ('testing %.2f ...' % ((i+1) * 100.0 / num_batches))
			for m in pred:
				preds.append(m)
		np.save('../data/testpredsCNN.npy',preds)			


	def evaluate_fusion(self, test_set, sess): # 此函数处理测试集
		preds = []
		batch_set = self.get_batch_set(test_set, 64, False) # 98275 一个batch的数据数量（必须大于测试集样本个数） False：不打散		
		num_batches = int(len(test_set)/self.batch_size) if len(test_set) % self.batch_size == 0 else int(len(test_set)/self.batch_size) + 1
		for i, batch_sample in enumerate(batch_set):
			docs, labels, doc_len, word_len = zip(*batch_sample)
			for j in range(len(docs)):
				for k in range(len(docs[j])):
					for l in range(len(docs[j][k])):
						docs[j][k][l] = [float(docs[j][k][l])]
			docsforCNN = docs
			# print(docsforCNN)			
			labelsforCNN = []
			labels = list(labels)
			for j in range(len(labels)):
				labels[j] = [float(labels[j])]
			labelsforCNN = labels
			# print(labelsforCNN)
			batch_loss, pred = sess.run([self.mean_loss_fusion, self.pred_fusion],
			 								 {self.docs: docs, self.labels: labels, self.docsforCNN: docsforCNN, self.labelsforCNN: labelsforCNN, self.doc_len: doc_len, 
			 								 self.word_len: word_len, self.dropout_keep_prob: 1.0})
			print ('testing %.2f ...' % ((i+1) * 100.0 / num_batches))
			for m in pred:
				preds.append(m)
		np.save('../data/testpredsFUSION.npy',preds)					

		



		
