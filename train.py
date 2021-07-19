from sentiment import Sentiment
from model_fnn1 import Model # modify this line
import tensorflow as tf 

tf.app.flags.DEFINE_string("task", "np_chunking", "Task.")
tf.app.flags.DEFINE_string("cell", "gru", "Rnn cell.") # 第二个参数决定用哪个rnn网络
tf.app.flags.DEFINE_integer("size", 40, "Size of each layer.") # keep 16, do not change this number
tf.app.flags.DEFINE_integer("batch", 100, "Batch size of train set.") # keep 80, do not change this number
tf.app.flags.DEFINE_integer("epoch", 20, "Number of training epoch.") # RNNatt: 20; RNN: 40; CNN: 40; FNN: with 4, without 3
tf.app.flags.DEFINE_string("loss", "cross_entropy", "Loss function.")
tf.app.flags.DEFINE_float("dropout", 0.5, "Dropout keep probability.") # 0.5
tf.app.flags.DEFINE_string("gpu", '0', "GPU id.")
tf.app.flags.DEFINE_string("cpu", '0', "CPU id")
tf.app.flags.DEFINE_string("opt",'Adam','Optimizer.') # Adam
tf.app.flags.DEFINE_float("lr",0.001,'Learning rate.') # keep 0.001, do not change this number

FLAGS = tf.app.flags.FLAGS

data_name = 'yelp13'

def train():
	d = Sentiment(data_name,1) # 这个1定义的是num_class，对应model的build函数中全连接层处，意为最后一个全连接层输出结果的格式。
	with tf.device('/gpu:'+FLAGS.gpu):
		m = Model(data_name, d.num_class, size = FLAGS.size, batch_size = FLAGS.batch, dropout = FLAGS.dropout,
			rnn_cell = FLAGS.cell, optimize = FLAGS.opt, lr = FLAGS.lr)
		m.fit(d.train_set_FNN, d.test_set_FNN, FLAGS.epoch) # modify this line

if __name__ == '__main__':
	train()

# RNN用两种预处理作为两个特征 CNN 用图片 FNN 三个特征