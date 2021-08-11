import numpy as np
import random
#import h5py
import os
import time
import collections
import cv2
from PIL import Image
import glob
import tensorflow as tf
# from embeddings import Embeddings 

class Sentiment(object):
	"""NP_chunking data preparation"""
	def __init__(self, data_name, num_class=5):
		self.data_name = data_name
		self.train_data_path1 = '../data/train.txt'
		self.train_data_path2 = '../data/train2.txt'
		self.test_data_path1 = '../data/test.txt'
		self.test_data_path2 = '../data/test2.txt'
		self.train_img_path = '../pic/pictrain/'
		self.test_img_path = '../pic/pictest/'
		self.labels_path = '../data/labels.txt'
		self.trainlabels_path = '../data/trainlabel.txt'
		self.testlabels_path = '../data/testlabel.txt'
		# self.train_data_path = '../data/val_train.txt'
		# self.test_data_path = '../data/val_validation.txt'
		self.num_class = num_class
		start_time = time.time() 
		# self.load_data()
		# self.load_data_CNN()
		self.load_data_FNN()


		print ('Reading datasets comsumes %.3f seconds' % (time.time()-start_time))
			
	def deal_with_data(self, path1, path2):
		users, products, labels, docs1, len_docs, len_words, docs2 = [], [], [], [], [], [], []
		k = 0
		for line in open(path1, 'r', encoding='UTF-8'):
			tokens = line.strip().split('\t\t')
			users.append(tokens[0])
			products.append(tokens[1])
			labels.append(float(tokens[2]))
			doc1 = tokens[3].strip().split('<sssss>')
			len_docs.append(len(doc1))
			doc1 = [sentence.strip().split('  ') for sentence in doc1]
			len_words.append([len(sentence) for sentence in doc1])
			docs1.append(doc1)
			k += 1
		for line in open(path2, 'r', encoding='UTF-8'):
			tokens = line.strip().split('\t\t')
			doc2 = tokens[3].strip().split('<sssss>')
			doc2 = [sentence.strip().split('  ') for sentence in doc2]
			len_words.append([len(sentence) for sentence in doc2])
			docs2.append(doc2)
			k += 1
		# print(labels)
		return users, products, labels, docs1, docs2
		# return images
	def load_data(self):
		train_users, train_products, train_labels, train_docs1, train_docs2 = self.deal_with_data(self.train_data_path1, self.train_data_path2)
		test_users, test_products, test_labels, test_docs1, test_docs2 = self.deal_with_data(self.test_data_path1, self.test_data_path2)

		self.train_set = list(zip(train_docs1, train_labels, train_users, train_products, train_docs2))
		self.test_set = list(zip(test_docs1, test_labels, test_users, test_products, test_docs2))
# a=Sentiment(object)
# print(a.deal_with_data('../data/train.txt', '../pic/pictrain'))

	def deal_with_data_CNN(self, path1, path2):
		users, products, labels, images = [], [], [], []
		for line in open (path1, 'r', encoding='UTF-8'):
			labels.append(line)

		imglist = os.listdir(path2)
		for imgname in imglist:
			img_path = path2 + imgname
			# print(img_path)
			image = cv2.imread(img_path)
			images.append(image)
		return labels, images

	def load_data_CNN(self):
		train_labels, train_images = self.deal_with_data_CNN(self.trainlabels_path, self.train_img_path)
		test_labels, test_images = self.deal_with_data_CNN(self.testlabels_path, self.test_img_path)

		self.train_set_CNN = list(zip(train_images, train_labels))
		self.test_set_CNN = list(zip(test_images, test_labels))

	def deal_with_data_FNN(self, path1, path2, path3, path4):
		docs1, docs2, images, labels = [], [], [], []
		for line in open (path1, 'r', encoding='UTF-8'):
			labels.append(line)

		imglist = os.listdir(path2)
		for imgname in imglist:
			img_path = path2 + imgname
			# print(img_path)
			image = cv2.imread(img_path)
			images.append(image)

		for line in open(path3, 'r', encoding='UTF-8'):
			tokens = line.strip().split('\t\t')
			doc1 = tokens[3].strip().split('<sssss>')
			doc1 = [sentence.strip().split('  ') for sentence in doc1]
			# len_words.append([len(sentence) for sentence in doc1])
			docs1.append(doc1)

		for line in open(path4, 'r', encoding='UTF-8'):
			tokens = line.strip().split('\t\t')
			doc2 = tokens[3].strip().split('<sssss>')
			doc2 = [sentence.strip().split('  ') for sentence in doc2]
			# len_words.append([len(sentence) for sentence in doc2])
			docs2.append(doc2)
		return labels, images, docs1, docs2

	def load_data_FNN(self):
		train_labels, train_images, train_docs1, train_docs2 = self.deal_with_data_FNN(self.trainlabels_path, self.train_img_path, self.train_data_path1, self.train_data_path2)
		test_labels, test_images, test_docs1, test_docs2 = self.deal_with_data_FNN(self.testlabels_path, self.test_img_path, self.test_data_path1, self.test_data_path2)

		self.train_set_FNN = list(zip(train_docs1, train_docs2, train_images, train_labels))
		self.test_set_FNN = list(zip(test_docs1, test_docs2, test_images, test_labels))
