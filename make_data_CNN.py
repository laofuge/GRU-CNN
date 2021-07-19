import pandas as pd
import numpy as np
import csv
import tensorflow as tf


datalist=[]
with open('../data/tw_spydata_raw.csv',newline='') as g:
	reader=csv.reader(g)
	for row in reader:
		line=row[1:]
		# print(line)
		datalist.append(line)
# print(datalist)


t_list, label=[], []
with open('../data/initial_dataset_forCNNdoc.txt','w', encoding='utf-8') as f:
	for i in range(len(datalist)-35):
		t_list0=[str(float(cl[0])/float(datalist[i][0])-1) for cl in datalist[i:i+30]]
		t_list1=[str(float(cl[1])/float(datalist[i][1])-1) for cl in datalist[i:i+30]]
		t_list2=[str(float(cl[2])/float(datalist[i][2])-1) for cl in datalist[i:i+30]]
		t_list3=[str(float(cl[3])/float(datalist[i][3])-1) for cl in datalist[i:i+30]]
		t_list4=[str(float(cl[4])/float(datalist[i][4])-1) for cl in datalist[i:i+30]]
		t_list5=[str(float(cl[5])/float(datalist[i][5])-1) for cl in datalist[i:i+30]]
		t_label=str(float(datalist[i+35][3])/float(datalist[i+30][3])-1)
		for j in range(0,30):
			t_list.append(t_list0[j])
			t_list.append(t_list1[j])
			t_list.append(t_list2[j])
			t_list.append(t_list3[j])
			t_list.append(t_list4[j])
			t_list.append(t_list5[j])
			# print(t_list_seq)
			# t_list.append(t_list_seq)
		# print(len(t_list))
		# seq='<sssss>'.join(t_list)
		# t_list=[]
		# #f.write(seq+'\n')
		# seqs='\t\t'.join(['1','1',t_label,seq])
		# f.write(seqs+'\n')
		if i % 100 == 0: 
			print(i)
		label.append(t_label)
	label = tf.reshape(label, [-1,])
	t_list = tf.reshape(t_list, [-1,180])
	print(t_list.get_shape().as_list())
	print(label.get_shape().as_list())
	f.write(t_list)


with open('../data/initial_dataset_forCNNlabel.txt','w', encoding='utf-8') as g:
	g.write(label)