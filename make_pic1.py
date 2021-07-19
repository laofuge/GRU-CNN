import csv
import pandas as pd
import math
import numpy as np

datalist=[]
with open('../data/tw_spydata_raw.csv',newline='') as g:
	reader=csv.reader(g)
	for row in reader:
		line=row[0:]
		# print(line)
		datalist.append(line)

print(datalist[0])
t_list=[]
with open('../data/initpic.txt','w',encoding='utf-8') as f:
	for i in range(len(datalist)-30):
		t_list0 = [int(datalist[i+s][0]) for s in range(30)]
		t_list1 = [float(datalist[i+s][1]) for s in range(30)]
		t_list2 = [float(datalist[i+s][2]) for s in range(30)]
		t_list3 = [float(datalist[i+s][3]) for s in range(30)]
		t_list4 = [float(datalist[i+s][4]) for s in range(30)]
		t_list5 = [int(datalist[i+s][5]) for s in range(30)]
		t_list6 = [int(datalist[i+s][6]) for s in range(30)]

		t_list = []
		for j in range(0,30):
			t_list_seq = [t_list0[j],t_list1[j],t_list2[j],t_list3[j],t_list4[j],t_list5[j],t_list6[j]]
			# print(t_list_seq)
			t_list.append(t_list_seq)
		# if i == 0:	
		# 	print(t_list)
		f.write(str(t_list)+'\n')
		if (i+1) % 100 == 0:
			print('finished %.2f' % (100*(i+1)/(len(datalist)-30)))