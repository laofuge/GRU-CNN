import pandas as pd
import numpy as np
import csv
import math


datalist=[]
with open('../data/tw_spydata_raw.csv',newline='') as g:
	reader=csv.reader(g)
	for row in reader:
		line=row[1:]
		# print(line)
		datalist.append(line)
# print(datalist[1])
# datalist_log, datalist_logpar = [], []
# for i in range(len(datalist)):
# 	for j in range(len(datalist[i])):
# 		datalist_logpar.append(math.log(float(datalist[i][j])))
# 	datalist_log.append(datalist_logpar)	
# print(datalist_log[1])
max0 = float(max(datalist[i][0] for i in range(len(datalist))))
max1 = float(max(datalist[i][1] for i in range(len(datalist))))
max2 = float(max(datalist[i][2] for i in range(len(datalist))))
max3 = float(max(datalist[i][3] for i in range(len(datalist))))
# max4 = float(max(datalist[i][4] for i in range(len(datalist))))
max4 = 16035388
max5 = 11667
min4 = 1579
min5 = 8
# max5 = float(max(datalist[i][5] for i in range(len(datalist))))

min0 = float(min(datalist[i][0] for i in range(len(datalist))))
min1 = float(min(datalist[i][1] for i in range(len(datalist))))
min2 = float(min(datalist[i][2] for i in range(len(datalist))))
min3 = float(min(datalist[i][3] for i in range(len(datalist))))
# min4 = float(min(datalist[i][4] for i in range(len(datalist))))
# min5 = float(min(datalist[i][5] for i in range(len(datalist))))

# print(max0, max1, max2, max3, max4, max5, min0, min1, min2, min3, min4, min5)




t_list=[]
ff=open('../data/price_close.txt','w',encoding='utf-8')
fff =open('../data/price_close30.txt','w',encoding='utf-8') 
with open('../data/initial_dataset.txt','w',encoding='utf-8') as f:
	for i in range(len(datalist)-35):


		# 2021-04-27

		t_list0 = [str(math.log(float(datalist[i+s+1][0])/float(datalist[i+s][0]))) for s in range(30)]
		t_list1 = [str(math.log(float(datalist[i+s+1][1])/float(datalist[i+s][1]))) for s in range(30)]
		t_list2 = [str(math.log(float(datalist[i+s+1][2])/float(datalist[i+s][2]))) for s in range(30)]
		t_list3 = [str(math.log(float(datalist[i+s+1][3])/float(datalist[i+s][3]))) for s in range(30)]
		t_list4 = [str(math.log(float(datalist[i+s+1][4])/float(datalist[i+s][4]))) for s in range(30)]
		t_list5 = [str(math.log(float(datalist[i+s+1][5])/float(datalist[i+s][5]))) for s in range(30)]
		t_label=str(math.log(float(datalist[i+34][3])/float(datalist[i+29][3])))
		# max0 = float(max(datalist[i+s][0] for s in range(30)))
		# max1 = float(max(datalist[i+s][1] for s in range(30)))
		# max2 = float(max(datalist[i+s][2] for s in range(30)))
		# max3 = float(max(datalist[i+s][3] for s in range(30)))
		# max4 = float(max(datalist[i+s][4] for s in range(30)))
		# max5 = float(max(datalist[i+s][5] for s in range(30)))

		# min0 = float(min(datalist[i+s][0] for s in range(30)))
		# min1 = float(min(datalist[i+s][1] for s in range(30)))
		# min2 = float(min(datalist[i+s][2] for s in range(30)))
		# min3 = float(min(datalist[i+s][3] for s in range(30)))
		# min4 = float(min(datalist[i+s][4] for s in range(30)))
		# min5 = float(min(datalist[i+s][5] for s in range(30)))

		# t_list0 = [str((float(datalist[i+s][0])-min0)/(max0-min0)) for s in range(30)]
		# t_list1 = [str((float(datalist[i+s][1])-min1)/(max1-min1)) for s in range(30)]
		# t_list2 = [str((float(datalist[i+s][2])-min2)/(max2-min2)) for s in range(30)]
		# t_list3 = [str((float(datalist[i+s][3])-min3)/(max3-min3)) for s in range(30)]
		# t_list4 = [str((float(datalist[i+s][4])-min4)/(max4-min4)) for s in range(30)]
		# t_list5 = [str((float(datalist[i+s][5])-min5)/(max5-min5)) for s in range(30)]
		# t_label=str((float(datalist[i+34][3])-min3)/(max3-min3))


		# t_list0 = [str((float(datalist[i+s+5][0])/float(datalist[i+s][0]))) for s in range(30)]
		# t_list1 = [str((float(datalist[i+s+5][1])/float(datalist[i+s][1]))) for s in range(30)]
		# t_list2 = [str((float(datalist[i+s+5][2])/float(datalist[i+s][2]))) for s in range(30)]
		# t_list3 = [str((float(datalist[i+s+5][3])/float(datalist[i+s][3]))) for s in range(30)]
		# t_list4 = [str((float(datalist[i+s+5][4])/float(datalist[i+s][4]))) for s in range(30)]
		# t_list5 = [str((float(datalist[i+s+5][5])/float(datalist[i+s][5]))) for s in range(30)]
		# ave = 0
		# for j in range(30):
		# 	ave+=float(datalist[i+j+1][3])
		# ave=ave/30



		ff.write(datalist[i+34][3]+'\n')
		fff.write(datalist[i+29][3]+'\n')
		# t_label=str((float(datalist[i+35][3])/ave))
		# t_label=str((float(datalist[i+35][3])/float(datalist[i][3])))
		# t_label=str(float(datalist[i+35][3]))

		for j in range(0,30):
			t_list_seq='  '.join([t_list0[j],t_list1[j],t_list2[j],t_list3[j],t_list4[j],t_list5[j]])
			# print(t_list_seq)
			t_list.append(t_list_seq)
		# print(t_list)
		seq='<sssss>'.join(t_list)
		t_list=[]
		#f.write(seq+'\n')
		seqs='\t\t'.join(['1','1',t_label,seq])
		f.write(seqs+'\n')
		# print('processing: %.2f percent' % (i/len(datalist-35)))
fff.close()
ff.close()
# 做一个close价格文档
# f=open('../data/price_close.txt','w',encoding='utf-8')
# for close in datalist:
# 	price=close[3]
# 	# print(price)
# 	f.write(price+'\n')
# f.close()

# 300014  002304  002371
with open('../data/initial_dataset2.txt','w',encoding='utf-8') as g:
	for i in range(len(datalist)-35):


		t_list0 = [str((float(datalist[i+s][0])-min0)/(max0-min0)) for s in range(30)]
		t_list1 = [str((float(datalist[i+s][1])-min1)/(max1-min1)) for s in range(30)]
		t_list2 = [str((float(datalist[i+s][2])-min2)/(max2-min2)) for s in range(30)]
		t_list3 = [str((float(datalist[i+s][3])-min3)/(max3-min3)) for s in range(30)]
		t_list4 = [str((float(datalist[i+s][4])-min4)/(max4-min4)) for s in range(30)]
		t_list5 = [str((float(datalist[i+s][5])-min5)/(max5-min5)) for s in range(30)]
		t_label=str((float(datalist[i+34][3])-min3)/(max3-min3))


		for j in range(0,30):
			t_list_seq='  '.join([t_list0[j],t_list1[j],t_list2[j],t_list3[j],t_list4[j],t_list5[j]])
			# print(t_list_seq)
			t_list.append(t_list_seq)
		# print(t_list)
		seq='<sssss>'.join(t_list)
		t_list=[]
		#f.write(seq+'\n')
		seqs='\t\t'.join(['1','1',t_label,seq])
		g.write(seqs+'\n')


with open('../data/labels.txt','w',encoding='utf-8') as h:
	for i in range(len(datalist)-35):
		t_label=str(math.log(float(datalist[i+34][3])/float(datalist[i+29][3])))
		seqs=t_label
		h.write(seqs+'\n')