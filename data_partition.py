import tushare as ts

import pandas as pd

import numpy as np

# ts.set_token('0748525a1f0867e8dafe706c2158bb31514d3e9795513614b3d00192')
# pro = ts.pro_api()
# df = pro.index_daily(ts_code='399300.SZ', start_date='20100101', end_date='20181231')
# df = df.drop(["pre_close", "change", "pct_chg",'ts_code','vol','amount'], axis = 1)
# #print(df)
# file1=open(r'index_train.txt','w',encoding='utf-8')
# file1.write(df.to_string())
# file1.close()

f_train=open('../data/train.txt','w',encoding='utf-8')
f_test=open('../data/test.txt','w',encoding='utf-8')
with open('../data/initial_dataset.txt','r',encoding='UTF-8') as f:
	k=0
	for line in f:
		if k<78800:
			f_train.write(line)
		else:
			f_test.write(line)
		k+=1
f_test.close()
f_train.close()

f_val_validation=open('../data/val_validation.txt','w',encoding='utf-8')
f_val_train=open('../data/val_train.txt','w',encoding='utf-8')
with open('../data/train.txt','r',encoding='utf-8') as f:
	k=0
	for line in f:
		if k<68800:
			f_val_train.write(line)
		else:
			f_val_validation.write(line)
		k+=1
f_val_validation.close()
f_val_train.close()


f_train2=open('../data/train2.txt','w',encoding='utf-8')
f_test2=open('../data/test2.txt','w',encoding='utf-8')
with open('../data/initial_dataset2.txt','r',encoding='UTF-8') as f:
	k=0
	for line in f:
		if k<78800:
			f_train2.write(line)
		else:
			f_test2.write(line)
		k+=1
f_test2.close()
f_train2.close()

f_val_validation2=open('../data/val_validation2.txt','w',encoding='utf-8')
f_val_train2=open('../data/val_train2.txt','w',encoding='utf-8')
with open('../data/train2.txt','r',encoding='utf-8') as f:
	k=0
	for line in f:
		if k<68800:
			f_val_train2.write(line)
		else:
			f_val_validation2.write(line)
		k+=1
f_val_validation2.close()
f_val_train2.close()

f_trainlabel=open('../data/trainlabel.txt','w',encoding='utf-8')
f_testlabel=open('../data/testlabel.txt','w',encoding='utf-8')
with open('../data/labels.txt','r',encoding='UTF-8') as f:
	k=0
	for line in f:
		if 78800:
			f_trainlabel.write(line)
		else:
			f_testlabel.write(line)
		k+=1
f_testlabel.close()
f_trainlabel.close()
