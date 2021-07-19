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

#需要在列间插入“<sssss>”
# list_399300=[]
# with open('../data/index_train.txt', 'r', encoding='UTF-8') as f:
# 	k=0
# 	for line in f:
# 		if k<100:
# 			tokens = line.strip().split('     ')[1]
# 			#print(tokens)
# 			list_399300.append(tokens.strip())
# 		elif k<1000:
# 			tokens = line.strip().split('    ')[1]
# 			#print(tokens)
# 			list_399300.append(tokens.strip())
# 		else:
# 			tokens = line.strip().split('   ')[1]
# 			#print(tokens)
# 			list_399300.append(tokens.strip())
# 		k+=1
# #print(list_399300)
# list_399300.reverse()

# del list_399300[0]

# del list_399300[-1]
# # print(list_399300)  #未删除日期，一个元素5个数，数中间两个空格，组成的列表

# # print(float(list_399300[60].split('  ')[1]))


# f=open('../data/new_index_train.txt','w',encoding='utf-8')

# for i in range(len(list_399300)-60):
# 	t_list=list_399300[i:i+60]
# 	t_label=str(float(list_399300[i+60].split('  ')[1])/float(list_399300[i].split('  ')[1])-1)
# 	# print(t_label)
# 	seq='<sssss>'.join(t_list)
# 	#f.write(seq+'\n')
# 	seqs='\t\t'.join(['1','1',t_label,seq])
# 	f.write(seqs+'\n')
# f.close()

# # embeddings=[]
# # for sentence in list_399300:
# # 	words=sentence.split('  ')[1:]
# # 	embeddings.append([float(word) for word in words]) #转为float没有引号 words是一个列表所以加[]
# # # print(embeddings)

# # import pickle as pk
# # new_embeddings=np.array(embeddings) #改为矩阵形式
# # pk.dump(new_embeddings, open('new_embeddings.save', 'wb')) #保存


# # f=open('date.txt','w',encoding='utf-8')
# # for all_date in list_399300:
# # 	date=all_date.split('  ')[0]
# # 	f.write(date+'\n')
# # f.close()

# #建立文档：第四列只含日期，中间<sssss>隔开
# # f=open('../data/new_index_train.txt','w',encoding='utf-8')
# # for i in range(len(list_399300)-60):
# # 	t_list=[date.split('  ')[0] for date in list_399300[i:i+60]]
# # 	t_label=str(float(list_399300[i+60].split('  ')[1])/float(list_399300[i].split('  ')[1])-1)
# # 	seq='<sssss>'.join(t_list)
# # 	#f.write(seq+'\n')
# # 	seqs='\t\t'.join(['1','1',t_label,seq])
# # 	f.write(seqs+'\n')
# # f.close()


