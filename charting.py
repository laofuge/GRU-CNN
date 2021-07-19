import numpy as np
import matplotlib.pyplot as plt 
import tensorflow as tf
import math


closeprice=[]
closeprice30=[]
real_train=[]
real_test=[]
real_train30=[]
real_test30=[]


# 做pred_test，pred_test为测试集预测结果
pred_test_log, pred_test = [], []
init_pred_test=np.load('../data/testpredsFNN.npy') # 改文件名
for i in init_pred_test:
	pred_test_log.append(float(i))

print(pred_test_log)
print(len(pred_test_log))

if len(pred_test_log) == 10000:
	len_testorval = 10000 # 验证集或测试集的长度，修改此变量：10000 or 19440
	len_division = 68800 # 数据对齐，修改此变量：68800 or 78800
else:
	len_testorval = 19440 # 验证集或测试集的长度，修改此变量：10000 or 19440
	len_division = 78800 # 数据对齐，修改此变量：68800 or 78800	

with open('../data/price_close.txt','r', encoding='utf-8') as f: # 要对比的数
	for line in f:
		closeprice.append(float(line.split()[0]))
for i in range(0, len(closeprice)):
	if i < 78800:
		real_train.append(closeprice[i])
	else:
		real_test.append(closeprice[i])
# print(closeprice)
print(len(real_test))

with open('../data/price_close30.txt','r', encoding='utf-8') as f: # 因子
	for line in f:
		closeprice30.append(float(line.split()[0]))
for i in range(0, len(closeprice30)):
	if i < 78800:
		real_train30.append(closeprice30[i])
	else:
		real_test30.append(closeprice30[i])
print(len(real_test30))
print(len(real_test))

for j in range(0, min(len(pred_test_log), 19475)):
	# mid = math.exp(pred_test_log[j])
	mid = math.exp(pred_test_log[j])

	pred_test.append(mid*real_test30[j])

	# pred_test.append(pred_test_log[j])
	# # pred_test.append(mid*46.9711+208.4789)
	# real_test[j] = math.log(real_test[j]/real_test30[j])

	# pred_test.append(mid*float(real_test30[j]))
print(len(pred_test))

# with open('../data/FNNatt.csv','w',encoding='utf-8') as f:
# 	for da in pred_test:
# 		f.write(str(da)+'\n')
# f.close()


if len(pred_test_log) == 10000:
	E, A, AP = [], [], []
	SE, AE, AP = 0, 0, 0
	for i in range(len_testorval):
		SE += ((abs(pred_test[i]-real_test[i]))**2)
		AE += abs(pred_test[i]-real_test[i])
		AP += (abs(pred_test[i]-real_test[i])**0.5) / pred_test[i]
	RMSE = (SE / len_testorval) ** 0.5
	RMAE = (AE / len_testorval) ** 0.5
	MAPE = (AP / len_testorval) * (100/len_division)
	print(RMSE, '\n', RMAE, '\n', MAPE)
else:
	E, A, AP = [], [], []
	SE, AE, AP = 0, 0, 0
	for i in range(len_testorval):
		SE += ((abs(pred_test[i]-real_test[i]))**2)
		AE += abs(pred_test[i]-real_test[i])
		AP += abs((pred_test[i]-real_test[i]) / pred_test[i])
	RMSE = (SE / len_testorval) ** 0.5
	RMAE = (AE / len_testorval) ** 0.5
	MAPE = (AP / len_testorval) * 100 # * (100/len_division)
	print(RMSE, '\n', RMAE, '\n', MAPE)

if len(pred_test_log) == 10000:
	def plot_results(predicted_data, true_data):
	    fig = plt.figure(facecolor='white')
	    # ax = fig.add_subplot(111)
	    plt.plot(true_data, label='True Data')
	    plt.plot(predicted_data, label='Prediction')
	    plt.legend()
	    plt.show()
	plot_results(pred_test, real_test)
else:
	def plot_results(predicted_data, true_data):
	    fig = plt.figure(facecolor='white')
	    # ax = fig.add_subplot(111)
	    plt.plot(true_data, label='True Data')
	    plt.plot(predicted_data, label='Prediction')
	    plt.legend()
	    plt.show()
	plot_results(pred_test, real_test)	


pred_train_log, pred_train = [], []
init_pred_train=np.load('../data/trainpredsFNN.npy') # 改文件名
for i in init_pred_train:
	pred_train_log.append(float(i))

# print(pred_train_log)
# for i in init_pred_train:
# 	pred_train_log.append(float(i))

for j in range(len(pred_train_log)):
	# mid = math.exp(pred_train_log[j])
	mid = math.exp(pred_train_log[j])
	pred_train.append(mid*real_train30[j])

	# pred_train.append(mid*mid*46.9711+208.4789)
print(len(pred_train))


def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    # ax = fig.add_subplot(111)
    plt.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()
plot_results(pred_train, real_train)

