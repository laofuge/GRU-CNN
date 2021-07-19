import mplfinance as mpf
import csv
import pandas as pd
import math
import numpy as np
from pandas.core.frame import DataFrame
import numpy as np
import matplotlib.pyplot as plt 

# datalist=[]
# with open('../data/tw_spydata_raw.csv',newline='') as g:
# 	reader=csv.reader(g)
# 	for row in reader:
# 		line=row[0:]
# 		# print(line)
# 		datalist.append(line)

# print(datalist[0])
# all_list, t_list=[], []
# with open('../data/initpic.txt','w',encoding='utf-8') as f:
# 	for i in range(len(datalist)-30):
# 		t_list0 = [int(datalist[i+s][0]) for s in range(30)]
# 		t_list1 = [float(datalist[i+s][1]) for s in range(30)]
# 		t_list2 = [float(datalist[i+s][2]) for s in range(30)]
# 		t_list3 = [float(datalist[i+s][3]) for s in range(30)]
# 		t_list4 = [float(datalist[i+s][4]) for s in range(30)]
# 		t_list5 = [int(datalist[i+s][5]) for s in range(30)]
# 		t_list6 = [int(datalist[i+s][6]) for s in range(30)]

# 		t_list = []
# 		for j in range(0,30):
# 			t_list_seq = [t_list0[j],t_list1[j],t_list2[j],t_list3[j],t_list4[j],t_list5[j],t_list6[j]]
# 			# print(t_list_seq)
# 			t_list.append(t_list_seq)
# 		# if i == 0:	
# 		# 	print(t_list)
# 		all_list.append(t_list)
# 		f.write(str(t_list)+'\n')
# 		if (i+1) % 100 == 0:
# 			print('data making finished %.2f' % (100*(i+1)/(len(datalist)-30)))


# # datapic = []
# # with open('../data/initpic.txt','r',encoding='utf-8') as f:
# # 	for line in f:
# # 		datapic.append(line.strip(' ').strip('\n'))
# # print((datapic[0]))

# for t, i in enumerate(all_list):
# 	# print(i[0])
# 	high, low, op, close, volume, amount = [], [], [], [], [], []
# 	times = pd.date_range('2016/10/16', periods = 30)
# 	for j in range(30):
# 		high.append(i[j][1])
# 		low.append(i[j][2])
# 		op.append(i[j][3])
# 		close.append(i[j][4])
# 		volume.append(i[j][5])
# 		amount.append(i[j][6])
# 	datapic_seq = DataFrame({'Date': times, 'Open': op, 'High': high, 'Low': low, 'Close': close, 'Volume': volume})
# 	datapic_seq.set_index(["Date"], inplace=True)

# 	# print(datapic_seq)
# 	if (t+1) % 100 == 0:
# 		print('trainpic list making finished %.2f' % (100*(t+1)/(len(all_list)-30)))
# 	mc = mpf.make_marketcolors(up = 'red', down = 'blue', edge = 'i', wick = 'i', volume = 'green')
# 	s = mpf.make_mpf_style(gridaxis = 'both', gridstyle = '-.', y_on_right = False, marketcolors = mc, edgecolor='white', figcolor='white', facecolor='white', gridcolor='white')
# 	mpf.plot(datapic_seq, type = 'candle', volume = True, style = s, savefig='../pic/candle-%d' % (t+1) + '.jpg')


from glob import glob
from PIL import Image
import os
import tensorflow as tf
img_path = glob("../pic/pictrain/*.jpg")
path_save = "../pictrain/"
a = range(0,len(img_path))
i = 1

for file in img_path:
    name = os.path.join(path_save, "candle-%d.jpg" % i)
    im = Image.open(file)
    im1 = im.resize((112, 112), Image.ANTIALIAS)
    # im1 = tf.image.resize_images(im, [112, 112], method = tf.image.ResizeMethod.BILINEAR)
    # sess = tf.InteractiveSession()
    # resized_fetch = im1.eval(session = sess)
    # resized_fetch = Image.fromarray(np.uint8(resized_fetch))
    # print(im.format, im.size, im.mode)
    # resized_fetch.save(name,'JPEG')
    im1.save(name,'JPEG')

    print('img_train finished %d' % i)
    i+=1

img_path = glob("../pic/pictest/*.jpg")
path_save = "../pictest/"
a = range(0,len(img_path))
i = 1
for file in img_path:
    name = os.path.join(path_save, "candle-%d.jpg" % i)
    im = Image.open(file)
    im1 = im.resize((112, 112), Image.ANTIALIAS)
    # im1 = tf.image.resize_images(im, [112, 112], method = tf.image.ResizeMethod.BILINEAR)
    # sess = tf.InteractiveSession()
    # resized_fetch = im1.eval(session = sess)
    # resized_fetch = Image.fromarray(np.uint8(resized_fetch))
    # # print(im.format, im.size, im.mode)
    # resized_fetch.save(name,'JPEG')
    im1.save(name,'JPEG') 
    print('img_test finished %d' % i)
    i+=1

#coding=utf-8
#
# 测试需要，裁剪图片，使用之前学过的技术
# 做一个裁剪的小案例

# import os
# #读取path路径下的 jpg文件
# # def getAllImages(path):
# #     #f.endswith（）  限制文件类型
# #     #f.endswith('.jpg')|f.endswith('.png')  改成这句可以读取jpg/png格式的文件
# #     #注意 返回的是绝对路径
# #     return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg')]

# import pylab as plb
# import PIL.Image as Image
# #循环读图
# img_path = glob("../pic/*.jpg")
# t = 1
# for path in img_path:
#     #读图

#     img = Image.open(path)
#     #显示
#     # plb.imshow(img)
#     #设置裁剪点（4个）
#     # corner = plb.ginput(4)
#     # #顺时针取点求解
#     # left = (corner[0][0] + corner[3][0])/2
#     # top = (corner[1][1] + corner[0][1])/2
#     # reight = (corner[1][0] + corner[2][0])/2
#     # under = (corner[3][1] + corner[2][1])/2
#     #print left,top,reight,under
#     #box = [left,top,reight,under]
#     #box中的数必须是 int 否则会报错
#     box = [180, 0, 685, 473]
#     #裁剪
#     img2 = img.crop(box)
#     #显示裁剪后的效果
#     #plb.imshow(img2)
#     #plb.show()
#     #储存为原来路径覆盖原文件
#     img2.save('../pic1/candle-%d' % t + '.jpg')
    
#     print('finished: %d' % t)
#     t+=1
# plb.show()
