#!/usr/bin/env python3
# _*_ coding: utf-8 _*_
# @Time : 2023/8/20 15:27:19
# @Author : Hetao
# @File : [Code] --> ts.py
# @Software: PyCharm
# @Function: TODO
import os
import numpy as np
# n1 = 4
# str = 'num:{:0>6d}'.format(n1)
# str = "%06i" % n1
# print(str)

import numpy as np

#输入可以是list,array,ndarray

# #list
# a = [1,4,0,3,2]
# b = np.argsort(a)
# c = np.sort(a)                               #其余情况类似,不再列举
# print(b)
# print(c)
#
# #array
# a = np.array([1,4,0,3,2])
# b = np.argsort(a)
# print(b)

#ndarray

# y_cnt_mean = int(sum(cnt[:,:,1])/len(cnt))
a = np.array([[[83, 313], [1175, 337]],[[81, 311], [1173, 335]]])
a = a.mean(axis=0)
print(a)
# y_mean = int(sum(list(zip(*a))[1])/4)
# print(y_mean)


# a = np.array([[ 120,238],[ 123,72],[1168,89],[1165,255]])
# b = np.sort(a, axis = 0)                 #axis默认为-1,即最后一个维度
# print(b)

# #mat
# a = np.mat([1,4,0,3,2])
# b = np.argsort(a)
# print(b)
