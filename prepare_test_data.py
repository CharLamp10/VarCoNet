# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 09:50:04 2024

@author: 100063082
"""

import numpy as np
import scipy.io as sio
from os.path import join
import os
from random import sample, randint


path = r'C:\Users\100063082\Desktop\SSL_FC_matrix_data'

path_data_1 = r'E:\REST1_ROIsignals'
path_data_2 = r'E:\REST2_ROIsignals'

dir1 = os.listdir(path_data_1)
dir2 = os.listdir(path_data_2)

names = []
with open(os.path.join(path,'names.txt'), 'r') as f:
    for line in f:
        names.append(line.strip())
        
test_dir1 =[x for x in dir1 if x not in names]
test_dir2 =[x for x in dir2 if x not in names]


test_names = list(set(test_dir1).intersection(test_dir2))
Time1 = []
Time2 = []
for name in test_names:
    name_path1 = join(path_data_1, name)
    name_path2 = join(path_data_2, name)
    feature = []
    if os.path.exists(name_path1):
        pe = randint(0,1)
        if pe == 0:
            pe = 'LR'
        else:
            pe = 'RL'
        flag = 0
        for time_dir in os.listdir(name_path1):
            if 'ROISignals' in time_dir and '.mat' in time_dir and pe in time_dir:
                feature_dir = os.path.join(name_path1, time_dir)        
                temp = np.array(sio.loadmat(feature_dir)['ROISignals'])
                if temp.shape[0] == 1200:
                    Time1.append(temp)
                    flag = 1
        if flag == 0:
            if pe == 'RL':
                pe = 'LR'
            elif pe == 'LR':
                pe = 'RL'
            for time_dir in os.listdir(name_path1):
                if 'ROISignals' in time_dir and '.mat' in time_dir and pe in time_dir:
                    feature_dir = os.path.join(name_path1, time_dir)   
                    temp = np.array(sio.loadmat(feature_dir)['ROISignals'])
                    if temp.shape[0] == 1200:
                        Time1.append(np.array(sio.loadmat(feature_dir)['ROISignals']))
                        flag = 1

    if os.path.exists(name_path2):
        pe = randint(0,1)
        if pe == 0:
            pe = 'LR'
        else:
            pe = 'RL'
        flag = 0
        for time_dir in os.listdir(name_path2):
            if 'ROISignals' in time_dir and '.mat' in time_dir and pe in time_dir:
                feature_dir = os.path.join(name_path2, time_dir)  
                temp = np.array(sio.loadmat(feature_dir)['ROISignals'])
                if temp.shape[0] == 1200:
                    Time2.append(temp)
                    flag = 1
        if flag == 0:
            if pe == 'RL':
                pe = 'LR'
            elif pe == 'LR':
                pe = 'RL'
            for time_dir in os.listdir(name_path2):
                if 'ROISignals' in time_dir and '.mat' in time_dir and pe in time_dir:
                    feature_dir = os.path.join(name_path2, time_dir)   
                    temp = np.array(sio.loadmat(feature_dir)['ROISignals'])
                    if temp.shape[0] == 1200:
                        Time2.append(temp)
                        flag = 1
    if len(Time1) > len(Time2):
        Time1 = Time1[:-1]
    if len(Time2) > len(Time1):
        Time2 = Time2[:-1]
                        

                        
                        

windowsize =[56,84,112,140,180]
strite = 110
random_list1 = []
window_point_Time = []
t = 15
for x in Time1:
    point_Time = []
    for size in windowsize:
        time = x.shape[0]
        window_point = []
        for i in range(t):
            k = windowsize[-1] / 2 + strite * i
            if k + windowsize[-1] / 2 >= (time - 1):
                k = randint(windowsize[-1] / 2, time - windowsize[-1] / 2 - 1)
            begin = int(k - size / 2)
            end = int(k + size / 2)
            window_point.append([begin, k , end])
        point_Time.append(window_point)

    window_point_Time.append(point_Time)

for j,x in enumerate(Time1):
    all_feature = []
    for n in range(t):
        for i in range(len(windowsize)):
            begin = window_point_Time[j][i][n][0]
            end =   window_point_Time[j][i][n][-1]
            feature_i = np.zeros((windowsize[-1],384))
            feature_i[0:windowsize[i],:] = x[begin:end,:]
            temp = feature_i.astype(np.float32)
            all_feature.append(temp)

    random_list1.append(all_feature)
    
    
windowsize =[56,84,112,140,180]
strite = 110
random_list2 = []
window_point_Time = []
t = 15
for x in Time2:
    point_Time = []
    for size in windowsize:
        time = x.shape[0]
        window_point = []
        for i in range(t):
            k = windowsize[-1] / 2 + strite * i
            if k + windowsize[-1] / 2 >= (time - 1):
                k = randint(windowsize[-1] / 2, time - windowsize[-1] / 2 - 1)
            begin = int(k - size / 2)
            end = int(k + size / 2)
            window_point.append([begin, k , end])
        point_Time.append(window_point)

    window_point_Time.append(point_Time)

for j,x in enumerate(Time2):
    all_feature = []
    for n in range(t):
        for i in range(len(windowsize)):
            begin = window_point_Time[j][i][n][0]
            end =   window_point_Time[j][i][n][-1]
            feature_i = np.zeros((windowsize[-1],384))
            feature_i[0:windowsize[i],:] = x[begin:end,:]
            temp = feature_i.astype(np.float32)
            all_feature.append(temp)

    random_list2.append(all_feature)
    
np.savez('random_list_test1',*random_list1)
np.savez('random_list_test2',*random_list2)