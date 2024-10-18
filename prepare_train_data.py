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


train_subjects = 500

path_data_1 = r'E:\REST1_ROIsignals'
path_data_2 = r'E:\REST2_ROIsignals'
path_save = r'C:\Users\100063082\Desktop\SSL_FC_matrix_data'

dir1 = os.listdir(path_data_1)
dir2 = os.listdir(path_data_2)

one_not_two = list(set(dir1).difference(dir2))
two_not_one = list(set(dir2).difference(dir1))

pos_remove = []
for ind in one_not_two:
    pos_remove.append(dir1.index(ind))
pos_remove.sort(reverse=True)
for pos in pos_remove:
    dir1.pop(pos)
    
pos_remove = []
for ind in two_not_one:
    pos_remove.append(dir2.index(ind))
pos_remove.sort(reverse=True)
for pos in pos_remove:
    dir2.pop(pos)

one_not_two.extend(two_not_one)

remaining_subjects = train_subjects - len(one_not_two)

rand_inds = sample(range(1, len(dir1)), remaining_subjects)

rand_subjects = [dir1[i] for i in rand_inds]

rand_subjects.extend(one_not_two)

Time = []
names = []
for name in rand_subjects:
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
                    Time.append(temp)
                    names.append(name)
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
                        Time.append(np.array(sio.loadmat(feature_dir)['ROISignals']))
                        names.append(name)
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
                    Time.append(temp)
                    names.append(name)
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
                        Time.append(temp)
                        names.append(name)
                    

wind_size_max = 300
windowsize =[56,84,112,140,180,220,260,wind_size_max]
""" S-A"""
strite = int(wind_size_max/2)
random_list_SA = []
t = 15
for x in Time:
    all_feature = []
    for size in windowsize:
        feature = []
        for i in range(t):
            time = x.shape[0]
            k = strite * i
            if k + size >= (time - 1):
                k = randint(1, time - size - 2)
            feature_i = np.zeros((wind_size_max,384))   
            feature_i[0:size,:] = x[k:k + size,:]
            temp = feature_i.astype(np.float32)
            feature.append(temp)
        all_feature.append(feature)
    random_list_SA.append(all_feature)
#windowsize = 140
#strite = int(windowsize/2)
#random_list_SA = []
#t = 20
#for x in Time:
#    all_feature = []
#    for i in range(t):
#        time = x.shape[0]
#        k = strite * i
#        if k + windowsize >= (time - 1):
#            k = randint(1, time - windowsize - 2)
#        feature_i = np.zeros((wind_size_max,384))   
#        feature_i[0:windowsize,:] = x[k:k + windowsize,:]
#        temp = feature_i.astype(np.float32)
#        all_feature.append(temp)
#    random_list_SA.append(all_feature)

"""M-A"""
#strite = 200
#windowsize =[56,84,112,140,wind_size_max]
random_list_MA = []
window_point_Time = []
t = 15
for x in Time:
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

for j,x in enumerate(Time):
    all_feature = []
    for n in range(t):
        for i in range(len(windowsize)):
            begin = window_point_Time[j][i][n][0]
            end =   window_point_Time[j][i][n][-1]
            feature_i = np.zeros((wind_size_max,384))
            feature_i[0:windowsize[i],:] = x[begin:end,:]
            temp = feature_i.astype(np.float32)
            all_feature.append(temp)

    random_list_MA.append(all_feature)
        
    
np.savez(os.path.join(path_save,'random_list_SA_300_dense'),*random_list_SA)
np.savez(os.path.join(path_save,'random_list_MA_300_dense'),*random_list_MA)

with open(os.path.join(path_save,'names_300_dense.txt'), 'w') as f:
    for item in names:
        f.write("%s\n" % item)
