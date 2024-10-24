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

features1 = []
for data in Time1:
    corr = np.corrcoef(data[:600,].T)
    upper_triangular = np.triu(corr,k=1)
    features1.append(upper_triangular[np.triu_indices_from(corr)])
features1 = np.array(features1)

features2 = []
for data in Time2:
    corr = np.corrcoef(data[:600,:].T)
    upper_triangular = np.triu(corr,k=1)
    features2.append(upper_triangular[np.triu_indices_from(corr)])
features2 = np.array(features2)

corr_coeffs = np.corrcoef(features1, features2)[0:features1.shape[0],features1.shape[0]:]

lower_indices = np.tril_indices(corr_coeffs.shape[0], k=-1)
upper_indices = np.triu_indices(corr_coeffs.shape[0], k=1)
corr_coeffs1 = corr_coeffs.copy()
corr_coeffs2 = corr_coeffs.copy()
corr_coeffs1[lower_indices] = -2
corr_coeffs2[upper_indices] = -2
counter1 = 0
counter2 = 0
for j in range(corr_coeffs1.shape[0]):
    if np.argmax(corr_coeffs1[j, :]) == j:
        counter1 += 1
for j in range(corr_coeffs2.shape[1]):
    if np.argmax(corr_coeffs2[:, j]) == j:
        counter2 += 1

# Append accuracy for this feature
total_samples = features1.shape[0] + features2.shape[0]
accuracy = (counter1 + counter2) / total_samples