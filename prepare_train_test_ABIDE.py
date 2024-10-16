import numpy as np
import os
from random import sample,randint,choices
from scipy.signal import resample
import pandas as pd


def resample_signal(signal,name):
    if 'Caltech' in name or 'CMU' in name or 'SDSU' in name or 'Stanford' in name or 'UM' in name:
        TR = 2;
    elif 'KKI' in name:
        TR = 2.5;
    elif 'Leuven' in name:
        TR = 1.66665;
    elif 'MaxMun' in name:
        TR = 3;
    elif 'NYU' in name:
        TR = 2;
    elif 'OHSU' in name:
        TR = 2.5;
    elif 'Olin' in name:
        TR = 1.5;
    elif'Pitt' in name:
        TR = 1.5;
    elif 'SBL' in name:
        TR = 2.185;
    elif 'Trinity' in name or'USM' in name:
        TR = 2;
    elif 'UCLA' in name:
        TR = 3;
    elif 'Yale' in name:
        TR = 2;
    else:
        raise Exception('Did not find correspondance')
    if TR > 1.5:
        ratio = TR/1.5
        new_len = int(np.round(signal.shape[0]*ratio))
        signal = resample(signal,new_len)
    return signal


wind_size_max = 393
path_data = r'G:\ABIDE_ADHD_AAL\ABIDEI\dparsf_cc200\filt_noglobal\rois_cc200'
save_path = r'C:\Users\100063082\Desktop\SSL_FC_matrix_data\dparsf_cc200'
path_phenotypic = r'G:\ABIDE_ADHD_AAL\ABIDEI\Phenotypic_V1_0b.csv'

data_dir = os.listdir(path_data)
if 'cc400' in path_data:
    rois = 392
elif 'cc200' in path_data:
    rois = 200


phenotypic = pd.read_csv(path_phenotypic)
sub_id = phenotypic['SUB_ID'].values
group = phenotypic['DX_GROUP'].values
group[group == 2] = 0

names = []
Times = []
classes = []
for data in data_dir:
    pos00 = data.find('00')
    ID = data[pos00+2:-14]
    pos_id = np.where(sub_id == int(ID))[0]
    if len(pos_id) != 1:
        raise TypeError("ID false identification")
    feature_dir = os.path.join(path_data, data)
    feature = []
    i = 0
    for line in open(feature_dir, "r"):
        if i == 0:
            i += 1
            continue
        temp = line[:-1].split('\t')
        feature.append([float(x) for x in temp])
    feature = np.array(feature)
    if feature.shape[0] <= 295 and feature.shape[0] > 100:
        if not np.any(np.all(feature == 0, axis=0)):
            classes.append(group[pos_id[0]])
            feature = resample_signal(feature, data)          
            Times.append(feature)
            names.append(data)

ASD_inds = np.where(np.array(classes) == 1)[0]
NC_inds = np.where(np.array(classes) == 0)[0]              
test_inds = list(ASD_inds[sample(range(len(ASD_inds)), 50)]) + list(NC_inds[sample(range(len(NC_inds)), 50)])
test = [data_dir[i] for i in test_inds]
train_inds = [item for item in range(len(Times)) if item not in test_inds]
train = [item for item in data_dir if item not in test]
Time_train = [Times[i] for i in train_inds]
Time_test = [Times[i] for i in test_inds]
class_train = [classes[i] for i in train_inds]
class_test = [classes[i] for i in test_inds]

Time_test_new = []
for data in Time_test:
    feature_padded = np.zeros((wind_size_max,rois))
    feature_padded[0:data.shape[0],:] = data
    Time_test_new.append(feature_padded)
    
Time_train_new = []
for data in Time_train:
    feature_padded = np.zeros((wind_size_max,rois))
    feature_padded[0:data.shape[0],:] = data
    Time_train_new.append(feature_padded)
    

""" S-A"""
random_list_SA = []
for x in Time_train:
    windowsize = [int(x.shape[0]/4),int(x.shape[0]/3)]
    t = 20
    all_feature = []
    for size in windowsize:
        strite = int(size/2)
        feature = []
        flag = True
        for i in range(t):
            if flag:
                time = x.shape[0]
                k = strite * i
                if k + size >= (time - 1):
                    k = randint(1, time - size - 2)
                    flag = False
                feature_i = np.zeros((wind_size_max,rois))   
                feature_i[0:size,:] = x[k:k + size,:]
                temp = feature_i.astype(np.float32)
                feature.append(temp)
        for j in range(10 - len(feature)):
            feature.append(np.zeros_like(temp))
        all_feature.append(feature)
    random_list_SA.append(all_feature)
    
#random_list_SA = []
#for x in Time_train:
#    windowsize = int(x.shape[0]/3)
#    strite = int(windowsize/2)
#    t = 20
#    all_feature = []
#    flag = True
#    for i in range(t):
#        if flag:
#            time = x.shape[0]
#            k = strite * i
#            if k + windowsize >= (time - 1):
#                k = randint(1, time - windowsize - 2)
#                flag = False
#            feature_i = np.zeros((wind_size_max,rois))   
#            feature_i[0:windowsize,:] = x[k:k + windowsize,:]
#            temp = feature_i.astype(np.float32)
#            all_feature.append(temp)
#    random_list_SA.append(all_feature)

"""M-A"""
random_list_MA = []
window_point_Time = []
t = 20
for x in Time_train:
    windowsize = [int(x.shape[0]/5),int(x.shape[0]/4),int(x.shape[0]/3),int(x.shape[0]/2),x.shape[0]]
    point_Time = []
    for size in windowsize:
        strite = int(size/2)+1
        if size != windowsize[-1]:
            time = x.shape[0]
            window_point = []
            flag = True
            for i in range(t):
                if flag:
                    k = strite * (i+1)
                    if k + size / 2 >= (time - 1):
                        flag = False
                    else:
                        begin = int(k - size / 2)
                        end = int(k + size / 2)
                        window_point.append([begin, k , end])
        else:
             window_point = [[0,int(x.shape[0]/2),x.shape[0]]]
        if len(point_Time) == 0:
            point_Time.append(window_point)
        else:
            if len(point_Time[0]) > len(window_point):
                if len(point_Time[0]) % len(window_point) == 0:
                    window_point = window_point*int(len(point_Time[0]) / len(window_point))
                else:
                    window_point = window_point + choices(window_point, k=len(point_Time[0]) - len(window_point))  
            point_Time.append(window_point)
    window_point_Time.append(point_Time)

for j,x in enumerate(Time_train):
    windowsize = [int(x.shape[0]/5),int(x.shape[0]/4),int(x.shape[0]/3),int(x.shape[0]/2),x.shape[0]]
    all_feature = []
    for n in range(len(window_point_Time[j][0])):
        for i in range(len(windowsize)):
            begin = window_point_Time[j][i][n][0]
            end =   window_point_Time[j][i][n][-1]
            feature_i = np.zeros((wind_size_max,rois))
            feature_i[0:windowsize[i],:] = x[begin:end,:]
            temp = feature_i.astype(np.float32)
            all_feature.append(temp)

    random_list_MA.append(all_feature)



class_train = np.array(class_train)   
class_test = np.array(class_test) 
np.save(os.path.join(save_path,'ABIDE_class_train.npy'),class_train)
np.save(os.path.join(save_path,'ABIDE_class_test.npy'),class_test) 
np.savez(os.path.join(save_path,'ABIDE_train_list_SA'),*random_list_SA)
np.savez(os.path.join(save_path,'ABIDE_train_list_MA'),*random_list_MA)
np.savez(os.path.join(save_path,'ABIDE_test_list'),*Time_test_new)
np.savez(os.path.join(save_path,'ABIDE_train_list'),*Time_train_new)   
with open(os.path.join(save_path,'ABIDE_names_train.txt'), 'w') as f:
    for item in train:
        f.write("%s\n" % item)
with open(os.path.join(save_path,'ABIDE_names_test.txt'), 'w') as f:
    for item in test:
        f.write("%s\n" % item)
