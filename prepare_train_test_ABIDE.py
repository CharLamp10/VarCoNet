import numpy as np
import os
from random import sample,randint,choices
from scipy.signal import resample


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


path_data = r'G:\ABIDE_ADHD_AAL\ABIDEI\dparsf_cc400\filt_noglobal\rois_cc400'
save_path = r'C:\Users\100063082\Desktop\SSL_FC_matrix_data'
data_dir = os.listdir(path_data)
if 'cc400' in path_data:
    rois = 392
elif 'cc200' in path_data:
    rois = 200

train = sample(data_dir, 775) #760 for cpac_cc200, 748 for cpac_cc400, 784 for dparsf_cc200, 775 for dparsf_cc400
test = [item for item in data_dir if item not in train]
train_names = []
test_names = []

Time_train = []
for data in train:
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
            feature = resample_signal(feature, data)
            Time_train.append(feature)
            train_names.append(data)

Time_test = []
for data in test:
    feature_dir = os.path.join(path_data, data)
    feature = []
    i = 0
    for line in open(feature_dir, "r"):
        if i == 0:
            i += 1
            continue
        temp = line[:-1].split('\t')
        feature.append([float(x) for x in temp])
    feature_padded = np.zeros((393,rois))
    feature = np.array(feature)
    if feature.shape[0] <= 295 and feature.shape[0] > 100:
        if not np.any(np.all(feature == 0, axis=0)):
            feature = resample_signal(feature, data)
            feature_padded[0:len(feature),:] = feature
            Time_test.append(feature_padded)
            test_names.append(data)
    
lens = []
for t in Time_train:
    lens.append(t.shape[0])
lens = np.array(lens)
indices_to_delete = np.where((lens < 193) | (lens > 393))[0]
Time_train_new = [item for idx, item in enumerate(Time_train) if idx not in indices_to_delete]
train_new = [item for idx, item in enumerate(train_names) if idx not in indices_to_delete]
    
lens = []
for t in Time_test:
    lens.append(t.shape[0])
lens = np.array(lens)
indices_to_delete = np.where((lens < 193) | (lens > 393))[0]
Time_test_new = [item for idx, item in enumerate(Time_test) if idx not in indices_to_delete]
test_new = [item for idx, item in enumerate(test_names) if idx not in indices_to_delete]
    

wind_size_max = 393
""" S-A"""
random_list_SA = []
for x in Time_train_new:
    windowsize = int(x.shape[0]/3)
    strite = int(windowsize/2)
    t = 20
    all_feature = []
    flag = True
    for i in range(t):
        if flag:
            time = x.shape[0]
            k = strite * i
            if k + windowsize >= (time - 1):
                k = randint(1, time - windowsize - 2)
                flag = False
            feature_i = np.zeros((wind_size_max,rois))   
            feature_i[0:windowsize,:] = x[k:k + windowsize,:]
            temp = feature_i.astype(np.float32)
            all_feature.append(temp)

    random_list_SA.append(all_feature)

"""M-A"""
random_list_MA = []
window_point_Time = []
t = 20
for x in Time_train_new:
    windowsize =[int(x.shape[0]/5),int(x.shape[0]/4),int(x.shape[0]/3),int(x.shape[0]/2),x.shape[0]]
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

for j,x in enumerate(Time_train_new):
    windowsize =[int(x.shape[0]/5),int(x.shape[0]/4),int(x.shape[0]/3),int(x.shape[0]/2),x.shape[0]]
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
    
np.savez(os.path.join(save_path,'ABIDE_train_list_SA'),*random_list_SA)
np.savez(os.path.join(save_path,'ABIDE_train_list_MA'),*random_list_MA)
np.savez(os.path.join(save_path,'ABIDE_test_list'),*Time_test_new)
with open(os.path.join(save_path,'ABIDE_names_train.txt'), 'w') as f:
    for item in train_new:
        f.write("%s\n" % item)
with open(os.path.join(save_path,'ABIDE_names_test.txt'), 'w') as f:
    for item in test_new:
        f.write("%s\n" % item)
