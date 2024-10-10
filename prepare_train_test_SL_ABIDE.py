import numpy as np
import os
import pandas as pd
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
path_phenotypic = r'G:\ABIDE_ADHD_AAL\ABIDEI\Phenotypic_V1_0b.csv'

if 'cc400' in path_data:
    rois = 392
elif 'cc200' in path_data:
    rois = 200

with open(os.path.join(save_path,'ABIDE_names_train.txt'),'r') as f:
    train = [line.strip() for line in f]
with open(os.path.join(save_path,'ABIDE_names_test.txt'),'r') as f:
    test = [line.strip() for line in f]


phenotypic = pd.read_csv(path_phenotypic)
sub_id = phenotypic['SUB_ID'].values
group = phenotypic['DX_GROUP'].values
group[group == 2] = 0

class_train = []
class_test = []
feature = []
Time_train = []
for data in train:
    pos00 = data.find('00')
    ID = data[pos00+2:-14]
    pos_id = np.where(sub_id == int(ID))[0]
    if len(pos_id) != 1:
        raise TypeError("ID false identification")
    class_train.append(group[pos_id[0]])
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
    feature = resample_signal(feature, data)
    feature_padded[0:len(feature),:] = feature
    Time_train.append(feature_padded)

#Time_test = []
for data in test:
    pos00 = data.find('00')
    ID = data[pos00+2:-14]
    pos_id = np.where(sub_id == int(ID))[0]
    if len(pos_id) != 1:
        raise TypeError("ID false identification")
    class_test.append(group[pos_id[0]])
    #feature_dir = os.path.join(path_data, data)
    #feature = []
    #i = 0
    #for line in open(feature_dir, "r"):
    #    if i == 0:
    #        i += 1
    #        continue
    #    temp = line[:-1].split('\t')
    #    feature.append([float(x) for x in temp])
    #feature_padded = np.zeros((295,200))
    #feature_padded[0:len(feature),:] = np.array(feature)
    #Time_test.append(np.array(feature_padded))

class_train = np.array(class_train)   
class_test = np.array(class_test) 
np.save(os.path.join(save_path,'ABIDE_class_train.npy'),class_train)
np.save(os.path.join(save_path,'ABIDE_class_test.npy'),class_test)
np.savez(os.path.join(save_path,'ABIDE_train_list'),*Time_train)