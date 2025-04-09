#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 11:23:24 2024

@author: student1
"""

import numpy as np
import os

path = r'C:\Users\100063082\Desktop\SSL_FC_matrix_data'
data = np.load(os.path.join(path,'random_list_test1.npz'))
test_data1 = []
for key in data:
    test_data1.append(data[key])

data = np.load(os.path.join(path,'random_list_test2.npz'))
test_data2 = []
for key in data:
    test_data2.append(data[key])

features1 = np.zeros((594,90,73536))
features2 = np.zeros((594,90,73536))
for j,data in enumerate(test_data1):
    for i in range(data.shape[0]):
        features = np.corrcoef(data[i,:,:].T)
        upper_tri_indices = np.triu_indices_from(features, k=1)  # k=1 excludes the diagonal
        features1[j,i,:] = features[upper_tri_indices]

for j,data in enumerate(test_data2):
    for i in range(data.shape[0]):
        features = np.corrcoef(data[i,:,:].T)
        upper_tri_indices = np.triu_indices_from(features, k=1)  # k=1 excludes the diagonal
        features2[j,i,:] = features[upper_tri_indices]

print('FC calculation done')

accuracies = []
for i in range(features1.shape[1]):
    # Calculate correlation coefficients in one go using broadcasting
    corr_coeffs = np.corrcoef(features1[:, i, :], features2[:, i, :])[0:features1.shape[0],features1.shape[0]:]

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
    accuracies.append((counter1 + counter2) / total_samples)
    print(i)

# Calculate mean and standard deviation
base_array = np.arange(90).reshape(15, 6)
arr1 = base_array[:, 0]
arr2 = base_array[:, 1]
arr3 = base_array[:, 2]
arr4 = base_array[:, 3]
arr5 = base_array[:, 4]
arr6 = base_array[:, 5]
accuracies = np.array(accuracies)
acc1 = accuracies[arr1]
acc2 = accuracies[arr2]
acc3 = accuracies[arr3]
acc4 = accuracies[arr4]
acc5 = accuracies[arr5]
acc6 = accuracies[arr6]
mean_acc1 = np.mean(acc1)
std_acc1 = np.std(acc1)
mean_acc2 = np.mean(acc2)
std_acc2 = np.std(acc2)
mean_acc3 = np.mean(acc3)
std_acc3 = np.std(acc3)
mean_acc4 = np.mean(acc4)
std_acc4 = np.std(acc4)
mean_acc5 = np.mean(acc5)
std_acc5 = np.std(acc5)
mean_acc6 = np.mean(acc6)
std_acc6 = np.std(acc6)

#np.save('pcc_acc_l_56.npy',acc1)
#np.save('pcc_acc_l_84.npy',acc2)
#np.save('pcc_acc_l_140.npy',acc4)
#np.save('pcc_acc_l_300.npy',acc6)