from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear,Conv1d, MaxPool1d
from utils import InfoNCE
from tqdm import tqdm
from torch.optim import Adam
#from classifier import LREvaluator
from utils import DualBranchContrast
from pl_bolts.optimizers import LinearWarmupCosineAnnealingLR
import random
import os
import pickle


class ConvKRegion(nn.Module):
    
    def __init__(self, k=1, out_size=8, kernel_size=8, pool_size=8, time_series=180):
        super().__init__()
        self.conv1 = Conv1d(in_channels=k, out_channels=32,
                            kernel_size=kernel_size, stride=2)
        output_dim_1 = (time_series-kernel_size)//2+1

        self.conv2 = Conv1d(in_channels=32, out_channels=32,
                            kernel_size=16)
        output_dim_2 = output_dim_1 - 16 + 1
        
        self.conv3 = Conv1d(in_channels=32, out_channels=16,
                            kernel_size=8)
        output_dim_3 = output_dim_2 - 8 + 1
        
        self.max_pool1 = MaxPool1d(pool_size)
        output_dim_4 = output_dim_3 // pool_size * 16
        
        self.in0 = nn.InstanceNorm1d(time_series)
        self.in1 = nn.BatchNorm1d(32)
        self.in2 = nn.BatchNorm1d(32)
        self.in3 = nn.BatchNorm1d(16)
        self.linear = nn.Sequential(
            Linear(output_dim_4, 32),
            nn.LeakyReLU(negative_slope=0.2),
            Linear(32, out_size)
        )


    def forward(self, x):

        b, k, d = x.shape

        x = self.in0(x)

        x = torch.transpose(x, 1, 2)
        x = x.contiguous()

        x = x.view((b*d, 1, k))

        x = self.conv1(x)

        x = self.in1(x)
        x = self.conv2(x)

        x = self.in2(x)
        x = self.conv3(x)

        x = self.in3(x)
        x = self.max_pool1(x)

        x = x.view((b, d, -1))
        if torch.sum(x==x[0,0,-1]) > d:
            x[x==x[0,0,-1]] = 0
        
        x = self.linear(x)
        return x


class SeqenceModel(nn.Module):

    def __init__(self, model_config, roi_num=384, time_series=180):
        super().__init__()

        self.extract = ConvKRegion(
            out_size=model_config['embedding_size'], kernel_size=model_config['window_size'],
            time_series=time_series, pool_size=model_config['pool_size'])

    def forward(self, x):
        x = self.extract(x)
        x = upper_triangular_cosine_similarity(x)
        return x

def upper_triangular_cosine_similarity(x):
    N, M, D = x.shape
    
    x_norm = F.normalize(x, p=2, dim=-1)
    
    cosine_similarity = torch.matmul(x_norm, x_norm.transpose(1, 2))
    
    triu_indices = torch.triu_indices(M, M, offset=1)
    
    upper_triangular_values = cosine_similarity[:, triu_indices[0], triu_indices[1]]

    return upper_triangular_values    

def removeDuplicates(names,inds):
    names_batch = []
    for ind in inds:
        names_batch.append(names[ind])
    names_unique,counts = np.unique(names_batch,return_counts=True)
    if len(names_unique) == len(names_batch):
        return inds
    else:
        non_common = list(set(names).symmetric_difference(set(names_batch)))
        positions = np.where(counts>1)[0]
        for pos in positions:
            name_dupl = names_unique[pos]
            pos_name = np.where(np.array(names_batch) == name_dupl)[0][1]
            names_batch[pos_name] = non_common[random.randint(0, len(non_common)-1)]
            possible_pos = np.where(np.array(names) == names_batch[pos_name])[0]
            inds[pos_name] = possible_pos[random.randint(0,len(possible_pos)-1)]
            non_common = list(set(non_common).symmetric_difference(set(names_batch)))
        return inds

def generate_random_numbers_with_distance(n, a, d):
    r = np.array([random.randint(0,d) for _ in range(n)])
    r1 = r*a+np.array([random.randint(0,a-1) for _ in range(n)])
    cat = np.floor(r1/a).astype('int')*a
    res = r1-cat
    list_of_options = []
    for i in range(len(res)):
        options = list(range(a))
        options.pop(res[i])
        list_of_options.append(options)
    options = np.array(list_of_options)
    r2 = r*a+options[np.arange(len(res)),np.array([random.randint(0,a-2) for _ in range(n)])]
    if (r1 == r2).any():
        raise Exception('r1 and r2 have common elements')
    return r1,r2
    
def generate_random_numbers(n, a1, b1, a2, b2):
    r = random.randint(a1,b1)
    r = np.array([r for _ in range(n)])
    r1 = np.array([random.randint(a2,b2) for _ in range(n)])
    r2 = np.array([random.randint(a2,b2) for _ in range(n)])
    while (r1 == r2).any():
        r2 = np.array([random.randint(a2,b2) for _ in range(n)])
    return r1,r2,r


def train(x1, x2, x3, x4, encoder_model, contrast_model, optimizer):
    encoder_model.train()
    optimizer.zero_grad()
    z1 = encoder_model(x1)
    z2 = encoder_model(x2)
    z3 = encoder_model(x3)
    z4 = encoder_model(x4)
    loss = contrast_model(z1, z2) + contrast_model(z3, z4)
    loss.backward()
    optimizer.step()
    return loss.item(), z1.shape[1]


def test(encoder_model, test_data1, test_data2, batch_size,device):
    encoder_model.eval()
    with torch.no_grad():
        outputs1 = []
        for data in test_data1:
            data = torch.from_numpy(data).to(device)
            outputs = []
            for i in range(0, data.size(0), batch_size):
                batch = data[i:i + batch_size]
                outputs.append(encoder_model(batch).cpu())
            outputs1.append(torch.cat(outputs, dim=0))
        outputs1 = torch.stack(outputs1)
        
        outputs2 = []
        for data in test_data2:
            data = torch.from_numpy(data).to(device)
            outputs = []
            for i in range(0, data.size(0), batch_size):
                batch = data[i:i + batch_size]
                outputs.append(encoder_model(batch).cpu())
            outputs2.append(torch.cat(outputs, dim=0))
        outputs2 = torch.stack(outputs2)
        
        outputs1 = outputs1.numpy()
        outputs2 = outputs2.numpy()
        accuracies = []
        for i in range(outputs1.shape[1]):
            corr_coeffs = np.corrcoef(outputs1[:, i, :], outputs2[:, i, :])[0:outputs1.shape[0],outputs1.shape[0]:]

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

            total_samples = outputs1.shape[0] + outputs2.shape[0]
            accuracies.append((counter1 + counter2) / total_samples)

        base_array = np.arange(outputs1.shape[1]).reshape(int(outputs1.shape[1]/6), 6)
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
        print('')
        for i in range(1, 7):
            mean_acc = eval(f'mean_acc{i}')
            std_acc = eval(f'std_acc{i}')
            print(f'meanAcc{i}: {mean_acc:.2f}, stdAcc{i}: {std_acc:.3f}')  
            
    return accuracies,mean_acc1,std_acc1,mean_acc2,std_acc2,mean_acc3,std_acc3,mean_acc4,std_acc4,mean_acc5,std_acc5,mean_acc6,std_acc6


def main(config):

    path = config['path_data']
    
    names = []
    with open(os.path.join(path,'names.txt'), 'r') as f:
        for line in f:
            names.append(line.strip())
            
    data = np.load(os.path.join(path,'random_list_MA.npz'))
    MA = []
    for key in data:
        MA.append(data[key])
    
    data = np.load(os.path.join(path,'random_list_SA.npz'))
    SA = []
    for key in data:
        SA.append(data[key])
    
    data = np.load(os.path.join(path,'random_list_test1.npz'))
    test_data1 = []
    for key in data:
        test_data1.append(data[key])
    
    data = np.load(os.path.join(path,'random_list_test2.npz'))
    test_data2 = []
    for key in data:
        test_data2.append(data[key])
    
    
    device = torch.device(config['device']) if torch.cuda.is_available() else torch.device("cpu")
    batch_size = config['batch_size']
    shuffle = config['shuffle']
    tau = config['tau']
    epochs = config['epochs']
    lr = config['lr']
    eval_epochs = config['eval_epochs']
    save_models = config['save_models']
    save_results = config['save_results']
    
    model_config = {}
    model_config['embedding_size'] = config['model_config']['embedding_size']
    model_config['window_size'] = config['model_config']['window_size']
    model_config['pool_size'] = config['model_config']['pool_size']
    
    MA_loader = DataLoader(MA, batch_size=batch_size, shuffle = shuffle)
    SA_loader = DataLoader(SA, batch_size=batch_size, shuffle = shuffle)
    
    encoder_model = SeqenceModel(model_config, 384, config['max_length']).to(device)
    contrast_model = DualBranchContrast(loss=InfoNCE(tau=tau),mode='L2L').to(device)
        
    optimizer = Adam(encoder_model.parameters(), lr=lr)
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer=optimizer,
        warmup_epochs=config['warm_up_epochs'],
        max_epochs=epochs)
              
    min_loss = 1000
    max_acc = 0
    test_result = []
    losses = []
    with tqdm(total=epochs, desc='(T)') as pbar:
        for epoch in range(1,epochs+1):
            total_loss = 0.0
            batch_count = 0                              
            for batch_idx, (sample_inds_sa, sample_inds_ma) in enumerate(zip(SA_loader.batch_sampler, MA_loader.batch_sampler)):
                sample_inds_sa = removeDuplicates(names,sample_inds_sa)
                batch_list = [SA[i] for i in sample_inds_sa]
                batch_loader = DataLoader(SA, batch_size=len(batch_list))
                batch_data = next(iter(batch_loader))
                random_inds1,random_inds2,random_inds = generate_random_numbers(len(batch_list),0,batch_data.shape[1]-1,0,batch_data.shape[2]-1)
                batch_data1_sa = batch_data[np.arange(len(batch_list)), random_inds, random_inds1]
                batch_data2_sa = batch_data[np.arange(len(batch_list)), random_inds, random_inds2]
                
                sample_inds_ma = removeDuplicates(names,sample_inds_ma)
                batch_list = [MA[i] for i in sample_inds_ma]
                batch_loader = DataLoader(MA, batch_size=len(batch_list))
                batch_data = next(iter(batch_loader))
                random_inds1,random_inds2 = generate_random_numbers_with_distance(len(batch_list), 6, 15-1)
                batch_data1_ma = batch_data[np.arange(len(batch_list)), random_inds1]
                batch_data2_ma = batch_data[np.arange(len(batch_list)), random_inds2]
                
                loss,input_dim = train(batch_data1_sa.to(device), batch_data2_sa.to(device),
                                       batch_data1_ma.to(device), batch_data2_ma.to(device),
                                       encoder_model, contrast_model, optimizer)
                scheduler.step()
                total_loss += loss
                batch_count += 1
                    
            average_loss = total_loss / batch_count if batch_count > 0 else float('nan')   
            losses.append(average_loss)
            if average_loss < min_loss:
                min_loss = average_loss
                min_loss_model = encoder_model.state_dict()
                min_loss_epoch = epoch
            pbar.set_postfix({'loss': average_loss})
            pbar.update()        
            
            if epoch in eval_epochs:
                res = test(encoder_model, test_data1, test_data2, batch_size,device)
                test_result.append(res) 
                if res[1]+res[3]+res[5]+res[7]+res[9]+res[11] > max_acc:
                    max_acc = res[1]+res[3]+res[5]+res[7]+res[9]+res[11]
                    max_acc_model = encoder_model.state_dict()
                    max_acc_epoch = epoch
    
    encoder_model.load_state_dict(min_loss_model)
    test_result.append(test(encoder_model, test_data1, test_data2, batch_size,device))
    
    if save_models:                
        torch.save(min_loss_model, os.path.join('models','min_loss_model.pth'))
        torch.save(max_acc_model, os.path.join('models','max_acc_model.pth'))
    if save_results:
        with open(os.path.join('results','losses.txt'), 'w') as f:
            for loss in losses:
                f.write(f"{loss}\n")
        with open(os.path.join('results','test_results.pkl'), 'wb') as f:
            pickle.dump(test_result,f)


if __name__ == '__main__':
    config = {}
    config['path_data'] = r'/home/student1/Desktop/Charalampos_Lamprou/SSL_FC_matrix_data/Python'
    config['max_length'] = 300                      #according to the wind_size_max from prepare_train_data
    config['batch_size'] = 128
    config['shuffle'] = True
    config['tau'] = 0.05
    config['epochs'] = 300
    config['warm_up_epochs'] = 100
    config['lr'] = 5e-3
    config['eval_epochs'] = [1,5,10,20,30,40,50,60,70,80,90,100,125,150,175,200,225,250,275,300]
    config['model_config'] = {}
    config['model_config']['embedding_size'] = 128
    config['model_config']['window_size'] = 32
    config['model_config']['pool_size'] = 4   
    config['device'] = "cuda:3"
    config['save_models'] = True
    config['save_results'] = True
    main(config)

