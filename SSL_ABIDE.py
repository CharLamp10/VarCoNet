from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear,Conv1d, MaxPool1d
from utils import InfoNCE
from tqdm import tqdm
from torch.optim import Adam
from utils import DualBranchContrast
from pl_bolts.optimizers import LinearWarmupCosineAnnealingLR
from utils import ABIDEDataset
import random
import os
from classifier import LREvaluator
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

    def __init__(self, model_config, roi_num=200, time_series=180):
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
    
def generate_random_numbers(n, a1, b1, non_zeros):
    r = np.array([random.randint(a1,b1) for _ in range(n)])
    nonzeros = non_zeros[np.arange(n),r]
    r1s = []
    r2s = []
    for b in nonzeros:
        r1 = random.randint(0,b)
        r2 = random.randint(0,b) 
        while r1 == r2:
            r2 = random.randint(0,b) 
        r1s.append(r1)
        r2s.append(r2)
    r1 = np.array(r1s)
    r2 = np.array(r2s)
    return r1,r2,r


def custom_train_test_split(X, y, test_size=0.1, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)

    class_0_indices = np.where(y == 0)[0]
    class_1_indices = np.where(y == 1)[0]
    
    n_val_samples_per_class = min(len(class_0_indices), len(class_1_indices), int(test_size * len(X) / 2))

    val_class_0_indices = np.random.choice(class_0_indices, n_val_samples_per_class, replace=False)
    val_class_1_indices = np.random.choice(class_1_indices, n_val_samples_per_class, replace=False)

    val_indices = np.concatenate([val_class_0_indices, val_class_1_indices])

    train_indices = np.setdiff1d(np.arange(len(X)), val_indices)

    X_train = [X[i] for i in train_indices]
    X_val = [X[i] for i in val_indices]
    y_train, y_val = y[train_indices], y[val_indices]

    return X_train, X_val, y_train, y_val, train_indices, val_indices


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


def test(encoder_model, train_loader,val_loader,test_loader, batch_size, device, num_epochs, lr):
    encoder_model.eval()
    with torch.no_grad():
        outputs_train = []
        y_train = []
        for (x,y) in train_loader:
            x = x.to(device)
            y_train.append(y)
            outputs_train.append(encoder_model(x))
        outputs_train = torch.cat(outputs_train, dim=0).clone().detach()
        y_train = torch.cat(y_train,dim=0).to(device)
        
        outputs_val = []
        y_val = []
        for (x,y) in val_loader:
            x = x.to(device)
            y_val.append(y)
            outputs_val.append(encoder_model(x))
        outputs_val = torch.cat(outputs_val, dim=0).clone().detach()
        y_val = torch.cat(y_val,dim=0).to(device)
        
        outputs_test= []
        y_test = []
        for (x,y) in test_loader:
            x = x.to(device)
            y_test.append(y)
            outputs_test.append(encoder_model(x))
        outputs_test = torch.cat(outputs_test, dim=0).clone().detach()
        y_test = torch.cat(y_test,dim=0).to(device)
        
    result = LREvaluator(num_epochs = num_epochs,learning_rate=lr).evaluate(encoder_model, outputs_train, y_train, outputs_val, y_val, outputs_test, y_test, 2, device)
                
    return result


def main(config):

    path = os.path.join(config['path_data'],config['dataset'])
            
    data = np.load(os.path.join(path,'ABIDE_train_list_MA.npz'))
    MA1 = []
    for key in data:
        MA1.append(data[key])
    
    data = np.load(os.path.join(path,'ABIDE_train_list_SA.npz'))
    SA1 = []
    for key in data:
        SA1.append(data[key])
    
    data = np.load(os.path.join(path,'ABIDE_train_list.npz'))
    train_data1 = []
    for key in data:
        train_data1.append(data[key])
    
    data = np.load(os.path.join(path,'ABIDE_test_list.npz'))
    test_data = []
    for key in data:
        test_data.append(data[key])
    
    y_train1 = np.load(os.path.join(path,'ABIDE_class_train.npy'))    
    y_test = np.load(os.path.join(path,'ABIDE_class_test.npy'))
    
        
    
    device = torch.device(config['device']) if torch.cuda.is_available() else torch.device("cpu")
    batch_size = config['batch_size']
    shuffle = config['shuffle']
    tau = config['tau']
    epochs = config['epochs']
    lr = config['lr']
    epochs_cls = config['epochs_cls']
    lr_cls = config['lr_cls']
    eval_epochs = list(range(3, epochs+1)) 
    save_models = save_models = config['save_models']
    save_results = config['save_results'] 
    
    model_config = {}
    model_config['embedding_size'] = config['model_config']['embedding_size']
    model_config['window_size'] = config['model_config']['window_size']       
    model_config['pool_size'] = config['model_config']['pool_size']  
    
    
    losses_all = []
    test_result_all = []
    for i in range(10):
        train_data, val_data, y_train, y_val, train_indices, val_indices = custom_train_test_split(train_data1, y_train1, test_size=0.1, random_state=42+i)
        MA = [MA1[i] for i in train_indices]
        SA = [SA1[i] for i in train_indices]
        
        train_dataset = ABIDEDataset(train_data, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle = shuffle)
        val_dataset = ABIDEDataset(val_data, y_val)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_dataset = ABIDEDataset(test_data, y_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        MA_loader = DataLoader(MA, batch_size=batch_size, shuffle = shuffle)
        SA_loader = DataLoader(SA, batch_size=batch_size, shuffle = shuffle)  
        
        if 'CC200' in config['dataset']:
            rois = 200
        elif 'CC400' in config['dataset']:
            rois = 392
        encoder_model = SeqenceModel(model_config, rois, config['max_length']).to(device)
        contrast_model = DualBranchContrast(loss=InfoNCE(tau=tau),mode='L2L').to(device)
        
            
        optimizer = Adam(encoder_model.parameters(), lr=lr)
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer=optimizer,
            warmup_epochs=config['warm_up_epochs'],
            max_epochs=epochs)
              
        min_loss = 1000
        min_val_loss = 1000
        max_val_auc = 0
        test_result = []
        losses = []
        with tqdm(total=epochs, desc='(T)') as pbar:
            for epoch in range(1,epochs+1):
                total_loss = 0.0
                batch_count = 0                
                        
                for batch_idx, (sample_inds_sa, sample_inds_ma) in enumerate(zip(SA_loader.batch_sampler, MA_loader.batch_sampler)):
                    batch_loader = DataLoader(SA, batch_size=len(sample_inds_sa))
                    batch_data = next(iter(batch_loader))
                    all_zeros = (batch_data[:,:,:,0] == 0).all(dim=-1)
                    non_zeros = np.zeros((batch_data.shape[0],batch_data.shape[1]))
                    for j in range(batch_data.shape[0]):
                        for n in range(batch_data.shape[1]):
                            non_zeros[j,n] = torch.min(torch.where(all_zeros[j,n,:])[0])-1
                    random_inds1,random_inds2,random_inds = generate_random_numbers(len(sample_inds_sa),0,batch_data.shape[1]-1,non_zeros)
                    batch_data1_sa = batch_data[np.arange(len(sample_inds_sa)), random_inds, random_inds1]
                    batch_data2_sa = batch_data[np.arange(len(sample_inds_sa)), random_inds, random_inds2]
                    
                    batch_loader = DataLoader(MA, batch_size=len(sample_inds_ma))
                    batch_data = next(iter(batch_loader))
                    random_inds1,random_inds2 = generate_random_numbers_with_distance(len(sample_inds_ma), 5, 8-1)
                    batch_data1_ma = batch_data[np.arange(len(sample_inds_ma)), random_inds1]
                    batch_data2_ma = batch_data[np.arange(len(sample_inds_ma)), random_inds2]
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
                    res = test(encoder_model, train_loader, val_loader, test_loader, batch_size, device, epochs_cls, lr_cls)
                    test_result.append(res) 
                    if res['best_val_auc'] > max_val_auc:
                        max_val_auc = res['best_val_auc']
                        max_val_auc_model = encoder_model.state_dict()
                        max_val_auc_epoch = epoch
                    if res['best_val_loss'] < min_val_loss:
                        min_val_loss = res['best_val_loss']
                        min_val_loss_model = encoder_model.state_dict()
                        min_val_loss_epoch = epoch
        losses_all.append(losses)
        test_result_all.append(test_result)
        if save_models:
            torch.save(max_val_auc_model, os.path.join('models_ABIDE',config['dataset'],'SSL','ABIDE_SSL_max_val_auc_model_' + str(i) + '.pth'))
            torch.save(min_val_loss_model, os.path.join('models_ABIDE',config['dataset'],'SSL','ABIDE_SSL_min_val_loss_model_' + str(i) + '.pth'))
            torch.save(min_loss_model, os.path.join('models_ABIDE',config['dataset'],'SSL','ABIDE_SSL_min_loss_model_' + str(i) + '.pth'))
                
                
    if save_results:           
        np.save(os.path.join('results_ABIDE',config['dataset'],'ABIDE_SSL_losses.npy'), losses_all)
        with open(os.path.join('results_ABIDE',config['dataset'],'ABIDE_SSL_test_results.pkl'), 'wb') as f:
            pickle.dump(test_result_all,f)


if __name__ == '__main__':
    config = {}
    config['dataset'] = 'DPARSF-CC200'
    config['path_data'] = r'/home/student1/Desktop/Charalampos_Lamprou/SSL_FC_matrix_data/Python'
    config['max_length'] = 393                      #according to the wind_size_max from prepare_train_test_ABIDE
    config['batch_size'] = 128
    config['shuffle'] = True
    config['tau'] = 0.02
    config['epochs'] = 70                           #epochs for the contrastive learning
    config['warm_up_epochs'] = 20                   #warm-up epochs for the contrastive learning scheduler
    config['lr'] = 2e-4                             #learning rate for the contrastive learning
    config['epochs_cls'] = 300                      #epochs for the linear layer (classification)
    config['lr_cls'] = 5e-4                         #learning rate for the linear layer (classification)
    config['model_config'] = {}
    config['model_config']['embedding_size'] = 128
    config['model_config']['window_size'] = 16
    config['model_config']['pool_size'] = 4   
    config['device'] = "cuda:3"
    config['save_models'] = True
    config['save_results'] = True
    main(config)
