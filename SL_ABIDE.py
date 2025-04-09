from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear,Conv1d, MaxPool1d
from tqdm import tqdm
from torch.optim import Adam
from pl_bolts.optimizers import LinearWarmupCosineAnnealingLR
from utils import ABIDEDataset
import os
import pickle
from sklearn.metrics import roc_auc_score


class ConvKRegion(nn.Module):

    def __init__(self, k=1, out_size=8, kernel_size=8, pool_size=8, time_series=295):
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

    def __init__(self, model_config, roi_num=200, time_series=393):
        super().__init__()


        self.extract = ConvKRegion(
            out_size=model_config['embedding_size'], kernel_size=model_config['window_size'],
            time_series=time_series, pool_size=model_config['pool_size'])
        self.linear = nn.Sequential(
            Linear(int((roi_num*(roi_num-1))/2), 2),
            nn.Softmax(dim=1))
        


    def forward(self, x):
        x = self.extract(x)
        x = upper_triangular_cosine_similarity(x)
        x = self.linear(x)
        return x

def upper_triangular_cosine_similarity(x):
    N, M, D = x.shape
    
    x_norm = F.normalize(x, p=2, dim=-1)
    
    cosine_similarity = torch.matmul(x_norm, x_norm.transpose(1, 2))
    
    triu_indices = torch.triu_indices(M, M, offset=1)
    
    upper_triangular_values = cosine_similarity[:, triu_indices[0], triu_indices[1]]

    return upper_triangular_values    

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

def train(x, y, encoder_model, optimizer, loss_func):
    encoder_model.train()
    optimizer.zero_grad()
    z = encoder_model(x)
    loss = loss_func(z, F.one_hot(y, num_classes=2).float())
    loss.backward()
    optimizer.step()
    z = z[:,-1].detach().cpu().numpy()
    y = y.to(torch.device("cpu")).numpy()
    auc_score = roc_auc_score(y, z)
    return loss.item(),auc_score


def test(encoder_model, test_data_loader, batch_size, loss_func, device):
    encoder_model.eval()
    with torch.no_grad():
        zs = []
        ys = []
        for (x,y) in test_data_loader:
            zs.append(encoder_model(x.to(device)))
            ys.append(y)
        z = torch.cat(zs,dim=0)
        y = torch.cat(ys,dim=0)
        loss = loss_func(z, F.one_hot(y, num_classes=2).float().to(device))
        z = z[:,-1].cpu().numpy()
        y = y.numpy()
        auc_score = roc_auc_score(y, z)
                   
    return loss.item(), auc_score


def main(config):
    path = os.path.join(config['path_data'],config['dataset'])
           
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
    epochs = config['epochs']
    lr = config['lr']
    save_models = config['save_models']
    save_results = config['save_results']   
    model_config = {}
    model_config['embedding_size'] = config['model_config']['embedding_size']
    model_config['window_size'] = config['model_config']['window_size']
    model_config['pool_size'] = config['model_config']['pool_size']
    
    test_losses = []
    test_aucs = []
    losses_all = []
    val_losses_all = []
    aucs_all = []
    val_aucs_all = []
    for i in range(10):
        train_data, val_data, y_train, y_val,_,_ = custom_train_test_split(train_data1, y_train1, test_size=0.1, random_state=42+i)
        train_dataset = ABIDEDataset(train_data, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle = shuffle, drop_last=True)
        val_dataset = ABIDEDataset(val_data, y_val)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_dataset = ABIDEDataset(test_data, y_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        if 'CC200' in config['dataset']:
            rois = 200
        elif 'CC400' in config['dataset']:
            rois = 392
        encoder_model = SeqenceModel(model_config, rois, config['max_length']).to(device)
        
        loss_func = nn.BCELoss()
        optimizer = Adam(encoder_model.parameters(), lr=lr)
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer=optimizer,
            warmup_epochs=20,
            max_epochs=epochs)
        
        min_loss = 1000
        min_val_loss = 1000
        max_auc = 0
        max_val_auc = 0
        losses = []
        val_losses = []
        aucs = []
        val_aucs = []
        with tqdm(total=epochs, desc='(T)') as pbar:
            for epoch in range(1,epochs+1):
                total_loss = 0.0
                total_auc = 0.0
                batch_count = 0                          
                for batch_idx, (batch_data, batch_labels) in enumerate(train_loader):            
                    loss,auc = train(batch_data.to(device), batch_labels.to(device), encoder_model, optimizer, loss_func)
                    scheduler.step()
                    total_loss += loss
                    total_auc += auc
                    batch_count += 1
                val_loss,val_auc = test(encoder_model,val_loader,batch_size,loss_func,device)
                        
                average_loss = total_loss / batch_count if batch_count > 0 else float('nan')   
                average_auc = total_auc / batch_count if batch_count > 0 else float('nan')  
                losses.append(average_loss)
                val_losses.append(val_loss)
                aucs.append(average_auc)
                val_aucs.append(val_auc)
                pbar.set_postfix({
                    'loss': average_loss, 
                    'auc': average_auc,
                    'val_loss': val_loss, 
                    'val_auc': val_auc
                })
                pbar.update()  
                if average_loss < min_loss:
                    min_loss = average_loss
                    min_loss_model = encoder_model.state_dict()
                    min_loss_epoch = epoch
                if val_loss < min_val_loss:
                    min_val_loss = val_loss
                    min_val_loss_model = encoder_model.state_dict()
                    min_val_loss_epoch = epoch
                if average_auc > max_auc:
                    max_auc = average_auc
                    max_auc_model = encoder_model.state_dict()
                    max_auc_epoch = epoch
                if val_auc > max_val_auc:
                    max_val_auc = val_auc
                    max_val_auc_model = encoder_model.state_dict()
                    max_val_auc_epoch = epoch
            
        encoder_model.load_state_dict(min_val_loss_model)
        test_loss,test_auc = test(encoder_model,test_loader,batch_size,loss_func,device)   
        test_losses.append(test_loss)
        test_aucs.append(test_auc)
        losses_all.append(losses)
        val_losses_all.append(val_losses)
        aucs_all.append(aucs)
        val_aucs_all.append(val_aucs)
        
        if save_models:
            torch.save(min_val_loss_model, os.path.join('models_ABIDE',config['dataset'],'SL','ABIDE_SL_min_val_loss_model_' + str(i) + '.pth'))
            torch.save(max_val_auc_model, os.path.join('models_ABIDE',config['dataset'],'SL','ABIDE_SL_max_val_auc_model_' + str(i) +'.pth'))
    if save_results:        
        np.save(os.path.join('results_ABIDE',config['dataset'],'ABIDE_SL_losses.npy'), losses_all)
        np.save(os.path.join('results_ABIDE',config['dataset'],'ABIDE_SL_aucs.npy'), aucs_all)
        np.save(os.path.join('results_ABIDE',config['dataset'],'ABIDE_SL_Vallosses.npy'), val_losses_all)
        np.save(os.path.join('results_ABIDE',config['dataset'],'ABIDE_SL_ValAucs.npy'), val_aucs_all)
        
        results = {
            "test_loss": test_losses,
            "test_auc": test_aucs
        }
        with open(os.path.join('results_ABIDE','ABIDE_SL_results.pkl'), "wb") as pickle_file:
            pickle.dump(results, pickle_file)

if __name__ == '__main__':
    config = {}
    config['dataset'] = 'DPARSF-CC200'
    config['path_data'] = r'/home/student1/Desktop/Charalampos_Lamprou/SSL_FC_matrix_data/Python'
    config['max_length'] = 393 #according to the wind_size_max from prepare_train_test_ABIDE
    config['batch_size'] = 128
    config['shuffle'] = True
    config['epochs'] = 100
    config['lr'] = 5e-5
    config['model_config'] = {}
    config['model_config']['embedding_size'] = 128
    config['model_config']['window_size'] = 16
    config['model_config']['pool_size'] = 4   
    config['device'] = "cuda:3"
    config['save_models'] = True
    config['save_results'] = True
    main(config)

