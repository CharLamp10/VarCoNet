# %%
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear,Conv1d, MaxPool1d,GRU
from tqdm import tqdm
from torch.optim import Adam
from pl_bolts.optimizers import LinearWarmupCosineAnnealingLR
from torch.optim.lr_scheduler import LinearLR
import random
import math
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, roc_auc_score, confusion_matrix


# %%
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

class GruKRegion(nn.Module):

    def __init__(self, kernel_size=8, layers=4, out_size=8, dropout=0.5):
        super().__init__()
        self.gru = GRU(1, kernel_size, layers,
                       bidirectional=True, batch_first=True)

        self.kernel_size = kernel_size

        self.linear = nn.Sequential(
            nn.Dropout(dropout),
            Linear(kernel_size*2, kernel_size),
            nn.LeakyReLU(negative_slope=0.2),
            Linear(kernel_size, out_size)
        )

    def forward(self, x):

        x = torch.transpose(x, 1, 2)
        b, k, d = x.shape

        x = x.contiguous().view((b*k, 1, d))
        zero_mask = (x == 0)
        lengths = zero_mask.squeeze(1).float().argmax(dim=1)
        no_padding = ~zero_mask.squeeze(1).any(dim=1)
        lengths[no_padding] = d
        x = torch.transpose(x, 1, 2)
        x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        x, h = self.gru(x)
        x, _ = pad_packed_sequence(x, batch_first=True)
        all_zeros = (x == 0).all(dim=-1)
        if not all_zeros.any():
            x = x[:,-1,:]
        else:
            last_non_zero_pos = all_zeros.float().argmax(dim=1)
            x = x[torch.arange(x.size(0)),last_non_zero_pos-1,:]

        x = x.view((b, k, -1))

        x = self.linear(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length = 295):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

def create_padding_mask(self, sequences, pad_token=0):
    # Create a mask where padding tokens (zeros) are True (to be masked)
    mask = (sequences == pad_token).transpose(0, 1)
    return mask

class Transformer(nn.Module):
    def __init__(self, out_size=128, d_model=200, n_layers=1, n_heads=8, dim_feedforward=2048, dropout=0.1, max_len=295):
        super(Transformer, self).__init__()
        self.d_model = d_model

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward, dropout, batch_first = True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, n_layers)
        self.pos_enc = PositionalEncoding(d_model,max_len)
        self.fc_out = nn.Linear(max_len, out_size)     


    def forward(self, x, x_mask=None):       
        x_mask = (x[:, :, 0] == 0)      
        x = self.pos_enc(x)
        x = self.transformer_encoder(x, src_key_padding_mask=x_mask)     
        x = torch.transpose(x, 1, 2)
        x = self.fc_out(x)
        return x


class SeqenceModel(nn.Module):

    def __init__(self, model_config, roi_num=200, time_series=393):
        super().__init__()

        if model_config['extractor_type'] == 'cnn':
            self.extract = ConvKRegion(
                out_size=model_config['embedding_size'], kernel_size=model_config['window_size'],
                time_series=time_series, pool_size=model_config['pool_size'])
        elif model_config['extractor_type'] == 'gru':
            self.extract = GruKRegion(
                out_size=model_config['embedding_size'], kernel_size=model_config['hidden_size'],
                layers=model_config['num_gru_layers'], dropout=model_config['dropout'])
        elif model_config['extractor_type'] == 'transformer':
            self.extract = Transformer(
                out_size=model_config['embedding_size'], d_model=model_config['d_model'],
                n_layers=model_config['num_tr_layers'],n_heads=model_config['num_heads'],
                dim_feedforward=model_config['dim_feedforward'],dropout=model_config['dropout'])
        self.linear = nn.Sequential(
            Linear(int((roi_num*(roi_num-1))/2), 2),
            nn.Softmax(dim=1))


    def forward(self, x):
        x = self.extract(x)
        x = upper_triangular_cosine_similarity(x)
        x = self.linear(x)
        #mask = F.gumbel_softmax(x, tau=1, hard=False)
        #x = x*mask
        return x

def upper_triangular_cosine_similarity(x):
    N, M, D = x.shape
    
    x_norm = F.normalize(x, p=2, dim=-1)
    
    cosine_similarity = torch.matmul(x_norm, x_norm.transpose(1, 2))
    
    triu_indices = torch.triu_indices(M, M, offset=1)
    
    upper_triangular_values = cosine_similarity[:, triu_indices[0], triu_indices[1]]

    return upper_triangular_values    

class ABIDEDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)  # Number of data points

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

def train(x, y, encoder_model, optimizer, loss_func):
    encoder_model.train()
    optimizer.zero_grad()
    z = encoder_model(x)
    loss = loss_func(z, F.one_hot(y, num_classes=2).float())
    z = z[:,-1].detach().cpu().numpy()
    y = y.to(torch.device("cpu")).numpy()
    try:
        auc_score = roc_auc_score(y, z)
    except:
        br=1
    loss.backward()
    optimizer.step()
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
        try:
            auc_score = roc_auc_score(y, z)
        except:
            br=1
                   
    return loss.item(), auc_score


#def main():

path = r'/home/student1/Desktop/Charalampos_Lamprou/SSL_FC_matrix_data/Python/dparsf_cc200/resampled'

       
data = np.load(os.path.join(path,'ABIDE_train_list.npz'))
train_data = []
for key in data:
    train_data.append(data[key])

data = np.load(os.path.join(path,'ABIDE_test_list.npz'))
test_data = []
for key in data:
    test_data.append(data[key])

y_train = np.load(os.path.join(path,'ABIDE_class_train.npy'))    
y_test = np.load(os.path.join(path,'ABIDE_class_test.npy'))
 
device = torch.device("cuda:3") if torch.cuda.is_available() else torch.device("cpu")
batch_size = 256
shuffle = True

train_data, val_data, y_train, y_val = train_test_split(train_data, y_train, test_size=0.15, random_state=42)
train_dataset = ABIDEDataset(train_data, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle = shuffle,drop_last=False)
val_dataset = ABIDEDataset(val_data, y_val)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_dataset = ABIDEDataset(test_data, y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

model_config = {}
model_config['extractor_type'] = 'cnn'
model_config['embedding_size'] = 128  #all
#model_config['d_model'] = 200         #transformer
#model_config['num_tr_layers'] = 2     #transformer
#model_config['num_heads'] = 16        #transformer
#model_config['dim_feedforward'] = 512 #transformer
#model_config['dropout'] = 0.5         #transformer

if model_config['extractor_type'] == 'cnn':
    epochs = 100
    lr = 5*1e-5
    model_config['window_size'] = 16
    model_config['pool_size'] = 4
    
elif model_config['extractor_type'] == 'gru':
    epochs = 100
    lr = 1e-3
    model_config['hidden_size'] = 16
    model_config['num_gru_layers'] = 2
    model_config['dropout'] = 0.00
    
encoder_model = SeqenceModel(model_config, 200, 393).to(device)

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
        # Iterate over the data loader
        total_loss = 0.0
        total_auc = 0.0
        total_val_loss = 0.0
        total_val_auc = 0.0
        batch_count = 0                          
        for batch_idx, (batch_data, batch_labels) in enumerate(train_loader):            
            loss,auc = train(batch_data.to(device), batch_labels.to(device), encoder_model, optimizer, loss_func)
            val_loss,val_auc = test(encoder_model,val_loader,batch_size,loss_func,device)
            scheduler.step()
            total_loss += loss
            total_auc += auc
            total_val_loss += val_loss
            total_val_auc += val_auc
            batch_count += 1
                
        average_loss = total_loss / batch_count if batch_count > 0 else float('nan')   
        average_auc = total_auc / batch_count if batch_count > 0 else float('nan') 
        average_val_loss = total_val_loss / batch_count if batch_count > 0 else float('nan') 
        average_val_auc = total_val_auc / batch_count if batch_count > 0 else float('nan') 
        losses.append(average_loss)
        val_losses.append(average_val_loss)
        aucs.append(average_auc)
        val_aucs.append(average_val_auc)
        pbar.set_postfix({
            'loss': average_loss, 
            'auc': average_auc,
            'val_loss': average_val_loss, 
            'val_auc': average_val_auc
        })
        pbar.update()  
        if average_loss < min_loss:
            min_loss = average_loss
            min_loss_model = encoder_model.state_dict()
            min_loss_epoch = epoch
        if average_val_loss < min_val_loss:
            min_val_loss = average_loss
            min_val_loss_model = encoder_model.state_dict()
            min_val_loss_epoch = epoch
        if average_auc > max_auc:
            max_auc = average_auc
            max_auc_model = encoder_model.state_dict()
            max_auc_epoch = epoch
        if average_val_auc > max_val_auc:
            max_val_auc = average_val_auc
            max_val_auc_model = encoder_model.state_dict()
            max_val_auc_epoch = epoch
        

encoder_model.load_state_dict(min_val_loss_model)
test_loss,test_auc = test(encoder_model,test_loader,batch_size,loss_func,device)   
'''
torch.save(min_val_loss_model, 'ABIDE_binary_min_val_loss_model_' + model_config['extractor_type'] + '.pth')
torch.save(max_val_auc_model, 'ABIDE_binary_max_val_auc_model_' + model_config['extractor_type'] +'.pth')
with open('losses_' + model_config['extractor_type'] + '.txt', 'w') as f:
    for loss in losses:
        f.write(f"{loss}\n")
with open('Binary_ValLosses_' + model_config['extractor_type'] + '.txt', 'w') as f:
    for val_loss in val_losses:
        f.write(f"{val_loss}\n")
with open('Binary_AUCs_' + model_config['extractor_type'] + '.txt', 'w') as f:
    for auc in aucs:
        f.write(f"{auc}\n")
with open('Binary_ValAUCs_' + model_config['extractor_type'] + '.txt', 'w') as f:
    for val_auc in val_aucs:
        f.write(f"{val_auc}\n")
results = {
    "test_loss": test_loss,
    "test_auc": test_auc
}
with open('Binary_results_' + model_config['extractor_type'] + '.pkl', "wb") as pickle_file:
    pickle.dump(results, pickle_file)
'''
#if __name__ == '__main__':
#    main()


#PCC accuracies
#meanAcc1: 0.14, stdAcc1: 0.012
#meanAcc2: 0.20, stdAcc2: 0.009
#meanAcc3: 0.26, stdAcc3: 0.012
#meanAcc4: 0.31, stdAcc4: 0.020
#meanAcc5: 0.37, stdAcc5: 0.024