# %%
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear,Conv1d, MaxPool1d,GRU
from utils import InfoNCE
from tqdm import tqdm
from torch.optim import Adam
#from classifier import LREvaluator
from utils import DualBranchContrast
from pl_bolts.optimizers import LinearWarmupCosineAnnealingLR
import random
import math
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import os
from classifier import LREvaluator
import pickle
from sklearn.model_selection import train_test_split


# %%
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

    def forward(self, raw):

        raw = torch.transpose(raw, 1, 2)
        b, k, d = raw.shape

        x = raw.contiguous().view((b*k, 1, d))
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
    def __init__(self, d_model, max_seq_length = 180):
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
    def __init__(self, out_size=128, d_model=384, n_layers=1, n_heads=8, dim_feedforward=2048, dropout=0.1, max_len=180):
        super(Transformer, self).__init__()
        self.d_model = d_model

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward, dropout, batch_first = True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, n_layers)
        self.pos_enc = PositionalEncoding(d_model,max_len)
        # Output layer
        self.fc_out = nn.Linear(max_len, out_size)
        
        self.dropout = nn.Dropout(dropout)


    def forward(self, x, x_mask=None):       
        x_mask = (x[:, :, 0] == 0)      
        x = self.pos_enc(x)
        x = self.transformer_encoder(x, src_key_padding_mask=x_mask)     
        x = torch.transpose(x, 1, 2)
        x = self.fc_out(x)
        return x


class SeqenceModel(nn.Module):

    def __init__(self, model_config, roi_num=384, time_series=180):
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


    def forward(self, x):
        x = self.extract(x)
        x = upper_triangular_cosine_similarity(x)
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

def generate_random_numbers_with_distance(n, a, d):
    r = np.array([random.randint(0,d) for _ in range(n)])
    r1 = r*a+np.array([random.randint(0,a-1) for _ in range(n)])
    cat = np.floor(r1/5).astype('int')*5
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
    
def generate_random_numbers(n, a, b):
    r1 = np.array([random.randint(a,b) for _ in range(n)])
    r2 = np.array([random.randint(a,b) for _ in range(n)])
    while (r1 == r2).any():
        r2 = np.array([random.randint(a,b) for _ in range(n)])
    return r1,r2


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


#def main():

path = r'/home/student1/Desktop/Charalampos_Lamprou/SSL_FC_matrix_data/Python/dparsf_cc200/resampled'
        
data = np.load(os.path.join(path,'ABIDE_train_list_MA.npz'))
MA = []
for key in data:
    MA.append(data[key])

data = np.load(os.path.join(path,'ABIDE_train_list_MA.npz'))
SA = []
for key in data:
    SA.append(data[key])

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
batch_size = 128
shuffle = True

train_data, val_data, y_train, y_val = train_test_split(train_data, y_train, test_size=0.15, random_state=42)

train_dataset = ABIDEDataset(train_data, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle = shuffle)
val_dataset = ABIDEDataset(val_data, y_val)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_dataset = ABIDEDataset(test_data, y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

MA_loader = DataLoader(MA, batch_size=batch_size, shuffle = shuffle)
SA_loader = DataLoader(SA, batch_size=batch_size, shuffle = shuffle)

model_config = {}
model_config['extractor_type'] = 'cnn'
model_config['embedding_size'] = 128  #all

if model_config['extractor_type'] == 'cnn':
    tau = 0.07
    epochs = 70
    num_epochs = 200
    lr_cls = 0.0005
    lr = 0.0005
    eval_epochs = list(range(1, epochs+1)) #[1,5,10,15,20,25,30,35,40,45,50,55,60,65,70] 
    model_config['window_size'] = 16       
    model_config['pool_size'] = 4         
elif model_config['extractor_type'] == 'gru':
    tau = 0.07
    epochs = 100
    num_epochs = 5000
    lr_cls = 0.0005
    lr = 0.005
    eval_epochs = [1,5,10,20,30,40,50,60,70,80,90,100] 
    model_config['hidden_size'] = 16      
    model_config['num_gru_layers'] = 2    
    model_config['dropout'] = 0.5         

encoder_model = SeqenceModel(model_config, 200, 393).to(device)
contrast_model = DualBranchContrast(loss=InfoNCE(tau=tau),mode='L2L').to(device)

    
optimizer = Adam(encoder_model.parameters(), lr=lr)
scheduler = LinearWarmupCosineAnnealingLR(
    optimizer=optimizer,
    warmup_epochs=20,
    max_epochs=epochs)
      
min_loss = 1000
min_val_loss = 1000
max_val_auc = 0
test_result = []
losses = []
with tqdm(total=epochs, desc='(T)') as pbar:
    for epoch in range(1,epochs+1):
        # Iterate over the data loader
        total_loss = 0.0
        batch_count = 0                
                
        for batch_idx, (sample_inds_sa, sample_inds_ma) in enumerate(zip(SA_loader.batch_sampler, MA_loader.batch_sampler)):
            batch_list = [SA[i] for i in sample_inds_sa]
            batch_loader = DataLoader(SA, batch_size=len(batch_list))
            batch_data = next(iter(batch_loader))
            random_inds1,random_inds2 = generate_random_numbers(len(batch_list),0,batch_data.shape[1]-1)
            batch_data1_sa = batch_data[np.arange(len(batch_list)), random_inds1]
            batch_data2_sa = batch_data[np.arange(len(batch_list)), random_inds2]
            
            batch_list = [MA[i] for i in sample_inds_ma]
            batch_loader = DataLoader(MA, batch_size=len(batch_list))
            batch_data = next(iter(batch_loader))
            random_inds1,random_inds2 = generate_random_numbers_with_distance(len(batch_list), 5, 8-1)
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
            res = test(encoder_model, train_loader, val_loader, test_loader, batch_size, device, num_epochs, lr_cls)
            test_result.append(res) 
            if res['best_val_auc'] > max_val_auc:
                max_val_auc = res['best_val_auc']
                max_val_auc_model = encoder_model.state_dict()
                max_val_auc_epoch = epoch
            if res['best_val_loss'] < min_val_loss:
                min_val_loss = res['best_val_loss']
                min_val_loss_model = encoder_model.state_dict()
                min_val_loss_epoch = epoch
            
            
'''               
torch.save(max_val_auc_model, 'ABIDE_SSL_max_val_auc_model_' + model_config['extractor_type'] + '.pth')
torch.save(min_val_loss_model, 'ABIDE_SSL_min_val_loss_model_' + model_config['extractor_type'] +'.pth')
torch.save(min_loss_model, 'ABIDE_SSL_min_loss_model_' + model_config['extractor_type'] +'.pth')
with open('ABIDE_SSL_losses_' + model_config['extractor_type'] + '.txt', 'w') as f:
    for loss in losses:
        f.write(f"{loss}\n")
with open('ABIDE_SSL_test_results_' + model_config['extractor_type'] + '.pkl', 'wb') as f:
    pickle.dump(test_result,f)
'''


#if __name__ == '__main__':
#    main()
