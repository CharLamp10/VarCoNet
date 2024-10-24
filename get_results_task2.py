import pickle
import numpy as np
import os  

case = 'CPAC-CC200'

SL_val_aucs = np.load(os.path.join('results_ABIDE',case,'ABIDE_SL_ValAucs.npy'))
SL_val_losses = np.load(os.path.join('results_ABIDE',case,'ABIDE_SL_Vallosses.npy'))


with open(os.path.join('results_ABIDE',case,'ABIDE_SL_results.pkl'), 'rb') as f:
    test_result_SL = pickle.load(f)
    
with open(os.path.join('results_ABIDE',case,'ABIDE_SSL_test_results.pkl'), 'rb') as f:
    test_result_SSL = pickle.load(f)

SSL_val_auc = np.zeros((10,68))   
SSL_val_loss = np.zeros((10,68))  
SSL_test_auc = np.zeros((10,68))   
SSL_test_loss = np.zeros((10,68))  
SL_val_auc = np.zeros((10,))   
SL_val_loss = np.zeros((10,))  
SL_test_auc = np.zeros((10,))   
SL_test_loss = np.zeros((10,))  
for i,test in enumerate(test_result_SSL):
    for j,test1 in enumerate(test):
        SSL_val_auc[i,j] = test1['best_val_auc']
        SSL_val_loss[i,j] = test1['best_val_loss']
        SSL_test_auc[i,j] = test1['best_test_auc']
        SSL_test_loss[i,j] = test1['best_test_loss']
    
        
min_positions_SSL = np.argmin(SSL_val_loss, axis=1)
best_val_aucs_SSL = SSL_val_auc[np.arange(10),min_positions_SSL]
best_val_losses_SSL = SSL_val_loss[np.arange(10),min_positions_SSL]
best_test_aucs_SSL = SSL_test_auc[np.arange(10),min_positions_SSL]
best_test_losses_SSL = SSL_test_loss[np.arange(10),min_positions_SSL]

min_positions_SL = np.argmin(SL_val_losses, axis=1)
best_val_aucs_SL = SL_val_aucs[np.arange(10),min_positions_SL]
best_val_losses_SL = SL_val_losses[np.arange(10),min_positions_SL]
best_test_aucs_SL = np.array(test_result_SL['test_auc'])
best_test_losses_SL = np.array(test_result_SL['test_loss'])

print(f"SSL: mean validation AUC= {np.mean(best_val_aucs_SSL):.2f}, std={np.std(best_val_aucs_SSL):.2f}")
print(f"SSL: mean validation Loss={np.mean(best_val_losses_SSL):.2f}, std={np.std(best_val_losses_SSL):.2f}")
print(f"SSL: mean test       AUC= {np.mean(best_test_aucs_SSL):.2f}, std={np.std(best_test_aucs_SSL):.2f}")
print(f"SSL: mean test       Loss={np.mean(best_test_losses_SSL):.2f}, std={np.std(best_test_losses_SSL):.2f}")
#print('')
#print(f"SL: mean validation AUC= {np.mean(best_val_aucs_SL):.2f}, std={np.std(best_val_aucs_SL):.2f}")
#print(f"SL: mean validation Loss={np.mean(best_val_losses_SL):.2f}, std={np.std(best_val_losses_SL):.2f}")
#print(f"SL: mean test       AUC= {np.mean(best_test_aucs_SL):.2f}, std={np.std(best_test_aucs_SL):.2f}")
#print(f"SL: mean test       Loss={np.mean(best_test_losses_SL):.2f}, std={np.std(best_test_losses_SL):.2f}")