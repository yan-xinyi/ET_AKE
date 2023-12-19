# -*- coding: utf-8 -*-
# @Time    : 2023/6/1
# @Author  : Yu Wenqi, Xinyi Yan
import torch

tag2ids = {'[PAD]': 0,'B': 1, 'I': 2, 'E': 3,'S': 4,"O": 5}
id2tags = {val: key for key, val in tag2ids.items()}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_path = './datas/Abstract320/train1.json'  # 1-5
test_path = './datas/Abstract320/test1.json'
vocab_path = './datas/Abstract320/vocab.json'
# train_path = './datas/Abstract5190/train.json'
# test_path = './datas/Abstract5190/test.json'
# vocab_path = './datas/Abstract5190/vocab.json'
logfile='./result/log/Abstract320/BiLSTM/train1-1.txt'
save_path = './result/prediction/Abstract320/BiLSTM/train1-1.txt'
feature = "no feature" #no feature/FFD/FN/TFD/FFD+FN/FN+TFD/FFD+TFD/FFD+FN+TFD



# Training parameters for:att-BiLSTM/att-BiLSTM-CRF 320
fs_num = 0
embed_dim = 64     
hidden_dim = 128    
batch_size = 32    
max_length = 512  
# Vocab parameters
vocab_size = 1226   
dropout_value = 0.5
emb_dropout_value = 0.5
lstm_dropout_value = 0.2
linear_dropout_value = 0.2

lr = 0.005   # 0.003
layers_num = 1
weight_decay = 1e-6   #1e-6
factor = 0.5
patience = 3
epochs = 100

# # Training parameters for:att-BiLSTM/att-BiLSTM-CRF  5190
# fs_num = 2
# embed_dim = 64     
# hidden_dim = 128    
# batch_size = 32    
# max_length = 512  
# Vocab parameters
# vocab_size = 2119   
# dropout_value = 0.5
# emb_dropout_value = 0.5
# lstm_dropout_value = 0.2
# linear_dropout_value = 0.2
#
# lr = 0.003   # 0.003
# layers_num = 1
# weight_decay = 1e-6   #1e-6
# factor = 0.5
# patience = 3
# epochs = 30

# #Training parameters for:BiLSTM/BiLSTM-CRF 5190 
# fs_num = 2
# embed_dim = 64     
# hidden_dim = 128    
# batch_size = 32    
# max_length = 512  
# Vocab parameters
# vocab_size = 2119   
# dropout_value = 0.5
# emb_dropout_value = 0.5
# lstm_dropout_value = 0.2
# linear_dropout_value = 0.2
#
# lr = 0.003   
# layers_num = 1
# weight_decay = 1e-6   
# factor = 0.5
# patience = 3
# epochs = 30


# Training parameters for:BiLSTM/BiLSTM-CRF-320
# fs_num =0
# embed_dim = 64     
# hidden_dim = 128    
# batch_size = 32    
# max_length = 512  
# Vocab parameters
# vocab_size = 1226   
# dropout_value = 0.5
# emb_dropout_value = 0.5
# lstm_dropout_value = 0.2
# linear_dropout_value = 0.2
#
# lr = 0.01   # 0.003
# layers_num = 1
# weight_decay = 1e-6  
# factor = 0.5
# patience = 3
# epochs = 30
