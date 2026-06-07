# -*- coding: utf-8 -*-
# @Time    : 2023/6/1
# @Author  : Yu Wenqi, Xinyi Yan
import torch

tag2ids = {'[PAD]': 0,'B': 1, 'I': 2, 'E': 3,'S': 4,"O": 5}
id2tags = {val: key for key, val in tag2ids.items()}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_path = './data/Abstract320/train1.json'  # 1-5
test_path = './data/Abstract320/test1.json'
vocab_path = './data/Abstract320/vocab.json'
# train_path = './data/Abstract5190/train.json'
# test_path = './data/Abstract5190/test.json'
# vocab_path = './data/Abstract5190/vocab.json'
logfile='./result/log/Abstract320/BiLSTM/train1-1.txt'
save_path = './result/prediction/Abstract320/BiLSTM/train1-1.txt'
feature = "no feature" #no feature/FFD/FN/TFD/FFD+FN/FN+TFD/FFD+TFD/FFD+FN+TFD


# Training parameters for:T5-Large/mT5-Large Chinese character-level AKE
# Use google/mt5-large by default because it is a T5-Large-family checkpoint
# with multilingual SentencePiece vocabulary that is suitable for Chinese text.
t5_model_name = 'google/mt5-large'
t5_feature = 'FFD+FN+TFD'  # no feature/FFD/FN/TFD/FFD+FN/FN+TFD/FFD+TFD/FFD+FN+TFD
t5_max_length = 512
t5_batch_size = 2
t5_eval_batch_size = 2
t5_gradient_accumulation_steps = 16
# For 15 epochs on the small Abstract320 corpus, keep a conservative LR to
# avoid quickly overfitting mT5-Large.  For Abstract5190, 2e-5 is a reasonable
# upper bound if validation F-score is still improving.
t5_lr = 1e-5
t5_weight_decay = 1e-2
t5_dropout_value = 0.1
t5_epochs = 15
t5_label_all_tokens = False
t5_max_grad_norm = 1.0
t5_early_stop = True
t5_early_stop_metric = 'F5'
t5_early_stop_patience = 3
t5_early_stop_min_delta = 1e-4



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
