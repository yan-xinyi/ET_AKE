# -*- coding: utf-8 -*-
# @Time    : 2022/3/8 17:02
# @Author  : leizhao150
import numpy as np
import logging
from torch import nn, optim
from tqdm import tqdm
from config import *
from evaluate import *
from torch.utils.data import Dataset, DataLoader

torch.cuda.set_device(0)
param_path = "./result/tf_param.pkl"

class TextDataSet(Dataset):

    def __init__(self, data_path, vocab_path):
        # 读取词典
        self.word2ids = {word: i for i, word in enumerate(json.load(open(vocab_path, 'r', encoding='utf-8')))}
        self.id2words = {val: key for key, val in self.word2ids.items()}

        # 读取数据
        self.datas = list(json.load(open(data_path, 'r', encoding='utf-8')).values())

    def __getitem__(self, item):
        text = self.datas[item]
        word_to_ids, tag_to_ids = [], []
        et_list = []
        i, text_len = 0, len(text)
        # print(text_len)
        attention_mask = []
        while i < max_length:
            if i < text_len:
                word = text[i]
                word_id = self.word2ids['[UNK]']
                if word[0] in self.word2ids.keys():
                    word_id = self.word2ids[word[0]]
                tag_id = tag2ids[word[-1]]
                et = list(map(float, word[1: 4]))

                word_to_ids.append(word_id)
                tag_to_ids.append(tag_id)
                et_list.append(et)
                attention_mask.append(1)
            else:
                word_to_ids.append(self.word2ids['[PAD]'])
                tag_to_ids.append(tag2ids['[PAD]'])
                et_list.append([0.0] * 3)
                attention_mask.append(0)
            i += 1
        return {
            "input_ids": torch.tensor(word_to_ids).long().to(device),
            "tags": torch.tensor(tag_to_ids).long().to(device),
            "et": torch.tensor(et_list).float().to(device),
            "attention_mask": torch.tensor(attention_mask).byte().to(device),
            "text_len": text_len
        }

    def __len__(self):
        return len(self.datas)


class BiLSTM(nn.Module):

    def __init__(self, vocab_size, embed_dim,
                 hidden_dim, num_layers=1, num_tags=6):
        super(BiLSTM, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.bilstm = nn.LSTM(input_size=embed_dim + fs_num,
                              hidden_size=hidden_dim, num_layers=num_layers,
                              bidirectional=True, batch_first=True)
        self.layernorm = nn.LayerNorm(normalized_shape=2 * hidden_dim)
        self.fc = nn.Linear(2 * hidden_dim, num_tags)
        self.dropout = nn.Dropout(p=dropout_value)
        self.softmax = torch.nn.Softmax()



    def forward(self, inputs, is_training=True):
        # word embedding
        input = self.embedding(inputs['input_ids'])

        # You can add features like ET and EEG here
        '''
        For example, add 8 EEG features:
         input = torch.cat([input, inputs['eeg']], dim=-1)
         add the first 8 ET features:
         input = torch.cat([input, inputs['et'][:,:,:8]], dim=-1)
        '''

        #全特征
        # input = torch.cat([input, inputs['et'][:,:,:3]], dim=-1)
        #单特征
        # input = torch.cat([input, inputs['et'][:, :,1:3]], dim=-1)
        #双特征
        # input = torch.cat([input, inputs['et'][:,:,::2]], dim=-1)

        # coding
        input, _ = self.bilstm(input)
        # input = self.dropout(torch.relu(self.layernorm(input)))    # -dropout
        output = self.fc(input)
        # logit = self.softmax(output)

        return output



def get_outputs(feats):
    outputs = torch.argmax(torch.softmax(feats,dim=-1),dim=-1)
    return outputs

def bl(train_path, test_path, vocab_path):

    torch.manual_seed(1)
    #处理训练数据和测试数据
    trainLoader = DataLoader(dataset=TextDataSet(train_path, vocab_path), batch_size=batch_size)
    testLoader = DataLoader(TextDataSet(test_path, vocab_path), batch_size=batch_size)

    #设定模型参数
    model = BiLSTM(vocab_size=vocab_size,
                   embed_dim=embed_dim,
                   hidden_dim=hidden_dim).to(device)
    print(model)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    #optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, alpha=0.9)

    # 训练模型
    epoch3,epoch5,epoch10, best_P3, best_R3, best_F3,best_P5,best_R5,best_F5,best_P10,best_R10,best_F10 = 0,0,0,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    fin_targets = []
    fin_prediction = []

    for epoch in range(epochs):
        print("epoch "+ str(epoch + 1) + " is starting!")
        model.train()
        avg_loss,target = [],[]
        with tqdm(trainLoader) as pbar_train:
            for inputs in pbar_train:
                pred = model(inputs)
                target = inputs['tags']
                optimizer.zero_grad()
                criterion = nn.CrossEntropyLoss()
                loss = criterion(pred.reshape(-1, 6), target.reshape(-1))
                # print('loss:{}'.format(loss))
                loss.backward()
                optimizer.step()
                pbar_train.set_description('loss: %.3f'%loss.item())
                avg_loss.append(loss.item())
        print("avg_loss: %.2f"%np.average(avg_loss))
        logging.info("epoch: "+str(epoch))
        logging.info("avg_loss: %.2f"%np.average(avg_loss))

        # 测试模型
        model.eval()
        y_true, y_pred = [], []

        targets = []
        prediction = []
        with tqdm(testLoader) as pbar_test:
            for inputs in pbar_test:
                logit = model(inputs, is_training=False)
                output = get_outputs(logit)
                outputs = output.tolist()

                out_list = []
                for txt in outputs:
                    txt_list = []
                    for item in txt:
                        if item!=0:
                            txt_list.append(item)
                    out_list.append(txt_list)


                true_tags = [tag[:inputs['text_len'][i]] for i, tag in enumerate(inputs['tags'].cuda().tolist())]
                y_true.extend([i for item in true_tags for i in item])
                y_pred.extend([i for item in out_list for i in item])

                # 获取id2word到words_set
                words_set = []

                for item in inputs['input_ids'].tolist():
                    word_list = []
                    for id in item:
                        if id in testLoader.dataset.id2words.keys():
                            word = testLoader.dataset.id2words[id]
                            if word != "[PAD]":
                                word_list.append(word)
                            else:
                                continue;
                    words_set.append(word_list)
                # print(words_set)


                for i in range(len(true_tags)):
                    kw_list = []
                    nkw_list = ""
                    j_len = min(len(true_tags[i]), len(words_set[i]))
                    for j in range(j_len):
                        item = true_tags[i][j]
                        if item == 5:
                            # continue;
                            nkw_list = ""
                        if item == 4:
                            if kw_list not in kw_list:
                                kw_list.append(str(words_set[i][j]))
                        if item == 1:
                            nkw_list += str(words_set[i][j])
                        if item == 2:
                            nkw_list += str(words_set[i][j])
                        if item == 3:
                            nkw_list += str(words_set[i][j])
                            if nkw_list not in kw_list:
                                kw_list.append(nkw_list)
                            nkw_list = ""
                    targets.append(kw_list)


                for i in range(len(out_list)):
                    #每篇摘要的词向量（预测）
                    kw_list = []
                    nkw_list = ""
                    j1_len = min(len(out_list[i]), len(words_set[i]))
                    for j in range(j1_len):
                        item = out_list[i][j]
                        word = str(words_set[i][j])
                        if item == 0:
                            continue;
                        if item == 5:
                            # continue;
                            nkw_list = ""
                        if item == 4:
                            if kw_list not in kw_list:
                                kw_list.append(word)
                        if item == 1:
                            nkw_list += word
                        if item == 2:
                            nkw_list += word
                        if item == 3:
                            nkw_list += word
                            if nkw_list not in kw_list:
                                kw_list.append(nkw_list)
                            nkw_list = ""
                    prediction.append(kw_list)


        # rev_pred, rev_targets = read_results(y_true, y_pred)
        #topk = 3
        P3, R3, F3 = evaluate3(prediction, targets)
        if F3 > best_F3:
            best_P3 = P3
            best_R3 = R3
            best_F3 = F3
            epoch3 = epoch
            fin_targets = targets
            fin_prediction = prediction
        P3_str = "P(3):" + "\t" + str(P3)
        R3_str = "R(3):" + "\t" + str(R3)
        F3_str = "F(3):" + "\t" + str(F3)
        print("按照关键词进行评估：")
        print(P3_str)
        print(R3_str)
        print(F3_str)


        P5, R5, F5 = evaluate5(prediction, targets)
        if F5 > best_F5:
            best_P5 = P5
            best_R5 = R5
            best_F5 = F5
            epoch5 = epoch
            fin_targets = targets
            fin_prediction = prediction
        P5_str = "P(5):" + "\t" + str(P5)
        R5_str = "R(5):" + "\t" + str(R5)
        F5_str = "F(5):" + "\t" + str(F5)
        print("按照关键词进行评估：")
        print(P5_str)
        print(R5_str)
        print(F5_str)

        P10, R10, F10 = evaluate10(prediction, targets)
        if F10 > best_F10:
            best_P10 = P10
            best_R10 = R10
            best_F10 = F10
            epoch10 = epoch

        P10_str = "P(10):" + "\t" + str(P10)
        R10_str = "R(10):" + "\t" + str(R10)
        F10_str = "F(10):" + "\t" + str(F10)
        print("按照关键词进行评估：")
        print(P10_str)
        print(R10_str)
        print(F10_str)

        # logging.info("Epoch" + str(epoch + 1) + "'s Evaluation:")
        # logging.info("P is :" + P_str)
        # logging.info("R is :" + R_str)
        # logging.info("F is :" + F_str)

        # model_log.add_metric(metric_name='test_precision', metric_value=P, epoch=epoch + 1)
        # model_log.add_metric(metric_name='test_recall', metric_value=R, epoch=epoch + 1)
        # model_log.add_metric(metric_name='test_F1', metric_value=F, epoch=epoch + 1)

    print(len(fin_targets))
    print(len(fin_prediction))

    # #将预测结果和目标结果存到txt中
    with open(save_path, mode='a+',encoding='utf-8') as f:
        len1 = len(fin_prediction)
        for i in range(0, len1):
            num1 = len(fin_prediction[i])
            st1 = ''
            for j in range(0, num1):
                word1 = fin_prediction[i][j]
                # print(word1)
                st1 = st1 + word1 + ','
            f.write(st1)
            f.write('\n')
        f.write('----------------------')
        f.write('\n')
        for i in range(0, len1):
            num2 = len(fin_targets[i])
            st2 = ''
            for j in range(0, num2):
                word2 = fin_targets[i][j]
                # print(word2)
                st2 = st2 + word2 + ','
            f.write(st2)
            f.write('\n')


    return epoch3,epoch5,epoch10,best_P3, best_R3, best_F3, best_P5, best_R5, best_F5, best_P10, best_R10, best_F10
    # return best_P, best_R, best_F, best_10F, best_15F



