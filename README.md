# 基于低成本眼动追踪数据的关键词抽取研究 
随着认知信号采集与分析技术的发展，眼动追踪语料已经在自然语言处理（NLP）等领域得到应用。但是由于高昂的采集与预处理成本，眼动语料构建以及下游NLP任务的测评研究收到很大程度的限制。关键词抽取是组织检索学术文本重要的技术，传统方法由于文档内部信息的有限性而很难得到提升。为了解决上述问题，本研究提出一种低成本的眼动数据采集方法。通过在阅读标注平台嵌入SearchGazer眼球追踪脚本，实现基于网络摄像头的低成本、多人同步的数据采集。基于构建的中文学术论文的眼动追踪数据集，我们将首次注视时间（FFD）、注视次数（FN）和总注视时间（TFD）这三种眼动特征在关键词抽取模型上进行测评。结果表面，每种眼动特征对关键词抽取任务均有一定提升效果，其中FFD的提升效果最为显著。组合的眼动特征对关键词抽取任务有进一步的提升效果，FFD与TFD的组合提升效果最佳。

## 项目概述
<b>这是 "基于低成本眼动追踪数据的关键词抽取研究"一文的源代码与关键词抽取数据来源。</b>

  近30多年来，阅读过程中的人类眼球运动被广泛视作注意力聚焦和进一步认知处理的反映(Ma et al., 2022)，已被广泛运用于心理学、语言学以及计算机领域的研究（添加相关文献）。目前，已有不少研究将眼动追踪数据集用于文本压缩、词性标注、情感识别、命名实体识别、关键词抽取等NLP任务中(Barrett et al., 2016; Hollenstein et al., 2019; Mishra et al., 2016; Zhang & Zhang, 2021)，并证实了眼动数据对这些任务的提升效果。
  本研究提出一种高效低成本的眼动数据采集方法，将SearchGazer脚本(Papoutsaki et al., 2017)嵌入部署在服务器的阅读平台中，以同时获取多名被试者的眼动数据。此外如何将眼动特征指标合适的运用到关键词抽取模型中是一个很重要的问题。Zhang等提出将眼动特征使用到神经网络模型中的两种方法，一种是在有注意力机制的神经网络模型中将眼动特征作为注意力输出的真值；另一种是将眼动特征作为模型的外部特征(Zhang & Zhang, 2021)。本研究在此基础上增加预训练语言模型，因为预训练语言模型在NLP领域更具有优势。
总的来说，本文的贡献包括以下三个方面：
* 首先，本研究提出了一种低成本高效率的眼动数据采集方法，通过将SearchGazer脚本嵌入阅读平台，实现了低成本的眼动数据采集。
* 其次，本研究基于三项眼动特征指标：FFD、FN及TFD，构建了字级的中文学术文本眼动数据集。
* 然后，我们将上述三种眼动特征及其组合应用于学术文本关键词抽取任务。结果表面，FFD对抽取模型的性能提升最为明显和稳定。且FFD与FN的眼动特征组合对模型性能提升的平均效果最好，但在不同的模型中存在较大的差异性。

## 项目文件结构
<pre>ET_AKE                                       # 根目录
├─ Data                                      # <b>数据文件夹</b>
│    ├── Abstract320                         # 眼动阅读语料
│    │    └── test
│    └── Abstract5190                        # 关键词抽取语料
│         ├── test           
│         └── train
├─ Result                                    # <b>结果存放文件夹</b>
├── main.py                                  # <b>主函数模块</b>
├── bilstm.py                                # BiLSTM模型实现模块
├── bilstmcrf.py                             # BiLSTM+CRF模型实现模块
├── attbilstm.py                             # 基于注意力机制的BiLSTM模型实现模块
├── attbilstm_crf.py                         # 基于注意力机制的BiLSTM+CRF模型实现模块
├── BERT.py                                  # BERT模型实现模块
├── macBERT.py                               # macBERT模型实现模块
├── RoBERTa.py                               # RoBERTa模型实现模块
├── config.py                                # 参数配置模块
└── evaluate.py                              # 评估模块
</pre>

## 数据描述
### 1、眼动阅读语料
本研究语料来自期刊《情报学报》、《情报科学》和《数据分析与知识发现》2000年至2022年间发布的学术文本摘要各110篇（共330篇）。在这些文本中，将10篇《情报学报》论文用于预实验，其余320篇用于正式眼动实验。正式阅读语料包括1215个完整句（含标题），总计64969个字符。此外，构建了Abstract320数据集，用于研究阅读眼动数据对关键词抽取任务的影响。
### 2、关键词抽取语料
我们从与眼动语料相同来源的期刊筛选出不包含英文字符、摘要字符数大于50的学术论文，剔除掉原眼动语料的320篇后构建了关键词抽取数据集Abstract5190，共包括5190篇。数据集具体信息参见表1。
<div align=middle>
<b>表1 关键词抽取测试数据集概况</b><br>
  <img src="https://yan-xinyi.github.io/figures/SSB_KPE_1.png" width="75%" alt="表1 关键词抽取测试数据集概况"><br>
  <b>Note.</b> 1: https://www.ncbi.nlm.nih.gov/pmc/<br><br>
</div>



Upon investigating the existing open-source datasets, it was observed that the HTML texts of each article within the PubMed dataset could be obtained directly from the PubMed website. In order to mitigate the issues of uniformity of section structures within a single domain, this study also selected academic articles from the fields of library and information science (LIS) and computer science (CS) as corpora for KPE. Following the completion of the data collection process, the academic articles with missing author's keyphrases are removed firstly. Subsequently, the HTML tags pertaining to paragraphs and headings within the articles were retained, while all other tags were removed. The details of the dataset are shown in Table 2. 

<div align=left>
<b>Table 2. Number of samples and author's keyphrases of training and test sets in different corpora.</b>
<img src="https://yan-xinyi.github.io/figures/SSB_KPE_2.png" width="75%" alt="Table 2. Number of samples and author's keyphrases of training and test sets in different corpora."><br>
</div>


## Requirements
System environment is set up according to the following configuration:
- Python==3.7
- Torch==1.8.0
- torchvision==0.9.0
- Sklearn==0.0
- Numpy 1.25.1+mkl
- nltk==3.6.2
- Tqdm==4.56.0

## Quick Start
In this paper, two classes of keyword extraction methods are selected to explore the role of chapter structure information on keyword extraction. One class is unsupervised keyword extraction methods based on TF*IDF and TextRank, and the other class is supervised key extraction methods based on Support Vector Machines, Conditional Random Fields, BiLSTM-CRF and BERT-BiLSTM-CRF.
### Implementation Steps for machine learing model
1. <b>Processing:</b> Run the processing.py file to process the data into json format:
    `python processing.py`

   The data is preprocessed to the format like: {['word','Value_et1',... ,'Value_et17','Value_eeg1',... ,'Value_eeg8','tag']}

2. <b>Configuration:</b> Configure hyperparameters in the `config.py` file. There are roughly the following parameters to set:
    - `modeltype`: select which model to use for training and testing.
    - `train_path`,`test_path`,`vocab_path`,`save_path`: path of train data, test data, vocab data and results.
    - `fs_name`, `fs_num`: Name and number of cognitive traits.
    - `run_times`: Number of repetitions of training and testing.
    - `epochs`: refers to the number of times the entire training dataset is passed through the model during the training process. 
    - `lr`: learning rate.
    - `vocab_size`: the size of vocabulary. 37347 for Election-Trec Dataset, 85535 for General-Twitter.
    - `embed_dim`,`hidden_dim`: dim of embedding layer and hidden layer.
    - `batch_size`: refers to the number of examples (or samples) that are processed together in a single forward/backward pass during the training or inference process of a machine learning model.
    - `max_length`: is a parameter that specifies the maximum length (number of tokens) allowed for a sequence of text input. It is often used in natural language processing tasks, such as text generation or text classification.
3. <b>Modeling:</b> Modifying combinations of additive cognitive features in the model.

   For example, the code below means add all 25 features into the model:

         `input = torch.cat([input, inputs['et'], inputs['eeg']], dim=-1)`
5. <b>Training and testing:</b> based on your system, open the terminal in the root directory 'AKE' and type this command:
    `python main.py` 
