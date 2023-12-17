# Rresearch on keyword extraction with low-cost eye-tracking data acquisition technology

## Overview
<b>This is data and source Code for the paper "Rresearch on keyword extraction with low-cost eye-tracking data acquisition technology".</b>

  Over the past three decades, human ocular movements during the process of reading have been widely regarded as reflections of attentional focus and further cognitive processing (Ma et al., 2022). This phenomenon has found extensive applications in the realms of psychology, linguistics, and computer science, as evidenced by numerous studies (cite relevant literature). Presently, a considerable body of research has employed eye-tracking datasets in various natural language processing (NLP) tasks, including text compression, part-of-speech tagging, sentiment analysis, named entity recognition, and keyword extraction (Barrett et al., 2016; Hollenstein et al., 2019; Mishra et al., 2016; Zhang & Zhang, 2021). These studies affirm the enhancing effects of eye-tracking data on the aforementioned tasks.

In this study, we propose an efficient and cost-effective method for collecting eye-tracking data. The SearchGazer script (Papoutsaki et al., 2017) is integrated into a reading platform deployed on a server, allowing simultaneous acquisition of eye-tracking data from multiple participants. Additionally, the judicious application of eye-tracking feature indicators to keyword extraction models poses a crucial question. Zhang et al. proposed two methods for incorporating eye-tracking features into neural network models: one involves treating eye-tracking features as the ground truth for attention output in neural network models with attention mechanisms, and the other entails treating eye-tracking features as external features of the model (Zhang & Zhang, 2021). Building upon this foundation, our study introduces pre-trained language models, leveraging the advantages they hold in the NLP domain.
In summary, this paper contributes in three main aspects:

Firstly, our research introduces a cost-effective and efficient method for eye-tracking data collection. By embedding the SearchGazer script into the reading platform, we achieve low-cost acquisition of eye-tracking data.

Secondly, based on three eye-tracking feature indicators—First Fixation Duration (FFD), Fixation Count (FN), and Total Fixation Duration (TFD)—this study constructs a character-level eye-tracking dataset for Chinese academic texts.

Thirdly, we apply the three aforementioned eye-tracking features and their combinations to the task of extracting keywords from academic texts. The results demonstrate that FFD exhibits the most noticeable and consistent enhancement in the performance of the extraction model. The combination of FFD and FN eye-tracking features yields the best average improvement in model performance, though substantial variations exist across different models.

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
  <img src="https://yan-xinyi.github.io/figures/ET_AKE_1.png" width="60%" alt="表1 关键词抽取测试数据集概况"><br>
  <b>注：</b> 总字符数包含中文字符和标点符号；关键字符为出现过在关键词部分的字符。<br><br>
</div>

## 参数环境
本研究将使用Abstract320和Abstract5190数据集作为测试集。因为摘要文本的最大长度约为500字，所以我们将max_length设置为 512。对样本较少的Abstract320进行五折交叉验证，以减少随机性偏差。Abstract5190数据集则以4:1的比例将其分为训练集和测试集。由于两个数据集的大小不同，因此针对不同模型两个数据集达到拟合所需的学习率和训练轮次也不同。针对样本较小的Abstract320数据集，在BiLSTM和BiLSTM+CRF模型中的训练轮次为30、学习率为0.01；而在基于注意力机制的模型中训练轮次为65、学习率为0.005。Abstract5190数据集在上述四种模型中的训练轮次都为30、学习率为0.003。在三个预训练语言模型中，Abstract320数据集与Abstract5190数据集的训练轮次分别为10和8，学习率都为5e-5。代码环境按照如下版本配置：
- Python==3.8
- Torch==2.0.1+cu118
- torchvision==0.9.0
- Sklearn==0.0
- Numpy 1.25.1+mkl
- nltk==3.6.1
- Tqdm==4.59.0

## 快速开始指南
为了更好的探究数据集中字级眼动特征运用于关键词抽取任务的有效性，本研究通过多种关键词抽取模型对眼动特征进行了测试。关键词抽取模型分为基于循环神经网络的提取模型和基于预训练语言模型的提取模型两类。
### 深度学习模型运行指南
1. <b>参数配置：</b> 在 `config.py` 文件中配置超参数。大致有以下参数需要设置：
    - `train_path`,`test_path`,`vocab_path`,`save_path`: 训练数据、测试数据、词汇数据和结果的路径。
    - `fs_name`, `fs_num`: 认知特征的名称和数量。
    - `epochs`: 指整个训练数据集在训练过程中通过模型的次数。 
    - `lr`: 学习率。
    - `vocab_size`: 词汇量。
    - `embed_dim`,`hidden_dim`: 嵌入层和隐藏层的维度。
    - `batch_size`: 是指在机器学习模型的训练或推理过程中，一次正向/反向传递中一起处理的示例（或样本）的数量。
    - `max_length`: 用于指定文本输入序列所允许的最大长度（标记数），常用于自然语言处理任务，如文本生成或文本分类。
2. <b>构建关键词抽取模型：</b> 运行main.py文件，选择要调用的模型并开始训练。

3. <b>配置加入的眼动特征组合：</b> 修改模型中眼动特征组合。例如，下面的代码表示将所有眼动特征添加到模型中：
   `input = torch.cat([input, inputs['et'][:,:,:3]], dim=-1)`
