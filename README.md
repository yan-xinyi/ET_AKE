# Rresearch on keyword extraction with low-cost eye-tracking data acquisition technology
With the advancement of cognitive signal acquisition and analysis technologies, eye-tracking corpora have found application in fields such as natural language processing (NLP). However, due to the exorbitant costs associated with collection and preprocessing, the construction of eye-tracking corpora and the evaluative research of downstream NLP tasks face significant constraints. Keyphrase extraction stands as a crucial technique in organizing and retrieving academic texts. Traditional methods struggle to achieve improvement due to the limited information within the documents. In order to address the aforementioned challenges, this study proposes a low-cost method for eye-tracking data collection. By embedding the SearchGazer eye-tracking script into a reading annotation platform, we realize cost-effective, multi-user synchronous data acquisition based on a webcam. Utilizing the constructed eye-tracking dataset of Chinese academic papers, we, for the first time, evaluate three eye-tracking features—First Fixation Duration (FFD), Fixation Count (FN), and Total Fixation Duration (TFD)—on a Keyphrase extraction model. The results indicate that each eye-tracking feature contributes to a certain enhancement in the Keyphrase extraction task, with FFD showing the most significant improvement. The combination of eye-tracking features further enhances the performance in Keyphrase extraction tasks, and the combination of FFD and TFD exhibits the optimal improvement effect.

## 项目概述
<b>这是 "Rresearch on keyword extraction with low-cost eye-tracking data acquisition technology"一文的源代码与关键词抽取数据来源。</b>

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
