# Enhancing Keyword Extraction with Low-cost Eye-tracking Data

## Overview
<b>This is data and source Code for the paper "Enhancing Keyword Extraction with Low-cost Eye-tracking Data".</b>

  Over the past three decades, human ocular movements during the process of reading have been widely regarded as reflections of attentional focus and further cognitive processing (Ma et al., 2022). This phenomenon has found extensive applications in the realms of psychology, linguistics, and computer science, as evidenced by numerous studies (cite relevant literature). Presently, a considerable body of research has employed eye-tracking datasets in various natural language processing (NLP) tasks, including text compression, part-of-speech tagging, sentiment analysis, named entity recognition, and keyphrase extraction (Barrett et al., 2016; Hollenstein et al., 2019; Mishra et al., 2016; Zhang & Zhang, 2021). These studies affirm the enhancing effects of eye-tracking data on the aforementioned tasks.

In this study, we propose an efficient and cost-effective method for collecting eye-tracking data. The SearchGazer script (Papoutsaki et al., 2017) is integrated into a reading platform deployed on a server, allowing simultaneous acquisition of eye-tracking data from multiple participants. Additionally, the judicious application of eye-tracking feature indicators to keyphrase extraction models poses a crucial question. Zhang et al. proposed two methods for incorporating eye-tracking features into neural network models: one involves treating eye-tracking features as the ground truth for attention output in neural network models with attention mechanisms, and the other entails treating eye-tracking features as external features of the model (Zhang & Zhang, 2021). Building upon this foundation, our study introduces pre-trained language models, leveraging the advantages they hold in the NLP domain.
In summary, this paper contributes in three main aspects:

* Firstly, our research introduces a cost-effective and efficient method for eye-tracking data collection. By embedding the SearchGazer script into the reading platform, we achieve low-cost acquisition of eye-tracking data.

* Secondly, based on three eye-tracking feature indicators—First Fixation Duration (FFD), Fixation Count (FN), and Total Fixation Duration (TFD)—this study constructs a character-level eye-tracking dataset for Chinese academic texts.

* Thirdly, we apply the three aforementioned eye-tracking features and their combinations to the task of extracting keyphrases from academic texts. The results demonstrate that FFD exhibits the most noticeable and consistent enhancement in the performance of the extraction model. The combination of FFD and FN eye-tracking features yields the best average improvement in model performance, though substantial variations exist across different models.

## Directory Structure
<pre>ET_AKE                                       # Root directory
├─ data                                      # <b>Experimental datasets</b>
│    ├── Abstract320                         # Eye-tracking Corpus
│    │    └── test
│    └── Abstract5190                        # AKE Corpus
│         ├── test           
│         └── train
├─ result                                    # <b>Result dir</b>
└─ codes                                     # <b>Codest dir</b>
     ├── main.py                             # <b>main function module</b>
     ├── bilstm.py                           # BiLSTM module
     ├── bilstmcrf.py                        # BiLSTM+CRF module
     ├── attbilstm.py                        # Attention based BiLSTM module
     ├── attbilstm_crf.py                    # Attention based BiLSTM+CRF module
     ├── BERT.py                             # BERT module
     ├── macBERT.py                          # macBERT module
     ├── RoBERTa.py                          # RoBERTa module
     ├── config.py                           # Configuration module
     └── evaluate.py                         # Evaluation module
</pre>

## Dataset Discription
### Reading Corpora for eye-tracking test
Eye-tracking Corpora is derived from 330 academic text abstracts published in the journals "Journal of Information Science," "Journal of Information," and "Data Analysis and Knowledge Discovery" between the years 2000 and 2022, with each journal contributing 110 abstracts. Ten articles from the "Journal of Information Science" are designated for the pre-experiment, while the remaining 320 abstracts are utilized for the formal eye-tracking experiment. The formal reading corpus comprises 1,215 complete sentences (including titles), totaling 64,969 characters. Additionally, the Abstract320 dataset is constructed to investigate the impact of reading eye-tracking data on keyphrase extraction tasks.
### AKE Corpora
We selected academic papers from the same journals as the eye-tracking corpus, excluding those containing English characters and with abstract character counts exceeding 50. After removing the initial 320 articles from the eye-tracking corpus, we constructed the keyphrase extraction dataset, named Abstract5190, consisting of a total of 5,190 papers. For specific details about the dataset, please refer to Table 1, providing an overview of the keyphrase extraction test dataset.
<div align=middle>
<b>Table 1: Overview of the keyphrase Extraction Test Dataset</b><br>
  <img src="https://yan-xinyi.github.io/figures/ET_AKE_1.png" width="60%" alt="Table 1: Overview of the keyphrase Extraction Test Dataset"><br>
  <b>Note: </b> The total character count includes both Chinese characters and punctuation marks. The crucial characters encompass those that have appeared in the keyphrase sections.<br><br>
</div>

## Configuration
In this study, the Abstract320 and Abstract5190 datasets will be utilized as the test set. Given that the maximum length of abstract text is approximately 500 characters, we set max_length to 512. To mitigate random bias, a five-fold cross-validation will be employed for the relatively smaller Abstract320 dataset. The Abstract5190 dataset will be split into a training set and a test set at a ratio of 4:1. Due to the disparate sizes of the two datasets, the required learning rates and training epochs for different models also differ. For the smaller Abstract320 dataset, the training epochs for the BiLSTM and BiLSTM+CRF models are set to 30, with a learning rate of 0.01. In contrast, for attention-based models, the training epochs are set to 65, with a learning rate of 0.005. The Abstract5190 dataset is trained for 30 epochs across all four models, with a learning rate of 0.003. Regarding the three pre-trained language models, the training epochs for both Abstract320 and Abstract5190 datasets are 10 and 8, respectively, with a learning rate of 5e-5. The code environment is configured according to the following versions:
- Python==3.8
- Torch==2.0.1+cu118
- torchvision==0.9.0
- Sklearn==0.0
- Numpy 1.25.1+mkl
- nltk==3.6.1
- Tqdm==4.59.0

## Quick Start
To delve more deeply into the effectiveness of character-level eye-tracking features applied to keyphrase extraction tasks within the dataset, this study conducted tests using various keyphrase extraction models. The keyphrase extraction models are categorized into two types: those based on recurrent neural networks and those based on pre-trained language models.
### Parameter Configuration
* Configure hyperparameters in the config.py file. The key parameters to set include the following.
    - `train_path`,`test_path`,`vocab_path`,`save_path`:Paths for training data, test data, vocabulary data, and result storage.
    - `fs_name`, `fs_num`:Name and quantity of cognitive features.
    - `epochs`: The number of passes through the entire training dataset during the training process. 
    - `lr`: Learning rate.
    - `vocab_size`: Vocabulary size.
    - `embed_dim`,`hidden_dim`: Dimensions of the embedding layer and hidden layer.
    - `batch_size`: The number of examples (or samples) processed together in one forward/backward pass during the training or inference process of a machine learning model.
    - `max_length`: Used to specify the maximum allowed length (number of tokens) for the input text sequence, commonly employed in natural language processing tasks such as text generation or text classification.
### Deep Learning Model Operation Guide
* <b>Constructing a Model for keyphrase Extraction:</b> Execute the main.py file, designate the model to be invoked, and commence the training process. Line 30~40 in main.py is the deep learning models.

* <b>Configuring Incorporated Eye-Tracking Feature Combinations:</b> Adjust the amalgamation of eye-tracking features within the model. For instance, the ensuing code signifies the inclusion of all eye-tracking features into the model:
   `input = torch.cat([input, inputs['et'][:,:,:3]], dim=-1)`

### Pretrained Language Model Operation Guide
* <b>Constructing a Model for keyphrase Extraction:</b> Execute the main.py file, designate the model to be invoked, and commence the training process. Line 40~46 in main.py is the pretrained language models.
* <b>Configuring Incorporated Eye-Tracking Feature Combinations:</b>
  - Set the number of features:  `self.classifier = nn.Linear(768 + 2, num_labels)`
  - Adjust the amalgamation of eye-tracking features within the model. For instance, the ensuing code signifies the inclusion of all eye-tracking features into the model:   `input = torch.cat([input, inputs['et'][:,:,:3]], dim=-1)`
## Citation
Please cite the following paper if you use this code and dataset in your work.
    
>Chengzhi Zhang, Xinyi Yan, Wenqi Yu. Enhancing Keyword Extraction with Low-cost Eye-tracking Data. 2023 (Working Paper）.
