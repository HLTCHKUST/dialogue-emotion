## Hierarchical Attention for Dialogue Emotion Classification
### CAiRE_HKUST submission for SemEval-2019 Task 3 Emo-Context

<img src="img/pytorch-logo-dark.png" width="10%"> [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

This is the implementation of our submission to Emo-Context. You can find our paper [here](https://www.aclweb.org/anthology/S19-2021). Shared task website: https://www.humanizing-ai.com/emocontext.html

This code has been written using PyTorch >= 1.0. If you use any source codes or datasets included in this toolkit in your work, please cite the following paper. The bibtex is listed below:
<pre>
@inproceedings{winata-etal-2019-caire,
    title = "{CA}i{RE}{\_}{HKUST} at {S}em{E}val-2019 Task 3: Hierarchical Attention for Dialogue Emotion Classification",
    author = "Winata, Genta Indra  and
      Madotto, Andrea  and
      Lin, Zhaojiang  and
      Shin, Jamin  and
      Xu, Yan  and
      Xu, Peng  and
      Fung, Pascale",
    booktitle = "Proceedings of the 13th International Workshop on Semantic Evaluation",
    month = jun,
    year = "2019",
    address = "Minneapolis, Minnesota, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/S19-2021",
    pages = "142--147",
}
</pre>

## Abstract
Detecting emotion from dialogue is a challenge that has not yet been extensively surveyed. One could consider the emotion of each dialogue turn to be independent, but in this paper, we introduce a hierarchical approach to classify emotion, hypothesizing that the current emotional state depends on previous latent emotions. We benchmark several feature-based classifiers using pre-trained word and emotion embeddings, state-of-the-art end-to-end neural network models, and Gaussian processes for automatic hyper-parameter search. In our experiments, hierarchical architectures consistently give significant improvements, and our best model achieves a 76.77% F1-score on the test set.

## Model Architecture
<img src="img/sem.pdf"/>

## Data
You can find the data in [Linkedin page](https://www.linkedin.com/groups/12133338/).

## Setup
To begin, you need to install libraries and their dependencies
```
bash setup.sh
```

## Flags
<!-- Add ```--dev_with_label``` to evaluate with a development set -->
<!-- Add ```--include_test``` to merge train and development set, split the merged dataset, and construct a new set of train and development set. The model will be evaluated with a test set -->

Add ```--fix_pretrain``` to avoid any update on pretrained embedding

## Train
Flat model
```console
❱❱❱ python3 main.py --num_split 10 --max_epochs=100 --pretrain_list glove840B~300 --emb_dim 300 --cuda --hidden_dim 1000 --model LSTM --patient 5 --drop 0.1 --model LSTM --noam --save_path save_final/LSTM_1000_DROP0.1_ATTN_GLOVE/ --pretrain_emb --attn
```

Hierarchical model
```console
❱❱❱ python3 main_hier.py --num_split 10 --max_epochs=100 --pretrain_list glove840B~300 --emb_dim 300 --cuda --hidden_dim 1000 --model LSTM --patient 5 --drop 0.1 --model LSTM --noam --save_path save_final/HLSTM_1000_DROP0.1_GLOVE/ --pretrain_emb
```

## Evaluation
```--pred_file_path```: prediction file path, ```--ground_file_path```: ground truth file path
```console
❱❱❱ python3 eval.py --pred_file_path save/HLSTM_1000_DROP0.4_ALL_ATTN/test_voting.txt --ground_file_path data/dev.txt
```

## Predict
```--load_model_path```: trained model path, ```--save_path```: prediction file path
```console
❱❱❱ python3 predict.py --load_model_path save/TEST2/model_0 --cuda --save_path save/TEST2/
python3 predict_hier.py --load_model_path save/HLSTM_1000_DROP0.4_ATTN_DOUBLE_0.1_GLOVE/model_5 --cuda --double_supervision --save_prediction_path save_final/HLSTM_1000_DROP0.4_ATTN_DOUBLE_0.1_GLOVE/model_5.txt --save_confidence_path save_final/HLSTM_1000_DROP0.4_ATTN_DOUBLE_0.1_GLOVE/model_5_confidence.txt 
```

## Voting
```console
❱❱❱ python3 voting.py --voting-dir-list save_final/HLSTM_1000_DROP0.4_ATTN_DOUBLE_0.1_GLOVE save_final/HLSTM_1000_DROP0.1_ATTN_DOUBLE_0.1_GLOVE --save_path voting_prediction.txt
```

## Bug Report
Feel free to create an issue or send email to giwinata@connect.ust.hk
