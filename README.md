# CAiRE_HKUST submission for SemEval-2019 Task 3 Emo-Context

## System Description Paper
This is the implementation of our submission to Emo-Context. You can find our paper [here](paper/paper.pdf). It will appear in the SemEval Proceedings. Shared task website: https://www.humanizing-ai.com/emocontext.html

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
```
python3 main.py --num_split 10 --max_epochs=100 --pretrain_list glove840B~300 --emb_dim 300 --cuda --hidden_dim 1000 --model LSTM --patient 5 --drop 0.1 --model LSTM --noam --save_path save_final/LSTM_1000_DROP0.1_ATTN_GLOVE/ --pretrain_emb --attn
```

Hierarchical model
```
python3 main_hier.py --num_split 10 --max_epochs=100 --pretrain_list glove840B~300 --emb_dim 300 --cuda --hidden_dim 1000 --model LSTM --patient 5 --drop 0.1 --model LSTM --noam --save_path save_final/HLSTM_1000_DROP0.1_GLOVE/ --pretrain_emb
```

## Evaluation
```--pred_file_path```: prediction file path, ```--ground_file_path```: ground truth file path
```
python3 eval.py --pred_file_path save/HLSTM_1000_DROP0.4_ALL_ATTN/test_voting.txt --ground_file_path data/dev.txt
```

## Predict
```--load_model_path```: trained model path, ```--save_path```: prediction file path
```
python3 predict.py --load_model_path save/TEST2/model_0 --cuda --save_path save/TEST2/
python3 predict_hier.py --load_model_path save/HLSTM_1000_DROP0.4_ATTN_DOUBLE_0.1_GLOVE/model_5 --cuda --double_supervision --save_prediction_path save_final/HLSTM_1000_DROP0.4_ATTN_DOUBLE_0.1_GLOVE/model_5.txt --save_confidence_path save_final/HLSTM_1000_DROP0.4_ATTN_DOUBLE_0.1_GLOVE/model_5_confidence.txt 
```

## Voting
```
python3 voting.py --voting-dir-list save_final/HLSTM_1000_DROP0.4_ATTN_DOUBLE_0.1_GLOVE save_final/HLSTM_1000_DROP0.1_ATTN_DOUBLE_0.1_GLOVE --save_path voting_prediction.txt
```
