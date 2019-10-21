echo "Downloading general requirements"
pip install 'torch>=0.4.1,<0.5.0' 'nltk' 'numpy' 'tqdm>=4.19' 'matplotlib==2.2.3'

# allennlp
pip install 'allennlp'

# bert
pip install 'pytorch-pretrained-bert'

# deepmoji setup
echo "Downloading deepmoji requirements"
pip install 'emoji==0.4.5' 'scipy==0.19.1' 'scikit-learn==0.19.0' 'text-unidecode==1.0'
pip install -e models/torchMoji
python3 models/torchMoji/scripts/download_weights.py

# classifier install
pip install xgboost