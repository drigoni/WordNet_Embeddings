# Wordnet_Embeddings
This repository contains the code needed to train wordnet embeddings.
In particular, most of the code is taken from: https://colab.research.google.com/github/hybridnlp/tutorial/blob/master/02_knowledge_graph_embeddings.ipynb

# Dependencies
This project uses the `conda` environment.
In the `root` folder you can find the `.yml` file for the configuration of the `conda` environment and also the `.txt` files for the `pip` environment.

# Usage
### Setup
```bash
# build conda env
conda env create -f env.yml 
conda activate wordnet_embeddings
pip install -r env.txt

# install scikit-kge
git clone https://github.com/hybridNLP2018/scikit-kge
cd scikit-kge
pip install nose
python setup.py sdist
pip install dist/scikit-kge-0.1.tar.gz
cd ../

# clone holographic-embeddings
git clone https://github.com/mnick/holographic-embeddings
```

### Data Pre-processing
In order to make the dataset, type the following command that creates the dataset: `./holographic-embeddings/data/wn30.bin`.
```bash
python HE_wordnet_preprocessing.py
```

### Model Training
In order to train the model:
```bash
cd holographic-embeddings
python kg/run_hole.py --fin data/wn30.bin --fout wn30_holE_500_150_0.1_0.2.bin --ncomp 150 --test-all 100
```

### Pretrained embeddings
In order to save the embeddings:
```bash
cd ../
python HE_wordnet_postprocessing.py
```
Then, the embeddings are saved in a `.pickle` file: `./holographic-embeddings/wn30_holE_500_150_0.1_0.2_embeddings.pickle`

# Licenze
MIT