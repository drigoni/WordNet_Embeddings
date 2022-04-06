"""
Created on 26/01/21
Author: Davide Rigoni
Emails: davide.rigoni.2@phd.unipd.it - drigoni@fbk.eu
Description: this file contains the code for building the wordnet dataset as triple.
Most of the code is copied from: https://colab.research.google.com/github/hybridnlp/tutorial/blob/master/02_knowledge_graph_embeddings.ipynb
"""
import nltk
from nltk.corpus import wordnet as wn
import pickle
import random  # for shuffling list of triples

# download wordnet datasets
nltk.download('wordnet')

if __name__ == "__main__":
    name = 'wn30_holE_1000_150_0.01_0.2'
    dataset_file = './holographic-embeddings/datasets/{}.bin'.format('wn30')

    # reading from model results
    wn30_holE_out = './holographic-embeddings/{}.bin'.format(name)
    with open(wn30_holE_out, 'rb') as fin:
        hole_model = pickle.load(fin)
    model = hole_model['model']
    E = model.params['E']

    # reading dataset
    vec_file = './holographic-embeddings/{}_embeddings.pickle'.format(name)
    vocab_file = './holographic-embeddings/{}_vocab.txt'.format(name)
    with open(dataset_file, 'rb') as fin:
      wn30_data = pickle.load(fin)
    entities = wn30_data['entities']
    print("Entities in the dataset: ", len(entities))

    # saving dictionary as needed
    final_dict = dict()
    for i, w in enumerate(entities):
        word = w.strip()
        embedding = [i for i in E[i]]
        final_dict[word] = embedding

    with open(vec_file, 'wb') as f:
        pickle.dump(final_dict, f)
    print("Saved: ", vec_file)

