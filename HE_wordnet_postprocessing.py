import nltk
from nltk.corpus import wordnet as wn
import pickle
import random  # for shuffling list of triples

name = 'wn30_holE_500_150_0.1_0.2'
dataset_file = './holographic-embeddings/data/{}.bin'.format('wn30')

# reading from model results
wn30_holE_out = './holographic-embeddings/{}.bin'.format(name)
with open(wn30_holE_out, 'rb') as fin:
    hole_model = pickle.load(fin)
# print(type(hole_model), len(hole_model))
# for k in hole_model:
#     print(k, type(hole_model[k]))
model = hole_model['model']
E = model.params['E']
# print(type(E), E.shape)

# reading dataset
vec_file = './holographic-embeddings/{}_embeddings.pickle'.format(name)
vocab_file = './holographic-embeddings/{}_vocab.txt'.format(name)
with open(dataset_file, 'rb') as fin:
  wn30_data = pickle.load(fin)
entities = wn30_data['entities']
print("Entities in the dataset: ", len(entities))

# creating files
# with open(vocab_file, 'w', encoding='utf_8') as f:
#     for i, w in enumerate(entities):
#         word = w.strip()
#         print(word, file=f)
#     print("Saved: ", vocab_file)
# with open(vec_file, 'w', encoding='utf_8') as f:
#     for i, w in enumerate(entities):
#         word = w.strip()
#         embedding = E[i]
#         print('\t'.join([word] + [str(x) for x in embedding]), file=f)
#     print("Saved: ", vocab_file)
final_dict = dict()
for i, w in enumerate(entities):
    word = w.strip()
    embedding = [i for i in E[i]]
    final_dict[word] = embedding
    # print(embedding)
    # print('\t'.join([word] + [str(x) for x in embedding]), file=f)
with open(vec_file, 'wb') as f:
    pickle.dump(final_dict, f)
print("Saved: ", vec_file)
