#!/usr/bin/env python
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

# accepted relations among synset and lemmas
syn_relations = {
    'hyponym': lambda syn: syn.hyponyms(),
    'instance_hyponym': lambda syn: syn.instance_hyponyms(),
    'member_meronym': lambda syn: syn.member_meronyms(),
    'has_part': lambda syn: syn.part_meronyms(),
    'topic_domain': lambda syn: syn.topic_domains(),
    'usage_domain': lambda syn: syn.usage_domains(),
    '_member_of_domain_region': lambda syn: syn.region_domains(),
    'attribute': lambda syn: syn.attributes(),
    'entailment': lambda syn: syn.entailments(),
    'cause': lambda syn: syn.causes(),
    'also_see': lambda syn: syn.also_sees(),
    'verb_group': lambda syn: syn.verb_groups(),
    'similar_to': lambda syn: syn.similar_tos()
}
lem_relations = {
    'antonym': lambda lem: lem.antonyms(),
    'derivationally_related_form': lambda lem: lem.derivationally_related_forms(),
    'pertainym': lambda lem: lem.pertainyms()
}
syn2lem_relations = {
    'lemma': lambda syn: syn.lemma_names()
}


def generate_syn_triples(entity_id_map, rel_id_map):
    """
    This function generates the synset triple.
    :param entity_id_map: map of the entities.
    :param rel_id_map: map of the relations.
    :return: triple
    """
    result = []
    for synset in list(wn.all_synsets()):
        h_id = entity_id_map.get(synset.name())
        if h_id is None:
            print('No entity id for ', synset)
            continue
        for synrel, srfn in syn_relations.items():
            r_id = rel_id_map.get(synrel)
            if r_id is None:
                print('No rel id for', synrel)
                continue
            for obj in srfn(synset):
                t_id = entity_id_map.get(obj.name())
                if t_id is None:
                    print('No entity id for object', obj)
                    continue
                result.append((h_id, t_id, r_id))

        for rel, fn in syn2lem_relations.items():
            r_id = rel_id_map.get(rel)
            if r_id is None:
                print('No rel id for', rel)
                continue
            for obj in fn(synset):
                lem = obj.lower()
                t_id = entity_id_map.get(lem)
                if t_id is None:
                    print('No entity id for object', obj, 'lowercased:', lem)
                    continue
                result.append((h_id, t_id, r_id))
    return result


def generate_lem_triples(entity_id_map, rel_id_map):
    """
    This function generates the lemmas triple.
    :param entity_id_map: map of the entities.
    :param rel_id_map: map of the relations.
    :return: triple
    """
    result = []
    for lemma in list(wn.all_lemma_names()):
        h_id = entity_id_map.get(lemma)
        if h_id is None:
            print('No entity id for lemma', lemma)
            continue
        _lems = wn.lemmas(lemma)
        for lemrel, lrfn in lem_relations.items():
            r_id = rel_id_map.get(lemrel)
            if r_id is None:
                print('No rel id for ', lemrel)
                continue
        for _lem in _lems:
            for obj in lrfn(_lem):
                t_id = entity_id_map.get(obj.name().lower())
                if t_id is None:
                    print('No entity id for obj lemma', obj, obj.name())
                    continue
                result.append((h_id, t_id, r_id))
    return result


def wnet30_holE_bin(out):
    """
    Creates a skge-compatible bin file for training HolE embeddings based on WordNet31.
    :param out: path to the file.
    """
    synsets = [synset.name() for synset in wn.all_synsets()]
    lemmas = [lemma for lemma in wn.all_lemma_names()]
    entities = list(synsets + list(set(lemmas)))
    print('Found %s synsets, %s lemmas, hence %s entities' % (len(synsets), len(lemmas), len(entities)))
    entity_id_map = {ent_name: id for id, ent_name in enumerate(entities)}
    n_entity = len(entity_id_map)

    print("N_ENTITY: %d" % n_entity)
    relations = list(list(syn_relations.keys()) + list(lem_relations.keys()) + list(syn2lem_relations.keys()))
    relation_id_map = {rel_name: id for id, rel_name in enumerate(relations)}
    n_rel = len(relation_id_map)
    print("N_REL: %d" % n_rel)
    print('relations', relation_id_map)

    syn_triples = generate_syn_triples(entity_id_map, relation_id_map)
    print("Syn2syn relations", len(syn_triples))
    lem_triples = generate_lem_triples(entity_id_map, relation_id_map)
    print("Lem2lem relations", len(lem_triples))
    all_triples = syn_triples + lem_triples
    print("All triples", len(all_triples))
    random.shuffle(all_triples)

    test_triple = all_triples[:500]
    valid_triple = all_triples[500:1000]
    train_triple = all_triples[1000:]

    to_pickle = {
        "entities": entities,
        "relations": relations,
        "train_subs": train_triple,
        "test_subs": test_triple,
        "valid_subs": valid_triple
    }

    with open(out, 'wb') as handle:
        pickle.dump(to_pickle, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("wrote to %s" % out)


if __name__ == "__main__":
    out_bin = './holographic-embeddings/datasets/wn30.bin'
    wnet30_holE_bin(out_bin)
    # wn30_holE_out='./holographic-embeddings/wn30_holE_2e.bin'
    # holE_dim=150
    # num_epochs=2
    # python ./holographic-embeddings/kg/run_hole.py --fin ./holographic-embeddings/datasets/wn30.bin --fout ./holographic-embeddings/wn30_holE_2e.bin --nb 100 --me 500 --margin 0.2 --lr 0.1 --ncomp 150