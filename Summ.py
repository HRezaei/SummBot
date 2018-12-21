import _pickle as cPickle
from GenerateDataset import *
import sys
import numpy as np

def sentences_features(text):
    feature_set = []
    all_words = word_tokenize(text)
    all_words = remove_stop_words(all_words)
    word_freq = FreqDist(all_words)
    #graph_model = build_graph_model()
    sentences = sent_tokenize(text)
    doc_features = {
        'num_words': len(all_words),
        'num_sens': len(sentences),
        'num_parag': sum([1 for p in text.split('\n') if len(p) > 0])
    }
    position = 0
    for position in range(len(sentences)):
        sen = sentences[position]
        sen = normalizer.normalize(sen)
        words = remove_stop_words(word_tokenize(sen))
        if len(words) < 1: continue
        features = doc_features.copy()
        add_features(features, words, sentences, word_freq, position)
        feature_set.append(features)
        position += 1
    return (feature_set, sentences)

def summ(text, clf, category):
    sens_feats, sentences = sentences_features(text)
    feature_set = []
    for sen in sens_feats:
        row = []
        for attr in valid_features:
            if attr == 'category':
                row.append(category_map[category])
            else:
                row.append(sen[attr])
        feature_set.append(row)

    result = clf.predict(feature_set)
    #result = np.random.rand(len(feature_set))
    dictv = {i:result[i] for i in range(len(result))}
    ranked = sorted(dictv.items(), key=operator.itemgetter(1), reverse=True)
    #print(ranked)
    cut = [i for (i,j) in ranked[:4]]
    #return cut
    selected = sorted(cut)
    summary = [sentences[i] for i in selected]
    #for i in range(len(result)):
    #    y = result[i]
    return " ".join(summary)



