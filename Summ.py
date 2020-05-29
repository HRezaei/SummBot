import _pickle as cPickle
from GenerateDataset import *
import sys
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def sentences_features(text, category):
    feature_set = document_feature_set(text, category)
    all_words = word_tokenize(text)
    all_words = remove_stop_words(all_words)
    word_freq = FreqDist(all_words)
    #graph_model = build_graph_model()
    sentences = sent_tokenize(text)
    doc_features = {
        'doc_words': len(all_words),
        'doc_sens': len(sentences),
        'doc_parag': sum([1 for p in text.split('\n') if len(p) > 0])
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


def summ(text, clf, category, used_feature_names, cutoff=None):
    feature_set, sentences = document_feature_set(text, category)
    if cutoff:
        summary_len = cutoff
    else:
        summary_len = 4
        text_len = len(sentences)
        if text_len > 30:
            summary_len = round(text_len/6)
        elif text_len > 10:
            summary_len = 5

    feature_set_filtered = []
    for sen in feature_set:
        row = []
        for attr in used_feature_names:
            if attr == 'category':
                row.append(category_map[sen[attr]])
            else:
                row.append(sen[attr])
        feature_set_filtered.append(row)

    feature_set_filtered = np.array(feature_set_filtered)
    normalize_dataset(feature_set_filtered, used_feature_names)


    result = clf.predict(feature_set_filtered)
    #result = np.random.rand(len(feature_set))
    dictv = {i:result[i] for i in range(len(result))}
    ranked = sorted(dictv.items(), key=operator.itemgetter(1), reverse=True)
    #print(ranked)
    cut = [i for (i,j) in ranked[:summary_len]]
    #return cut
    selected = sorted(cut)
    summary = [sentences[i] for i in selected]
    #for i in range(len(result)):
    #    y = result[i]
    return summary
    return " ".join(summary)



