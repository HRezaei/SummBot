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
    doc_features = {'num_words': len(all_words), 'num_sens': len(sentences)}
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

def summ(text, clf):
    sens_feats, sentences = sentences_features(text)
    feature_set = []
    for sen in sens_feats:
        row = []
        for attr in valid_features:
            row.append(sen[attr])
        feature_set.append(row)

    result = clf.predict(feature_set)
    #result = np.random.rand(len(feature_set))
    dictv = {i:result[i] for i in range(len(result))}
    ranked = sorted(dictv.items(), key=operator.itemgetter(1), reverse=True)
    #print(ranked)
    cut = [i for (i,j) in ranked[:5]]
    #return cut
    selected = sorted(cut)
    summary = [sentences[i] for i in selected]
    #for i in range(len(result)):
    #    y = result[i]
    return " ".join(summary)

#path = 'resources/pasokh/Single-Dataset/Source/DUC/ALF.CU.13910117.019.txt'


if len(sys.argv) > 2 :
    with open('dtr_regressor.pkl', 'rb') as fid:
        clf = cPickle.load(fid)
    command = sys.argv[1]
    s = 'No summary'
    if command == 'path':
        path = sys.argv[2]
        text = read_file(path)
        s = summ(text, clf)
        file = open(path + '.summ', "w+", encoding='utf8')
        file.write(s)
        file.close()
    elif command == 'text':
        text = sys.argv[2]
        s = summ(text, clf)
        print(s)
    
