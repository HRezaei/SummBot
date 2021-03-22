import doc_eng_competition.doc_eng as eng
import utilities
import numpy as np
import operator


def summ(doc, clf, used_feature_names, cutoff=None):
    feature_set, sentences = eng.document_feature_set(doc)
    if cutoff:
        summary_len = cutoff
    else:
        # text_len = len(sentences)
        # summary_len = max(round(text_len/10), 3)
        doc_words = sum([len(sen) for sen in sentences])
        summary_len = max(round(doc_words/10), 3)

    feature_set_filtered = []
    for sen in feature_set:
        if type(sen) is tuple:
            sen = sen[0] # The feature set has been returned from cache, while cache has been
            #   filled with tuples of type (sen, target)
        row = []
        for attr in used_feature_names:
            if attr == 'category':
                row.append(utilities.cnn_category_mapping(sen[attr]))
            else:
                row.append(sen[attr])
        feature_set_filtered.append(row)

    feature_set_filtered = np.array(feature_set_filtered)
    #utilities.normalize_dataset(feature_set_filtered, used_feature_names)
    result = clf.predict(feature_set_filtered)
    # result = np.random.rand(len(feature_set))
    dictv = {i: result[i] for i in range(len(result))}
    ranked = sorted(dictv.items(), key=operator.itemgetter(1), reverse=True)
    # print(ranked)
    # cut = [i for (i, j) in ranked[:summary_len]]
    cut = []
    picked_len = 0
    while picked_len < summary_len:
        picked = ranked.pop(0)
        picked_len += len(sentences[picked[0]])
        cut.append(picked[0])
    if len(cut) < 3:
        remained = 3-len(cut)
        for j in range(remained):
            cut.append(ranked.pop(0)[0])
    # return cut
    selected = sorted(cut)
    summary = [(i, sentences[i]) for i in selected]
    # for i in range(len(result)):
    #    y = result[i]
    return summary
    return " ".join(summary)