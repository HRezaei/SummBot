from nltk.probability import FreqDist
import json, sys, hashlib
from rouge import Rouge
from utilities import *
#from FarsnetLoader import *
import Features
import numpy as np


def add_features(features, sent, all_sentences_tokenized, word_freq, position):
    '''
    Args:
        sent: array of words
    '''
    total_sentences = len(all_sentences_tokenized)
    features["tfisf"] = Features.tf_isf_score(sent, all_sentences_tokenized, word_freq)
    features["cosine_position"] = Features.cosine_position_score(position, total_sentences)
    features['position'] = 1/(position+1)
    features["tf"] = Features.frequency_score(sent, word_freq)
    features["cue_words"] = Features.cue_words(sent, cue_words)
    features['len'] = len(sent)
    avg_len = sum([len(s) for s in all_sentences_tokenized])/total_sentences
    features['relative_len'] = len(sent)/avg_len
    Features.pos_ratio_based(features, sent)
    return features



def build_feature_set():
    datasetJson = read_file('resources/pasokh/all.json') 
    pasokh = json.loads(datasetJson)
    output = {}
    for key in pasokh:
        doc = pasokh[key]
        text = doc["text"]
        #title = doc["title"]
        feature_set, tmp = document_feature_set(text, key[4:6], doc['summaries'])
        output[key] = feature_set

    return output


def document_feature_set(text, category, golden_summaries=[], key=''):
    """
    Converts a raw text to a matrix of features.
    Each row corresponds with a sentence in given text
    If golden summaries is passed, it also computes target attributes and a few other additional features
    This function is used both in generating dataset and in summarizing an individual text
    :param text:
    :param category:
    :param golden_summaries:
    :param key:
    :return:
    """
    hash_key = hashlib.md5((text+category).encode('utf-8')).hexdigest()
    if hash_key in document_feature_set.cache:
        return document_feature_set.cache[hash_key]
    feature_set = []
    sentence_words = []
    tagged_sentences = []
    num_verbs = 0  # in doc
    num_nouns = 0
    num_advbs = 0
    num_adjcs = 0
    doc_nums = 0

    all_words = word_tokenize(text)
    all_words = remove_stop_words(all_words)
    word_freq = FreqDist(all_words)

    sentences = sent_tokenize(text)

    for sen in sentences[:]:
        normal_sen = normalizer.normalize(sen)
        words = remove_stop_words(word_tokenize(normal_sen))
        if len(words) < 1:
            sentences.remove(sen)
            continue
        sentence_words.append(words)
        tagged_sen = tagger.tag(words)
        num_nouns += sum(1 if tag in 'N' else 0 for (w, tag) in tagged_sen)
        num_verbs += sum(1 if tag == 'V' else 0 for (w, tag) in tagged_sen)
        num_adjcs += sum(1 if tag == 'AJ' or tag == 'AJe' else 0 for (w, tag) in tagged_sen)
        num_advbs += sum(1 if tag == 'ADV' else 0 for (w, tag) in tagged_sen)
        doc_nums += sum(1 if tag == 'NUM' else 0 for (w, tag) in tagged_sen)
        tagged_sentences.append(tagged_sen)

    doc_features = {
        'doc_words': len(all_words),
        'doc_sens': len(sentence_words),
        'doc_parag': sum([1 for p in text.split('\n') if len(p) > 0]),
        'category': category,
        'doc_verbs': num_verbs,
        'doc_adjcs': num_adjcs,
        'doc_advbs': num_advbs,
        'doc_nouns': num_nouns,
        'doc_nums': doc_nums,
        'political': category == 'PO',
        'social': category == 'SO',
        'sport': category == 'SP',
        'culture': category == 'CU',
        'economy': category == 'EC',
        'science': category == 'SC'
    }

    if golden_summaries:
        normalized_summaries = []
        gold_summaries = {}
        for summ in golden_summaries:
            summary = normalizer.normalize(golden_summaries[summ])
            # word tokenized summaries for computing bleu scores:
            normalized_summaries.append(summary.split())
            summ_sens = sent_tokenize(summary)
            gold_summaries[summ] = {'sens': [remove_stop_words(word_tokenize(sen)) for sen in summ_sens]}

    position = 0
    for sen in sentence_words:
        document_feature_set.id += 1
        words = sentence_words[position]
        features = doc_features.copy()
        add_features(features, words, sentence_words, word_freq, position)
        features['id'] = document_feature_set.id

        if golden_summaries:
            features['target_bleu'] = avg_bleu_score(sen, normalized_summaries)
            features['target_bleu_avg'] = avg_bleu_score(sen, normalized_summaries, True)
            features['target'] = average_similarity(words, gold_summaries)
            included = (features['target'] > similarity_threshold)
            features['included'] = included
            features['source_file'] = key
            features['text'] = ' '.join(sen)
            feature_set.append((features, included))
        else:
            feature_set.append(features)
        position += 1

    document_feature_set.cache[hash_key] = (feature_set, sentences)
    return feature_set, sentences


document_feature_set.id = 0
document_feature_set.cache = {}

from Features import *


cue_words = read_file("resources/cue-words.txt").split()
rouge = Rouge()



#farsnet = importEFromPaj("resources/farsnet/synset_related_to.paj")


#normalizer = Normalizer()
#stemmer = Stemmer()


def generate_dataset():    
    feats = build_feature_set()
    f_file = open('features.json', '+w')
    json.dump(feats, f_file, ensure_ascii=False, default=encode_complex)
    f_file.close()
    print("features.json has been written successfully")
    '''f_file = open('referense_sens.json', '+w')
    json.dump(refs, f_file, ensure_ascii=False, default=encode_complex)
    f_file.close()'''
    write_dataset_csv(feats, 'dataset.csv')


#if len(sys.argv) > 1 and sys.argv[1] == 'all':
#    generate_dataset()
