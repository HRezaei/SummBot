import nltk, math, operator
from nltk.probability import FreqDist
from nltk import bleu
import json, sys, hashlib
from hazm import *
from rouge import Rouge
from utilities import *
#from FarsnetLoader import *
import Features
from fractions import Fraction
import numpy as np



def add_features(features, sent, all_sentences_tokenized, word_freq, position):
    '''
    Args:
        sent: array of words
    '''
    total_sentences = len(all_sentences_tokenized)
    features["tfisf"] = Features.tf_isf_score(sent, all_sentences_tokenized, word_freq)
    features["cosine_position"] = Features.cosine_position_score(position, total_sentences)
    features["tf"] = Features.frequency_score(sent, word_freq)
    features["cue_words"] = Features.cue_words(sent, cue_words)
    Features.pos_ratio_based(features, sent)
    return features

def are_similar_rouge(sen1, sen2):
    scores = rouge.get_scores(sen1, sen2)
    return (scores[0]['rouge-2']['f'] >= 0.7)


def are_similar(sen1, sen2):
    threshold = 0.5
    denominator = float(len(set(sen1).union(sen2)))
    if denominator > 0:
        ratio = len(set(sen1).intersection(sen2)) / denominator
    else:
        ratio = 0
    return (ratio >= threshold, ratio)


def avg_bleu_score(sen, summaries, avg=False):
    min_length = 5
    if avg:
        total = 0
        for summ in summaries:
            total += bleu([summ], sen, smoothing_function=chencherry.method2)
        score = total / len(summaries)
    else:
#        score = bleu(summaries, sen, smoothing_function=chencherry.method2)
        score = nltk.translate.bleu_score.modified_precision(summaries, sen, 2)
        if len(sen) < min_length:
            score *= np.exp(1-(min_length/len(sen)))
    return score

def build_feature_set():
    datasetJson = read_file('resources/pasokh/all.json') 
    pasokh = json.loads(datasetJson)
    output = {}
    for key in pasokh:
        doc = pasokh[key]
        text = doc["text"]
        #title = doc["title"]
        feature_set, tmp = document_feature_set(text, key, doc['summaries'])
        output[key.replace(".", "")] = feature_set   

    return output


def average_similarity(sen, gold_summaries):
    total_similarity = 0
    for key in gold_summaries:
        max = 0
        for sum_sen in gold_summaries[key]['sens']:
            (similar, similarity) = are_similar(sen, sum_sen)
            if(similarity > max):
                max = similarity
        total_similarity += max
    return total_similarity/len(gold_summaries)
                

def encode_complex(obj):
    if isinstance(obj, Fraction):
        return obj.numerator/ obj.denominator
    raise TypeError(repr(obj) + " is not JSON serializable")


def document_feature_set(text, category, golden_summaries=[], key=''):
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
        tagged_sentences.append(tagged_sen)

    doc_features = {
        'doc_words': len(all_words),
        'doc_sens': len(sentence_words),
        'doc_parag': sum([1 for p in text.split('\n') if len(p) > 0]),
        'category': category,
        'doc_verbs': num_verbs,
        'doc_adjcs': num_adjcs,
        'doc_advbs': num_advbs,
        'doc_nouns': num_nouns
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
            included = (features['target'] > 0.5)
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


normalizer = Normalizer()
stemmer = Stemmer()

from nltk.translate.bleu_score import SmoothingFunction
chencherry = SmoothingFunction()

def generate_dataset():    
    feats = build_feature_set()
    f_file = open('features.json', '+w')
    json.dump(feats, f_file, ensure_ascii=False, default=encode_complex)
    f_file.close()
    print("features.json has been written successfully")
    '''f_file = open('referense_sens.json', '+w')
    json.dump(refs, f_file, ensure_ascii=False, default=encode_complex)
    f_file.close()'''

    output = [','.join(all_features) + " \r\n"]

    for key in feats:
        for (sen, target) in feats[key]:
            row = []
            for attr in all_features:
                row.append(str(sen[attr]))
            output.append(','.join(row) + "\r\n")
            '''str(sen['id']) + "," + str(sen['pos_nn_ratio']) + "," + str(sen['pos_ve_ratio']) + "," +\
            str(sen['pos_aj_ratio']) + "," + str(sen['pos_av_ratio']) + "," + str(sen['tfisf']) + "," + \
            str(sen['tf']) + "," + str(sen['cue_words']) + "," + str(sen['cosine_position']) + "," + \
            str(target)+ "\r\n" )'''

    f_file = open('dataset.csv', '+w')
    f_file.writelines(output)
    f_file.close()
    print("dataset.csv has been written successfully")


if len(sys.argv) > 1 and sys.argv[1] == 'all':
    generate_dataset()
