import nltk, math, operator
from nltk.probability import FreqDist
import re, json, sys
from hazm import *
from rouge import Rouge
from utilities import *
from FarsnetLoader import *

def remove_stop_words(words):
    #This should be read once instead of every time this function is caled
    return [word for word in words if word not in stop_words and re.sub("\s|\u200c", "", word).isalnum()]


def generate_features(sent, all_sentences, word_freq, position, title):
    '''
    Args:
        sent: array of words
    '''
    features = {}
    total_sentences = len(all_sentences)
    all_sentences_tokenized = [remove_stop_words(word_tokenize(sen)) for sen in all_sentences]
    title_words = remove_stop_words(word_tokenize(title))
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
    threshold = similarity_threshold
    denominator = float(len(set(sen1).union(sen2)))
    if denominator > 0:
        ratio = len(set(sen1).intersection(sen2)) / denominator
    else:
        ratio = 0
    return (ratio >= threshold, ratio)

def build_feature_set():
    datasetJson = read_file('resources/pasokh/all.json') 
    pasokh = json.loads(datasetJson)
    total_similar = 0
    p = 0
    id = 0
    output = {}
    output2 = {}
    for key in pasokh:
        feature_set = []
        doc = pasokh[key]
        text = doc["text"]
        title = doc["title"]
        all_words = word_tokenize(text)
        all_words = remove_stop_words(all_words)
        word_freq = FreqDist(all_words)
        #No matter who has summarized this text, we only used one summary
        key, summary = doc["summaries"].popitem()
        sentences = sent_tokenize(text)
        summ_sens = sent_tokenize(summary)
        total_similar = 0
        summ_sens_words = [remove_stop_words(word_tokenize(sen)) for sen in summ_sens]
        position = 0
        for sen in sentences:
            id += 1
            sen = normalizer.normalize(sen)
            words = remove_stop_words(word_tokenize(sen))
            if len(words) < 1: continue
            similar = False
            i = 0
            features = generate_features(words, sentences, word_freq, position, title)
            while not similar and i < len(summ_sens):
                sen2 = summ_sens_words[i]
                (similar, similarity) = are_similar(words, sen2)
                features['target'] = similar
                features['id'] = id
                output2[id] = " ".join(words)
                if similar: total_similar += 1
                i += 1
            feature_set.append((features, similar))
            position += 1
        output[key.replace(".", "")] = feature_set    
        
    return (output, output2)

'''normalizer = Normalizer()
normalizer.normalize('اصلاح نويسه ها و استفاده از نیم‌فاصله پردازش را آسان مي كند')
'اصلاح نویسه‌ها و استفاده از نیم‌فاصله پردازش را آسان می‌کند'

sent_tokenize('ما هم برای وصل کردن آمدیم! ولی برای پردازش، جدا بهتر نیست؟')
['ما هم برای وصل کردن آمدیم!', 'ولی برای پردازش، جدا بهتر نیست؟']
word_tokenize('ولی برای پردازش، جدا بهتر نیست؟')
['ولی', 'برای', 'پردازش', '،', 'جدا', 'بهتر', 'نیست', '؟']

stemmer = Stemmer()
stemmer.stem('کتاب‌ها')
'کتاب'
lemmatizer = Lemmatizer()
lemmatizer.lemmatize('می‌روم')
'رفت#رو'

tagger = POSTagger(model='resources/postagger.model')
tafs = tagger.tag(word_tokenize('ما بسیار کتاب می‌خوانیم'))
[('ما', 'PRO'), ('بسیار', 'ADV'), ('کتاب', 'N'), ('می‌خوانیم', 'V')]
print(tafs)
'''

from features import *


cue_words = read_file("resources/cue-words.txt").split()
stop_words = read_file("resources/stop-words.txt").split()
rouge = Rouge()



#farsnet = importEFromPaj("resources/farsnet/synset_related_to.paj")

tagger = POSTagger(model='resources/postagger.model')
normalizer = Normalizer()

(feats, refs) = build_feature_set()
f_file = open('features.json', '+w')
json.dump(feats, f_file, ensure_ascii=False)
f_file.close()

f_file = open('referense_sens.json', '+w')
json.dump(refs, f_file, ensure_ascii=False)
f_file.close()

cols = []
for key in feats:
    for(sen, target) in feats[key]:
        for attr in sen:
            cols.append(attr)
        break
    break
output = [','.join(cols) + " \r\n"]

for key in feats:
    for (sen, target) in feats[key]:
        row = []
        for attr in sen:
            row.append(str(sen[attr]))
        output.append(','.join(row) + "\r\n")
        '''str(sen['id']) + "," + str(sen['pos_nn_ratio']) + "," + str(sen['pos_ve_ratio']) + "," +\
        str(sen['pos_aj_ratio']) + "," + str(sen['pos_av_ratio']) + "," + str(sen['tfisf']) + "," + \
        str(sen['tf']) + "," + str(sen['cue_words']) + "," + str(sen['cosine_position']) + "," + \
        str(target)+ "\r\n" )'''

f_file = open('dataset.csv', '+w')
f_file.writelines(output)
f_file.close()