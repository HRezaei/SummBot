from utilities import read_file, avg_bleu_score, average_similarity, similarity_threshold, encode_complex, write_dataset_csv
import json
import nltk
import hashlib
from nltk.corpus import stopwords

def build_feature_set():
    datasetJson = read_file('resources/DUC/all.json')
    duc = json.loads(datasetJson)
    output = {}
    for key in duc:
        doc = duc[key]
        text_sentences = doc["text"]
        #title = doc["title"]
        if len(doc['summaries']) < 1:
            print('No golden summary ' + key)
            continue
        feature_set, tmp = document_feature_set(text_sentences, key[4:6], doc['summaries'])
        output[key] = feature_set

    return output


def document_feature_set(text_sentences, category, golden_summaries=[], key=''):
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
    text = ' '.join(text_sentences)
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
    all_words = nltk.word_tokenize(text)
    all_words = remove_stop_words(all_words)
    #tags = nltk.pos_tag(words, tagset='universal')

    word_freq = nltk.FreqDist(all_words)

    for sen in text_sentences[:]:
        words = remove_stop_words(nltk.word_tokenize(sen))
        '''if len(words) < 1:
            text_sentences.remove(sen)
            continue'''
        sentence_words.append(words)
        tagged_sen = nltk.pos_tag(words, tagset='universal')
        num_nouns += sum(1 if tag == 'NOUN' else 0 for (w, tag) in tagged_sen)
        num_verbs += sum(1 if tag == 'V' else 0 for (w, tag) in tagged_sen)
        num_adjcs += sum(1 if tag == 'ADJ' or tag == 'AJe' else 0 for (w, tag) in tagged_sen)
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
            summary_sentences_indices = golden_summaries[summ]
            summ_sens = [text_sentences[i] for i in summary_sentences_indices]
            # word tokenized summaries for computing bleu scores:
            normalized_summaries.append(' '.join(summ_sens).split())
            gold_summaries[summ] = {'sens': [remove_stop_words(nltk.word_tokenize(sen)) for sen in summ_sens]}

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

    document_feature_set.cache[hash_key] = (feature_set, text_sentences)
    return feature_set, text_sentences


document_feature_set.id = 0
document_feature_set.cache = {}


def remove_stop_words(all_words):
    stop_words = set(stopwords.words('english'))
    return [w for w in all_words if w not in stop_words]


def add_features(features, sent, all_sentences_tokenized, word_freq, position):
    '''
    Args:
        sent: array of words
    '''
    import Features
    total_sentences = len(all_sentences_tokenized)
    features["tfisf"] = Features.tf_isf_score(sent, all_sentences_tokenized, word_freq)
    features["cosine_position"] = Features.cosine_position_score(position, total_sentences)
    features['position'] = 1/(position+1)
    features["tf"] = Features.frequency_score(sent, word_freq)
    features["cue_words"] = Features.cue_words(sent, cue_words)
    features['len'] = len(sent)
    avg_len = sum([len(s) for s in all_sentences_tokenized])/total_sentences
    features['relative_len'] = len(sent)/avg_len
    Features.pos_ratio_based_en(features, sent)
    return features


cue_words = read_file("resources/cue-words-en.txt").split()
feats = build_feature_set()
f_file = open('features_duc.json', '+w')
json.dump(feats, f_file, ensure_ascii=False, default=encode_complex)
f_file.close()
print("features_duc.json has been written successfully")

write_dataset_csv(feats, 'dataset_duc.csv')