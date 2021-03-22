
import hazm
import hashlib
from utilities import *
import time


normalizer = hazm.Normalizer()
stemmer = hazm.Stemmer()
tagger = hazm.POSTagger(model='resources/postagger.model')


def remove_stop_words_and_puncs(words):
    hash_key = hashlib.md5(str(words).encode('utf-8')).hexdigest()
    if hash_key in remove_stop_words_and_puncs.cache:
        return remove_stop_words_and_puncs.cache[hash_key]
    is_string = isinstance(words, str)
    if is_string:
            words = hazm.word_tokenize(words)

    stop_words_list = stop_words('fa')
    removed = [word for word in words if word not in stop_words_list and re.sub("\s|\u200c", "", word).isalnum()]
    if is_string:
        removed = ' '.join(removed)

    remove_stop_words_and_puncs.cache[hash_key] = removed
    return removed


remove_stop_words_and_puncs.cache = {}


def pos_ratio_based(features, sentence_words):
    tags = tagger.tag(sentence_words)
    all_count = len(sentence_words)
    nn_count = sum(1 if tag=='N' else 0 for (w, tag) in tags)
    ve_count = sum(1 if tag=='V' else 0 for (w, tag) in tags)
    aj_count = sum(1 if tag=='AJ' or tag=='AJe' else 0 for (w, tag) in tags)
    av_count = sum(1 if tag=='ADV' else 0 for (w, tag) in tags)
    num_count = sum(1 if tag == 'NUM' else 0 for (w, tag) in tags)
    features['num_count'] = num_count
    features['pos_nn_ratio'] = nn_count/all_count
    features['pos_ve_ratio'] = ve_count/all_count
    features['pos_aj_ratio'] = aj_count/all_count
    features['pos_av_ratio'] = av_count/all_count
    features['pos_num_ratio'] = num_count/all_count
    features['nnf_isnnf'] = (nn_count/features['doc_nouns']) if features['doc_nouns'] > 0 else 0
    features['vef_isvef'] = (ve_count/features['doc_verbs']) if features['doc_verbs'] > 0 else 0
    features['ajf_isajf'] = (aj_count/features['doc_adjcs']) if features['doc_adjcs'] > 0 else 0
    features['avf_isavf'] = (av_count/features['doc_advbs']) if features['doc_advbs'] > 0 else 0
    features['nuf_isnuf'] = (num_count / features['doc_nums']) if features['doc_nums'] > 0 else 0

def extract_features():
    start = time.time()
    data = build_feature_set()
    end = time.time()
    print(end - start)
    write_dataset_csv(data, 'resources/CNN/features.csv')
    json_write(data, 'resources/CNN/features.json')
    import _pickle as cPickle
    with open('resources/CNN/features.pkl', 'wb') as fid:
        cPickle.dump(data, fid)


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

    all_words = hazm.word_tokenize(text)
    all_words = remove_stop_words_and_puncs(all_words)
    word_freq = nltk.FreqDist(all_words)

    sentences = hazm.sent_tokenize(text)

    for sen in sentences[:]:
        normal_sen = normalizer.normalize(sen)
        words = remove_stop_words_and_puncs(hazm.word_tokenize(normal_sen))
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
            summ_sens = hazm.sent_tokenize(summary)
            words_stemmed = [[stemmer.stem(w) for w in ref_sen] for ref_sen in summ_sens]
            gold_summaries[summ] = {
                'sens': [remove_stop_words_and_puncs(hazm.word_tokenize(sen)) for sen in summ_sens],
                'sens_stemmed': words_stemmed
            }

    position = 0
    for sen in sentence_words:
        document_feature_set.id += 1
        words = sentence_words[position]
        features = doc_features.copy()
        add_features(features, words, sentence_words, word_freq, position)
        features['id'] = document_feature_set.id

        if golden_summaries:
            features['target_bleu'] = str(avg_bleu_score(sen, normalized_summaries))
            features['target_bleu_avg'] = avg_bleu_score(sen, normalized_summaries, True)
            gold_sentences = [gold_summaries[k]['sens'] for k in gold_summaries]
            features['target'] = average_similarity(words, gold_sentences)
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
    features["cue_words"] = Features.cue_words(sent, cue_words('fa'))
    features['len'] = len(sent)
    avg_len = sum([len(s) for s in all_sentences_tokenized])/total_sentences
    features['relative_len'] = len(sent)/avg_len
    pos_ratio_based(features, sent)
    return features


def extract_features():
    start = time.time()
    data = build_feature_set()
    end = time.time()
    print(end - start)
    write_dataset_csv(data, 'resources/pasokh/features.csv')
    json_write(data, 'resources/pasokh/features.json')
    import _pickle as cPickle
    with open('resources/pasokh/features.pkl', 'wb') as fid:
        cPickle.dump(data, fid)


def used_features():
    features = ['cue_words',
                      'cosine_position',
         'tfisf',
         'relative_len',
                      # 'tf',
        # 'pos_ve_ratio',
        # 'pos_aj_ratio',
        # 'pos_nn_ratio',
        # 'pos_av_ratio',
        # 'pos_num_ratio',
        #   'len',
        #   'position',
          #'doc_words', 'doc_sens',# 'doc_parag',
        'category',
          # 'doc_verbs', 'doc_adjcs', 'doc_advbs', 'doc_nouns',
        'nnf_isnnf',
        'vef_isvef',
        'ajf_isajf',
        'avf_isavf',
        'nuf_isnuf',
                      #'political', 'social', 'sport', 'culture', 'economy', 'science'
                      ]
    return features


def learn():
    import sys
    from Learn import run_experiments_without_cross_validation
    valid_features = used_features()
    sys.setrecursionlimit(2000)
    models = ['svm', 'dtr', 'linear', 'nb']
    #models = ['ideal', 'dummy']
    #models = ['nb']
    run_experiments_without_cross_validation(models, valid_features)


def run_one_to_one_compare():
    import sys
    from Learn import run_experiments_without_cross_validation
    sys.setrecursionlimit(2000)

    map = {
        'pos_ve_ratio': 'vef_isvef',
        'pos_aj_ratio': 'ajf_isajf',
        'pos_nn_ratio': 'nnf_isnnf',
        'pos_av_ratio': 'avf_isavf',
        'pos_num_ratio': 'nuf_isnuf',
        'len': 'relative_len',
        'position': 'cosine_position',
    }

    features = list(map.keys())
    models = ['nb']
    rouge_stats, model_stats = run_experiments_without_cross_validation(models, features)
    reports = {
        'all unaware': rouge_stats
    }
    for key in map:
        print('swapped ' + key + ' with ' + map[key])
        features = [k if k != key else map[k] for k in map]
        print(features)
        normalize_dataset.scalers = None
        rouge_stats, model_stats = run_experiments_without_cross_validation(models, features)
        reports[map[key]] = rouge_stats
    #draw_bar_chart(reports, 'F-measure', 'Effect of document aware features')
    for measure in ['f', 'p', 'r']:
        print()
        print()
        print("{:<25}, {:<25}, {:<25}, {:<25}".format(measure, 'rouge-1', 'rouge-2', 'rouge-l'))
        for test_name in reports:
            v = reports[test_name]
            print("{:<25}, {:<25}, {:<25}, {:<25}".format(test_name, v['rouge-1'][measure], v['rouge-2'][measure],
                                                         v['rouge-l'][measure]))