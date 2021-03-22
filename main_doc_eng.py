from utilities import *
from doc_eng_competition.doc_eng import *
import time
import sys

def parse_dataset():
    '''
    Parses XML files of CNN corpus and writes them in a JSON file for later use
    :return:
    '''
    cnn_dataset = read_cnn_directory('/home/hrezaei/Documents/CNN_Corpus')
    json_write(cnn_dataset, 'resources/CNN/documents.json')


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


def used_features():
    features = [
        'cue_words',
        'tfisf',
        'cosine_position',
        'relative_len',
        # 'title_resemblance', 'keywords_resemblance',
        # 'tf',
        'pos_ve_ratio',
        'pos_aj_ratio',
        'pos_nn_ratio',
        'pos_av_ratio',
        'pos_num_ratio',
        #'len',
        #'position',
        #'doc_words', 'doc_sens', # 'doc_parag',
        #'category',
        # 'doc_verbs', 'doc_adjcs', 'doc_advbs', 'doc_nouns',
        #'nnf_isnnf',
        #'vef_isvef',
        # 'ajf_isajf',
         #'avf_isavf',
        #'nuf_isnuf',
        # 'id_ref'
        # 'category_world', 'category_us', 'category_sport', 'category_travel', 'category_showbiz',
        #      'category_business', 'category_health','category_justice','category_living','category_opinion',
        #      'category_politics', 'category_tech'
    ]
    #features = ['category', 'cue_words', 'tfisf', 'cosine_position', 'relative_len', 'nnf_isnnf', 'vef_isvef', 'ajf_isajf', 'avf_isavf', 'nuf_isnuf']
    #features = ['category', 'cue_words', 'tfisf', 'pos_ve_ratio', 'pos_aj_ratio', 'pos_nn_ratio', 'pos_av_ratio', 'pos_num_ratio', 'len', 'position']
    return features


def learn():
    from doc_eng_competition.learn import learn_models
    valid_features = used_features()
    models = ['svm', 'dtr', 'linear', 'nb']
    models = ['dummy']
    learn_models(models, valid_features)


def summarize_cnn_folder(input='/home/hrezaei/Documents/competition_folder', output=''):
    import _pickle as cpickle
    from doc_eng_competition.Summ import summ
    import doc_eng_competition.doc_eng as eng
    documents = read_cnn_directory(input)
    model = cpickle.load(open('svm.pkl', 'rb'))
    normalize_dataset.scalers = cpickle.load(open('scalers.pkl', 'rb'))
    features = used_features()
    for key in sorted(documents):
        doc = documents[key]
        summary = summ(doc, model, features)
        eng.write_xml(summary, key, doc['file_name'], output)
        print('Summarized: ' + doc['file_name'])


def store_cache():
    fp = open('resources/CNN/document_feature_set_cache.pkl', 'wb')
    import _pickle as cPickle
    cPickle.dump(document_feature_set.cache, fp)


def load_cache():
    fp = open('resources/CNN/document_feature_set_cache.pkl', 'rb')
    import _pickle as cPickle
    document_feature_set.cache = cPickle.load(fp)


def run_one_to_one_compare():
    from doc_eng_competition.learn import learn_models

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
    rouge_stats, model_stats = learn_models(models, features)
    reports = {
        'all unaware': rouge_stats
    }
    for key in map:
        print('swapped ' + key + ' with ' + map[key])
        features = [k if k != key else map[k] for k in map]
        print(features)
        normalize_dataset.scalers = None
        rouge_stats, model_stats = learn_models(models, features)
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