import json, random, re, hashlib
#from hazm import *
import nltk, math, operator
from nltk import bleu
from fractions import Fraction

all_features = ['position', 'cosine_position', 'cue_words', 'tfisf', 'tf', 'len', 'relative_len', 'num_count',
                'pos_ve_ratio', 'pos_aj_ratio', 'pos_nn_ratio', 'pos_av_ratio', 'pos_num_ratio', 'doc_words', 'doc_sens', 'doc_parag',
                'included', 'target', 'target_bleu_avg', 'text', 'target_bleu', 'source_file', 'id', 'category',
                'doc_verbs', 'doc_adjcs', 'doc_advbs', 'doc_nouns', 'doc_nums', 'nnf_isnnf', 'vef_isvef', 'ajf_isajf',
                'avf_isavf', 'nuf_isnuf', 'political', 'social', 'sport', 'culture', 'economy', 'science'
]

learning_features = ['position', 'cosine_position', 'cue_words', 'tfisf', 'tf', 'len', 'relative_len', 'num_count',
                     'pos_ve_ratio', 'pos_aj_ratio', 'pos_nn_ratio', 'pos_av_ratio', 'pos_num_ratio',
                     'doc_words', 'doc_sens', 'doc_parag', 'category', 'doc_verbs', 'doc_adjcs', 'doc_advbs',
                     'doc_nouns', 'doc_nums', 'nnf_isnnf', 'vef_isvef', 'ajf_isajf', 'avf_isavf', 'nuf_isnuf',
                     'political', 'social', 'sport', 'culture', 'economy', 'science'
]


def dynamic_mapping(category, namespace=''):
    '''
    Takes a category and a namespace and assigns a unique integer ID to
    that category within given namespace. Used for transforming nominal
    features to numerical ones
    :param category:
    :param namespace:
    :return:
    '''
    maps = dynamic_mapping.maps

    if namespace not in maps:
        maps[namespace] = {}

    map = maps[namespace]
    if category in map:
        return map[category]
    next_id = len(map) + 1
    map[category] = next_id
    return next_id

dynamic_mapping.maps = {}


def pasokh_category_mapping(category):
    category_map = {
        'PO': 1,
        'SO': 2,
        'SP': 3,
        'CU': 4,
        'EC': 5,
        'SC': 6
    }
    return category_map[category]


def cnn_category_mapping(category):
    category_map = {
        'living': 1,
        'politics': 2,
        'world': 3,
        'business': 4,
        'health': 5,
        'showbiz': 6,
        'us': 7,
        'justice': 8,
        'opinion': 9,
        'travel': 10,
        'tech': 11,
        'sport': 12,
    }

    return category_map[category]


similarity_threshold = 0.45


def read_file(path):
    file = open(path, "r", encoding='utf8')
    content = file.read()
    return content


def write_file(path, data):
    file = open(path, "w+")
    output = file.write(data)
    file.close()
    return output


def load_dataset(path):
    dataset = json.loads(read_file(path))
    features = []
    target = []
    labels = []
    for key in dataset:
        for (sen, label) in dataset[key]:
            row = []
            for attr in learning_features:
                if attr == 'category':
                    row.append(category_map[sen[attr]])
                else:
                    row.append(sen[attr])
            features.append(row)
            target.append(sen['target'])
            labels.append(label)
    return features, target, labels


def split_dataset(dataset, features_used, split_size=0.25, dataset_name='pasokh'):
    #dataset = json.loads(read_file(path))

    '''
    import main_doc_eng
    dataset = main_doc_eng.build_feature_set()
    '''
    all_vectors = []
    features = {
        'train': [],
        'test': []
    }
    target = {
        'train': [],
        'test': []
    }
    labels = {
        'train': [],
        'test': []
    }
    documents = {
        'train': [],
        'test': []
    }
    i = 0
    for document_key in dataset:
        #if i > 1000:
        #    break
        i += 1
        rand = random.random()
        key = 'train'
        if rand < split_size:
            key = 'test'
        documents[key].append(document_key)
        for (sen, label) in dataset[document_key]:
            row = []
            for attr in features_used:
                if attr == 'category':
                    if dataset_name == 'pasokh':
                        row.append(pasokh_category_mapping(sen[attr]))
                    else:
                        row.append(cnn_category_mapping(sen[attr]))
                else:
                    row.append(sen[attr])
            features[key].append(row)
            all_vectors.append(row)
            target[key].append(sen['target'])
            labels[key].append(label)
    return features, target, labels, documents, all_vectors


def select_features(feature_names, matrix):
    feature_indexes = []
    for name in feature_names:
        if name in learning_features:
            col_index = learning_features.index(name)
            feature_indexes.append(col_index)
    features = matrix[:, feature_indexes]
    return features


def balance_dataset(feature_vectors, targets, labels, ratio):
    num_true = 0
    num_false = 0
    false_indices = []
    balanced_x_train = []
    balanced_y_train = []
    balanced_labels = []
    for i in range(len(feature_vectors)):
        if targets[i] > similarity_threshold:
            num_true += 1
            balanced_x_train.append(feature_vectors[i])
            balanced_y_train.append(targets[i])
            balanced_labels.append(labels[i])
        else:
            num_false += 1
            false_indices.append(i)
    print("Number of positives/negatives: {}/{} ".format(num_true, num_false))
    selected_indices = random.sample(false_indices, int(num_true*ratio))
    print("After balancing, positives/negatives: {}/{} ".format(num_true, len(selected_indices)))
    for i in selected_indices:
        balanced_x_train.append(feature_vectors[i])
        balanced_y_train.append(targets[i])
        balanced_labels.append(labels[i])
    return balanced_x_train, balanced_y_train, balanced_labels


def normalize_dataset(feature_matrix, feature_names, mode='utilization'):
    if normalize_dataset.scalers is None:
        from sklearn.preprocessing import MinMaxScaler
        scalers = {
            'doc_words': MinMaxScaler(),
            'doc_nouns': MinMaxScaler(),
            'doc_verbs': MinMaxScaler(),
            'doc_adjcs': MinMaxScaler(),
            'doc_advbs': MinMaxScaler(),
            'doc_sens': MinMaxScaler(),
            'doc_parag': MinMaxScaler(),
            'tf': MinMaxScaler(),
            #'position': MinMaxScaler(),
            'tfisf': MinMaxScaler(),
            'cue_words': MinMaxScaler(),
            'len': MinMaxScaler()
        }
        normalize_dataset.scalers = scalers
    elif mode == 'learn':
        raise Exception('normalize_dataset must not be called again in learn mode')
    else:
        scalers = normalize_dataset.scalers

    for feature_name in scalers:
        if feature_name in feature_names:
            scaler = scalers[feature_name]
            col_index = feature_names.index(feature_name)
            col = feature_matrix[:, col_index].reshape(-1, 1)
            if mode == 'learn':
                scaler.fit(col)
            feature_matrix[:, col_index] = scaler.transform(col).reshape(1, -1)
normalize_dataset.scalers = None


def are_similar_rouge(sen1, sen2):
    scores = rouge.get_scores(sen1, sen2)
    return (scores[0]['rouge-2']['f'] >= 0.7)


def are_similar(sen1, sen2):
    denominator = float(len(set(sen1).union(sen2)))
    if denominator > 0:
        ratio = len(set(sen1).intersection(sen2)) / denominator
    else:
        ratio = 0
    return ratio >= similarity_threshold, ratio


def average_similarity(sen, gold_summaries):
    total_similarity = 0
    for sentence_list in gold_summaries:
        max_similarity = 0
        for sum_sen in sentence_list:
            (similar, similarity) = are_similar(sen, sum_sen)
            if similarity > max_similarity:
                max_similarity = similarity
        total_similarity += max_similarity
    return total_similarity / len(gold_summaries)


def avg_bleu_score(sen, summaries, avg=False):
    min_length = 5
    if avg:
        from nltk.translate.bleu_score import SmoothingFunction
        chencherry = SmoothingFunction()
        total = 0
        for summ in summaries:
            total += bleu([summ], sen, smoothing_function=chencherry.method2)
        score = total / len(summaries)
    else:
#        score = bleu(summaries, sen, smoothing_function=chencherry.method2)
        score = nltk.translate.bleu_score.modified_precision(summaries, sen, 2)
        if len(sen) < min_length:
            import numpy as np
            score *= np.exp(1-(min_length/len(sen)))
    return score


def encode_complex(obj):
    if isinstance(obj, Fraction):
        return obj.numerator/ obj.denominator
    raise TypeError(repr(obj) + " is not JSON serializable")


def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrixes
    Thanks to https://gist.github.com/zachguo/10296432
    """
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth
    # Print header
    print("    " + empty_cell, end=" ")
    for label in labels:
        print("%{0}s".format(columnwidth) % label, end=" ")
    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label1, end=" ")
        for j in range(len(labels)):
            cell = "%{0}.0f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end=" ")
        print()

    tp = cm[1, 1]
    tn = cm[0, 0]
    tn_percent = 100 * tn / sum(cm[0])
    tp_percent = 100 * tp / sum(cm[1])
    fp = cm[0, 1]
    fn = cm[1, 0]
    total = sum(cm[0]) + sum(cm[1])
    accuracy = 100 * (tp + tn) / total
    precision = 100 * tp / (tp + fp)
    recall = 100 * tp / (tp + fn)
    print('True Positives: {:.2f}%'.format(tp_percent))
    print('True Negatives: {:.2f}%'.format(tn_percent))
    print('Accuracy: {:.2f}%'.format(accuracy))
    print('Precision: {:.2f}%'.format(precision))
    print('Recall: {:.2f}%'.format(recall))


def print_rouges(rouges):
    #print("Diff" + str(diff_summs))
    print("{:<8} {:<25} {:<25} {:<25}".format('Test,','f-measure,','precision,', 'recall'))
    for k in ['rouge-1', 'rouge-2', 'rouge-l']:
        v = rouges[k]
        print("{:<8}, {:<25}, {:<25}, {:<25}".format(k, v['f'], v['p'], v['r']))


def write_dataset_csv(feats, path):
    first_row = feats[list(feats.keys())[0]][0][0] #get one of values
    all_columns = sorted(first_row.keys())
    output = [','.join(all_columns) + " \r\n"]

    for key in feats:
        for (sen, target) in feats[key]:
            row = []
            for attr in all_columns:
                row.append(str(sen[attr]))
            output.append(','.join(row) + "\r\n")
            '''str(sen['id']) + "," + str(sen['pos_nn_ratio']) + "," + str(sen['pos_ve_ratio']) + "," +\
            str(sen['pos_aj_ratio']) + "," + str(sen['pos_av_ratio']) + "," + str(sen['tfisf']) + "," + \
            str(sen['tf']) + "," + str(sen['cue_words']) + "," + str(sen['cosine_position']) + "," + \
            str(target)+ "\r\n" )'''

    f_file = open(path, '+w')
    f_file.writelines(output)
    f_file.close()
    print(path + " has been written successfully")


def cue_words(language):
    if language in cue_words.static:
        return cue_words.static[language]
    lang_map = {
        'en': "resources/cue-words-en.txt",
        'fa': "resources/cue-words.txt"
    }

    cue_words.static[language] = read_file(lang_map[language]).split()
    return cue_words.static[language]


cue_words.static = {}


def stop_words(language):
    if language in stop_words.static:
        return stop_words.static[language]
    if language == 'en':
        stop_words_list = set(nltk.corpus.stopwords.words('english'))
    elif language == 'fa':
        stop_words_list = read_file("resources/stop-words.txt").split()

    stop_words.static[language] = stop_words_list
    return stop_words_list


stop_words.static = {}


def json_write(data, path):
    import json
    file = open(path, "w+")
    json.dump(data, file, ensure_ascii=False)
    file.close()
    return True


def json_read(path):
    import json
    file = open(path, "r")
    data = json.load(file)
    file.close()
    return data


def export_model(model, export_name):
    import _pickle as cPickle
    with open('models/' + export_name + '.pkl', 'wb') as fid:
        cPickle.dump(model, fid)
    cPickle.dump(normalize_dataset.scalers, open('data/scalers.pkl', 'wb'))


def english_stemmer():
    if english_stemmer.cache:
        return english_stemmer.cache

    from nltk.stem.snowball import SnowballStemmer
    english_stemmer.cache = SnowballStemmer("english")
    return english_stemmer.cache


english_stemmer.cache = None


def cnn_html_escape(text):
    html_escape_table = {
        "&": "&amp;",
        '"': "&quot;",
        "'": "&apost;",
        ">": "&gt;",
        "<": "&lt;",
    }
    return "".join(html_escape_table.get(c,c) for c in text)


def load_features(dataset):
    if load_features.cache:
        return load_features.cache
    import _pickle as cPickle
    load_features.cache = cPickle.load(open('resources/' + dataset + '/features.pkl', 'rb'))
    return load_features.cache


load_features.cache = None


def draw_bar_chart(data, y_label, title):
    import matplotlib.pyplot as plt;
    plt.rcdefaults()
    import numpy as np
    import matplotlib.pyplot as plt

    objects = data.keys()
    y_pos = np.arange(len(objects))
    performance = data.values()

    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel(y_label)
    plt.title(title)

    plt.show()