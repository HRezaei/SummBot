import json, random, re, hashlib
from hazm import *

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

category_map = {
    'PO': 1,
    'SO': 2,
    'SP': 3,
    'CU': 4,
    'EC': 5,
    'SC': 6
}

similarity_threshold = 0.45

def read_file(path):
    file = open(path, "r", encoding='utf8')
    content = file.read()
    return content


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


def load_dataset_splitted(path, split_size=0.25):
    dataset = json.loads(read_file(path))
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
    for document_key in dataset:
        rand = random.random()
        key = 'train'
        if rand < split_size:
            key = 'test'
        documents[key].append(document_key)
        for (sen, label) in dataset[document_key]:
            row = []
            for attr in learning_features:
                if attr == 'category':
                    row.append(category_map[sen[attr]])
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
            feature_matrix[:, col_index] = scaler.transform(col).reshape(1, -1)*10
normalize_dataset.scalers = None


def remove_stop_words(words):
    hash_key = hashlib.md5(str(words).encode('utf-8')).hexdigest()
    if hash_key in remove_stop_words.cache:
        return remove_stop_words.cache[hash_key]
    is_string = isinstance(words, str)
    if is_string:
        words = word_tokenize(words)
#This should be read once instead of every time this function is called
    removed = [word for word in words if word not in stop_words and re.sub("\s|\u200c", "", word).isalnum()]
    if is_string:
        return ' '.join(removed)
    return removed
remove_stop_words.cache = {}


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


def print_rouges(rouges):
    #print("Diff" + str(diff_summs))
    print("{:<8} {:<25} {:<25} {:<25}".format('Test,','f-measure,','precision,', 'recall'))
    for k in ['rouge-1', 'rouge-2', 'rouge-l']:
        v = rouges[k]
        print("{:<8}, {:<25}, {:<25}, {:<25}".format(k, v['f'], v['p'], v['r']))

tagger = POSTagger(model='resources/postagger.model')
stop_words = read_file("resources/stop-words.txt").split()
