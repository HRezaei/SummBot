import json, random, re, hashlib
from hazm import *

all_features = ['cosine_position', 'cue_words', 'tfisf', 'tf', 'pos_ve_ratio', 'pos_aj_ratio', 'pos_nn_ratio',
                'pos_av_ratio', 'doc_words', 'doc_sens', 'included', 'target', 'target_bleu_avg', 'text',
                'target_bleu', 'source_file', 'id', 'category', 'doc_verbs', 'doc_adjcs', 'doc_advbs', 'doc_nouns']


valid_features = ['cosine_position', 'cue_words', 'tfisf',
                   'tf',
                  'pos_ve_ratio', 'pos_aj_ratio', 'pos_nn_ratio', 'pos_av_ratio',
     'doc_words', 'doc_sens' , 'doc_parag', 'category',
                  'doc_verbs', 'doc_adjcs', 'doc_advbs', 'doc_nouns'
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
            for attr in valid_features:
                if attr == 'category':
                    row.append(category_map[sen[attr]])
                else:
                    row.append(sen[attr])
            features.append(row)
            target.append(sen['target'])
            labels.append(label)
    return features, target, labels


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


def normalize_column(dataset, column_name, normalizers, mode='utilization'):
    if column_name not in valid_features:
        return
    if column_name not in normalizers:
        raise Exception('No scaler is defined for:' + column_name)
    scaler = normalizers[column_name]
    col_index = valid_features.index(column_name)
    col = dataset[:, col_index].reshape(-1, 1)
    if mode == 'learn':
        scaler.fit(col)
    dataset[:, col_index] = scaler.transform(col).reshape(1, -1)


tagger = POSTagger(model='resources/postagger.model')
stop_words = read_file("resources/stop-words.txt").split()
