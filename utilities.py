import json, random, re
from hazm import *

all_features = ['cosine_position', 'cue_words', 'tfisf', 'tf', 'pos_ve_ratio', 'pos_aj_ratio', 'pos_nn_ratio',
'pos_av_ratio', 'num_words', 'num_sens', 'included', 'target', 'target_bleu_avg', 'text', 'target_bleu', 'source_file', 'id', 'category' ]


valid_features = ['cosine_position', 'cue_words', 'tfisf',
                  # 'tf',
                  'pos_ve_ratio', 'pos_aj_ratio', 'pos_nn_ratio', 'pos_av_ratio',
     'num_words', 'num_sens', 'num_parag', 'category'
 ]

category_map = {
    'PO': 1,
    'SO': 2,
    'SP': 3,
    'CU': 4,
    'EC': 5,
    'SC': 6
}


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


def balance_dataset(X_train, y_train, ratio):
    num_true = 0
    num_false = 0
    false_indices = []
    balanced_x_train = []
    balanced_y_train = []
    for i in range(len(X_train)):
        if y_train[i]>0.5:
            num_true += 1
            balanced_x_train.append(X_train[i])
            balanced_y_train.append(y_train[i])
        else:
            num_false += 1
            false_indices.append(i)
    print("Number of positives/negatives: {}/{} ".format(num_true, num_false))
    selected_indices = random.sample(false_indices, int(num_true*ratio))
    print("After balancing, positives/negatives: {}/{} ".format(num_true, len(selected_indices)))
    for i in selected_indices:
        balanced_x_train.append(X_train[i])
        balanced_y_train.append(y_train[i])
    return (balanced_x_train, balanced_y_train)


def remove_stop_words(words):
    is_string = isinstance(words, str)
    if is_string:
        words = word_tokenize(words)
#This should be read once instead of every time this function is called
    removed = [word for word in words if word not in stop_words and re.sub("\s|\u200c", "", word).isalnum()]
    if is_string:
        return ' '.join(removed)
    return removed


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

tagger = POSTagger(model='resources/postagger.model')
stop_words = read_file("resources/stop-words.txt").split()
