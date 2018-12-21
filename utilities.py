import json, random, nltk
from hazm import *

all_features = ['cosine_position', 'cue_words', 'tfisf','tf', 'pos_ve_ratio', 'pos_aj_ratio', 'pos_nn_ratio',
'pos_av_ratio', 'num_words', 'num_sens', 'included','target','target_bleu_avg','text','target_bleu','source_file','id', 'category' ]


valid_features = ['cosine_position', 'cue_words', 'tfisf','tf', 'pos_ve_ratio', 'pos_aj_ratio', 'pos_nn_ratio',
'pos_av_ratio'
   # , 'num_words', 'num_sens', 'num_parag', 'category'
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
    for key in dataset:
        for (sen, nomatter) in dataset[key]:
            row = []
            for attr in valid_features:
                if attr == 'category':
                    row.append(category_map[sen[attr]])
                else:
                    row.append(sen[attr])
            features.append(row)
            target.append(sen['target'])
    return (features, target)

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

tagger = POSTagger(model='resources/postagger.model')
