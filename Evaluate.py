from sklearn.model_selection import cross_validate, ShuffleSplit
from utilities import *
import numpy as np
from sklearn import linear_model
from sklearn.svm import SVR
from Learn import normalize_dataset, evaluate_summarizer

def k_fold_evaluate(clf, samples, targets, labels, dataset, feature_names):
    num_folds = 4
    cv = ShuffleSplit(n_splits=num_folds, test_size=0.25, random_state=1)
    scoring = ['neg_mean_squared_error', 'r2']
    scores = cross_validate(clf, samples, targets, cv=cv, scoring=scoring, return_train_score=True,
                            return_estimator=True)
    mse = -1 * scores['test_neg_mean_squared_error']
    rouge_scores = {
        'rouge-1': {'p': [], 'f': [], 'r': []},
        'rouge-2': {'p': [], 'f': [], 'r': []},
        'rouge-l': {'p': [], 'f': [], 'r': []}
    }
    i = 0
    for fitted_clf in scores['estimator']:
        i += 1
        print("Summarizing dataset by model %d and evaluating ROUGE " % i)
        rouge_score = evaluate_summarizer(fitted_clf, dataset, feature_names, True)
        print_rouges(rouge_score)
        for test_type in rouge_scores:
            for param in rouge_scores[test_type]:
                rouge_scores[test_type][param].append(rouge_score[test_type][param])

    avg_scores = {
        'rouge-1':{'p': 0, 'f': 0, 'r': 0},
        'rouge-2':{'p': 0, 'f': 0, 'r': 0},
        'rouge-l':{'p': 0, 'f': 0, 'r': 0}
    }
    for test_type in rouge_scores:
        for param in rouge_scores[test_type]:
            avg_scores[test_type][param] = np.array(rouge_scores[test_type][param]).mean()
    result = {
        'mse': mse.mean(),
        'r2': scores['test_r2'].mean(),
        'rouge': avg_scores
    }
    return result
    #print("MSE: %0.5f (+/- %0.5f)" % (mse.mean(), mse.std() * 2))
    #print(scores)


def paper_evaluate():
    document_feature_names = ['doc_words', 'doc_sens', #'doc_parag', #'category',
                         # 'doc_verbs', 'doc_adjcs', 'doc_advbs', 'doc_nouns',
                         'nnf_isnnf', 'vef_isvef', 'ajf_isajf', 'avf_isavf', 'nuf_isnuf',
                          'political', 'social', 'sport', 'culture', 'economy', 'science'
                         ]
    sentence_feature_names = ['cosine_position', 'tfisf', 'relative_len', 'cue_words',
                         # 'tf',
                         ]

    document_unawares = ['pos_ve_ratio', 'pos_aj_ratio', 'pos_nn_ratio', 'pos_av_ratio',
                          'len', 'position'
                         ]
    all_feature_names = document_feature_names + sentence_feature_names + document_unawares
    all_features, targets, labels = load_dataset('features.json')

    all_features = np.array(all_features)
    normalize_dataset(all_features, all_feature_names, 'learn')

    experiment1_feature_names = ['cue_words'] + document_unawares
    experiment2_feature_names = sentence_feature_names + document_feature_names
    experiment3_feature_names = all_feature_names

    selected_feature_names = experiment2_feature_names
    selected_features = select_features(selected_feature_names, all_features)

    clf = SVR(verbose=False, epsilon=0.01, gamma='auto')
    documents = json.loads(read_file('resources/pasokh/all.json'))
    exp2_result = k_fold_evaluate(clf, selected_features, targets, labels, documents, selected_feature_names)

    print("MSE: %.5f"  % exp2_result['mse'])
    print('R2 score: %.5f' % exp2_result['r2'])

    #print("MSE on train: %.5f"  % mean_squared_error(y_balanced, y_pred_train))
    #print('Variance score on train: %.5f' % r2_score(y_balanced, y_pred_train))

    print_rouges(exp2_result['rouge'])


paper_evaluate()