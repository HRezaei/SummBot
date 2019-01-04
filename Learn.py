from sklearn import tree
from utilities import *
from sklearn.model_selection import train_test_split
#import matplotlib.pyplot as plt
import numpy as np
import _pickle as cPickle
from Summ import summ
from rouge import Rouge
import json, sys
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix
from sklearn.svm import SVR

from sklearn.preprocessing import StandardScaler, MinMaxScaler



def visualize(y, predicted, title):
    plt.figure()
    plt.scatter(range(len(y)), y, s=10, edgecolor="darkorange",
                c="darkorange", label="train")
    plt.scatter(range(len(y)), predicted, color="cornflowerblue",
            label="test", linewidth=1, marker='+')
    #plt.plot(X_test, predicted, color="yellowgreen", label="predicted", linewidth=2)
    plt.xlabel("data")
    plt.ylabel("target")
    plt.title(title)
    plt.legend()
    plt.show()


def evaluate_summarizer(clf, dataset, remove_stopwords=False, normalizers=[]):

    rouge = Rouge()
    empty_score = {
        'rouge-1': {'p':0, 'f': 0, 'r': 0},
        'rouge-2': {'p':0, 'f': 0, 'r': 0},
        'rouge-l': {'p':0, 'f': 0, 'r': 0}
        }
    total_scores = {
        'rouge-1':{'p': 0, 'f': 0, 'r': 0},
        'rouge-2':{'p': 0, 'f': 0, 'r': 0},
        'rouge-l':{'p': 0, 'f': 0, 'r': 0}
        }
    avg_scores = empty_score
    total_summaries = 0
    for key in dataset:
        total_summaries += 1
        text = dataset[key]['text']
        gold_summaries = dataset[key]['summaries']
        best_score = empty_score
        for ref_key in gold_summaries:
            ref = gold_summaries[ref_key]
            ref_len = len(sent_tokenize(ref))
            if remove_stopwords:
                ref = remove_stop_words(ref)
            summary = summ(text, clf, key[4:6], normalizers, ref_len)
            if remove_stopwords:
                summary = remove_stop_words(summary)
            #if len(summary) != len(ref):
            #    diff_summs += 1
            scores = rouge.get_scores(ref, summary)[0]
            best_score = best_rouge_f(best_score, scores)
   
        for test_type in best_score:
            for param in best_score[test_type]:
                total_scores[test_type][param] += best_score[test_type][param]
    
    total_docs = len(dataset)
    for test_type in total_scores:
        for param in total_scores[test_type]:
            avg_scores[test_type][param] = total_scores[test_type][param]/total_summaries
    
    print("{:<8} {:<25} {:<25} {:<25}".format('Test','f-measure','precision', 'recall'))
    for k in ['rouge-1', 'rouge-2', 'rouge-l']:
        v = avg_scores[k]
        print("{:<8} {:<25} {:<25} {:<25}".format(k, v['f'], v['p'], v['r']))


def evaluate_model(model):
    y_pred = model.predict(X_test)

    y_pred_train = model.predict(X_balanced)

    # The mean squared error
    print("Mean squared error: %.5f"
          % mean_squared_error(y_test, y_pred))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.5f' % r2_score(y_test, y_pred))
    # visualize(y_test, y_pred, "Decision Tree Regression - Test set")

    # visualize(y_balanced, y_pred_train, "Decision Tree Regression - Train set")
    print("MSE on train: %.5f"
          % mean_squared_error(y_balanced, y_pred_train))
    print('Variance score on train: %.5f' % r2_score(y_balanced, y_pred_train))

    print("Confusion Matrix on test:")
    print_cm(confusion_matrix(labels_test, [True if y > similarity_threshold else False for y in y_pred]), labels=['T', 'F'])

    print("Confusion Matrix on train:")
    print_cm(confusion_matrix(labels_balanced, [True if y > similarity_threshold else False for y in y_pred_train]), labels=['T', 'F'])


def export_model(model, export_name):
    with open(export_name + '.pkl', 'wb') as fid:
        cPickle.dump(model, fid)


def best_rouge_f(score1, score2):
    sumF1 = score1["rouge-1"]["f"] + score1["rouge-2"]["f"] + score1["rouge-l"]["f"]
    sumF2 = score2["rouge-1"]["f"] + score2["rouge-2"]["f"] + score2["rouge-l"]["f"]
    if sumF2 > sumF1:
        return score2
    return score1


features, targets, labels = load_dataset('features.json')
# X_normal = StandardScaler().fit_transform(dataset[0])
X_normal = np.array(features)

normalizers = {
    'doc_words': MinMaxScaler(),
    'doc_nouns': MinMaxScaler(),
    'doc_verbs': MinMaxScaler(),
    'doc_adjcs': MinMaxScaler(),
    'doc_advbs': MinMaxScaler(),
    'doc_sens': MinMaxScaler(),
    'doc_parag': MinMaxScaler(),
    'tf': MinMaxScaler(),
    'category': MinMaxScaler(),
    'tfisf': MinMaxScaler(),
    'cue_words': MinMaxScaler()
}

normalize_column(X_normal, 'doc_words', normalizers, 'learn')
normalize_column(X_normal, 'doc_nouns', normalizers, 'learn')
normalize_column(X_normal, 'doc_verbs', normalizers, 'learn')
normalize_column(X_normal, 'doc_adjcs', normalizers, 'learn')
normalize_column(X_normal, 'doc_advbs', normalizers, 'learn')
normalize_column(X_normal, 'doc_sens', normalizers, 'learn')
normalize_column(X_normal, 'doc_parag', normalizers, 'learn')
normalize_column(X_normal, 'tf', normalizers, 'learn')
normalize_column(X_normal, 'category', normalizers, 'learn')
normalize_column(X_normal, 'tfisf', normalizers, 'learn')
normalize_column(X_normal, 'cue_words', normalizers, 'learn')


X_train, X_test, y_train, y_test, labels_train, labels_test = \
    train_test_split(X_normal, targets, labels,
                     test_size=0.25,
                     random_state=0)

print("Dataset size: {}".format(len(X_normal)))
print("Number of True/False labels: {}/{}".format(sum(labels), sum(1 for i in labels if not i)))
(X_balanced, y_balanced, labels_balanced) = (X_train, y_train, labels_train)
#X_balanced, y_balanced, labels_balanced = balance_dataset(X_train, y_train, labels_train, 3)
print("Train set size: {}".format(len(X_balanced)))
print("Number of True/False labels: {}/{}".format(sum(labels_balanced), sum(1 for i in labels_balanced if not i)))
print("Test set size: {}".format(len(X_test)))
print("Number of True/False labels: {}/{}".format(sum(labels_test), sum(1 for i in labels_test if not i)))
print("Used features: {}".format(len(X_balanced[0])))

dataset_json = json.loads(read_file('resources/pasokh/all.json'))
for model_type in ['ideal', 'dummy', 'linear', 'svm', 'dtr']:
    print('**********************' + model_type + '**********************')
    if model_type == 'dtr':
        # max_depth=6
        regr = tree.DecisionTreeRegressor(max_depth=6)
        regr = regr.fit(X_balanced, y_balanced)
        export_name = 'dtr'
    elif model_type == 'linear':
        regr = linear_model.LinearRegression()
        # Train the model using the training sets
        regr.fit(X_balanced, y_balanced)
        # The coefficients
        print('Coefficients: \n', regr.coef_)
        export_name = 'linear'
    elif model_type == 'svm':
        regr = SVR(verbose=True, epsilon=0.001, gamma='auto')
        # Train the model using the training sets
        regr.fit(X_balanced, y_balanced)
        # The coefficients
        print('Coefficients: \n', regr.get_params())
        export_name = 'svm'
    elif model_type == 'dummy':
        from DummyRegressor import RndRegressor
        regr = RndRegressor()
        export_name = 'dummy'
    elif model_type == 'ideal':
        from IdealRegressor import IdealRegressor
        regr = IdealRegressor(X_normal, targets)
        export_name = 'ideal'
    else:
        print("Regression type is undefined:" + model_type)
        continue
    # Make predictions using the testing set


    evaluate_model(regr)

    print('Summarizing dataset and evaluating Rouge...')
    evaluate_summarizer(regr, dataset_json, True, normalizers)
    print('*****************************************************************************')