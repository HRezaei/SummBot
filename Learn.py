from sklearn import tree
import utilities
from sklearn.model_selection import train_test_split
#import matplotlib.pyplot as plt
import numpy as np
from Summ import summ
from rouge import Rouge
import json, sys
from sklearn import linear_model
from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, recall_score
from sklearn.svm import SVR
from sklearn.naive_bayes import *
import doc_per as farsi
import hazm


from sklearn.preprocessing import StandardScaler, MinMaxScaler



from DummyRegressor import RndRegressor

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


def evaluate_summarizer(clf, dataset, used_features, remove_stopwords=False):

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
    #diff_summs = 0
    for key in dataset:
        total_summaries += 1
        text = dataset[key]['text']
        gold_summaries = dataset[key]['summaries']
        best_score = empty_score
        for ref_key in gold_summaries:
            ref = gold_summaries[ref_key]
            ref_len = len(hazm.sent_tokenize(ref))
            if remove_stopwords:
                ref = farsi.remove_stop_words_and_puncs(ref)
            summary = summ(text, clf, key[4:6], used_features, ref_len)

            lines = [s + "\n\n" for s in summary]
            summary = " ".join(summary)
            if remove_stopwords:
                summary = farsi.remove_stop_words_and_puncs(summary)
            #if len(summary) != len(ref):
            #    diff_summs += 1
            if len(summary) == 0:
                continue
            try:
                scores = rouge.get_scores(ref, summary)[0]
            except:
                print(ref)
                print(summary)
                o = 1
                o += 1
            """f_file = open('/tmp/summaries/' + ref_key + str(scores["rouge-1"]["f"]) + '.txt', '+w')
            f_file.writelines(lines)
            f_file.close()"""
            best_score = best_rouge_f(best_score, scores)
   
        for test_type in best_score:
            for param in best_score[test_type]:
                total_scores[test_type][param] += best_score[test_type][param]
    
    total_docs = len(dataset)
    for test_type in total_scores:
        for param in total_scores[test_type]:
            avg_scores[test_type][param] = total_scores[test_type][param]/total_summaries
    return avg_scores


def evaluate_model(model, X_test, X_balanced, y_test, y_balanced, labels_test, labels_balanced, is_regressor=True):
    y_pred = model.predict(X_test)

    y_pred_train = model.predict(X_balanced)

    result = {
        'test_mse': mean_squared_error(y_test, y_pred),
        'test_r2': r2_score(y_test, y_pred),
        'train_mse': mean_squared_error(y_balanced, y_pred_train),
        'train_r2': r2_score(y_balanced, y_pred_train)
    }

    # The mean squared error
    print("Mean squared error: %.5f" % result['test_mse'])

    print('R2 score: %.5f' % result['test_r2'])
    # visualize(y_test, y_pred, "Decision Tree Regression - Test set")

    # visualize(y_balanced, y_pred_train, "Decision Tree Regression - Train set")
    print("MSE on train: %.5f"  % result['train_mse'])
    print('R2 score on train: %.5f' % result['train_r2'])

    if not is_regressor:
        predicted_labels = y_pred
        predicted_labels_train = y_pred_train
    else:
        predicted_labels = [True if y > utilities.similarity_threshold else False for y in y_pred]
        predicted_labels_train = [True if y > utilities.similarity_threshold else False for y in y_pred_train]

    print("Confusion Matrix on test:")
    utilities.print_cm(confusion_matrix(labels_test, predicted_labels), labels=['F', 'T'])

    print("Confusion Matrix on train:")
    utilities.print_cm(confusion_matrix(labels_balanced, predicted_labels_train), labels=['F', 'T'])

    return result


def best_rouge_f(score1, score2):
    sumF1 = score1["rouge-1"]["f"] + score1["rouge-2"]["f"] + score1["rouge-l"]["f"]
    sumF2 = score2["rouge-1"]["f"] + score2["rouge-2"]["f"] + score2["rouge-l"]["f"]
    if sumF2 > sumF1:
        return score2
    return score1


def run_experiments_without_cross_validation(model_names, features_to_use):
    dataset_features = utilities.load_features('pasokh')
    features, targets, labels, documents, all_vec = utilities.split_dataset(dataset_features, features_to_use, 0.40)

    X_normal = np.array(all_vec)

    utilities.normalize_dataset(X_normal, features_to_use, 'learn')

    X_train = np.array(features['train'])
    X_test = np.array(features['test'])
    y_train = np.array(targets['train'])
    y_test = np.array(targets['test'])
    labels_train = np.array(labels['train'])
    labels_test = np.array(labels['test'])

    print("Dataset size: {}".format(len(X_normal)))
    #print("Number of True/False labels: {}/{}".format(sum(labels), sum(1 for i in labels if not i)))
    (X_balanced, y_balanced, labels_balanced) = (X_train, y_train, labels_train)
    #X_balanced, y_balanced, labels_balanced = utilities.balance_dataset(X_train, y_train, labels_train, 3)
    print("Train set size: {}".format(len(X_balanced)))
    print("Number of True/False labels: {}/{}".format(sum(labels_balanced), sum(1 for i in labels_balanced if not i)))
    print("Test set size: {}".format(len(X_test)))
    print("Number of True/False labels: {}/{}".format(sum(labels_test), sum(1 for i in labels_test if not i)))
    print("Used features: {}".format(len(X_balanced[0])))

    dataset_json = json.loads(utilities.read_file('resources/pasokh/all.json'))
    is_regressor = True
    for model_type in model_names:
        print('**********************' + model_type + '**********************')
        if model_type == 'dtr':
            # max_depth=6
            regr = tree.DecisionTreeRegressor()
            regr = regr.fit(X_balanced, y_balanced)
            export_name = 'dtr'
        elif model_type == 'linear':
            regr = linear_model.LinearRegression(normalize=True)
            # Train the model using the training sets
            regr.fit(X_balanced, y_balanced)
            # The coefficients
            print('Coefficients: \n', regr.coef_)
            export_name = 'linear'
        elif model_type == 'svm':
            regr = SVR(verbose=True, epsilon=0.00001, gamma='auto', tol=.00001)
            # Train the model using the training sets
            regr.fit(X_balanced, y_balanced)
            # The coefficients
            print('Coefficients: \n', regr.get_params())
            export_name = 'svm'
        elif model_type == 'dummy':
            regr = RndRegressor()
            export_name = 'dummy'
        elif model_type == 'ideal':
            from IdealRegressor import IdealRegressor
            regr = IdealRegressor(X_train, y_train)
            regr.fit(X_test, y_test)
            export_name = 'ideal'
        elif model_type == 'nb':
            # from sklearn import svm
            # regr = svm.SVC(gamma='scale').fit(X_train, labels_train)
            from sklearn.naive_bayes import ComplementNB, GaussianNB
            from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
            from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

            regr = ComplementNB(alpha=1)
            regr.fit(X_train, labels_train)
            is_regressor = False
            export_name = 'nb'
        else:
            print("Regression type is undefined:" + model_type)
            continue
        # Make predictions using the testing set

        model_results = evaluate_model(regr, X_test, X_balanced, y_test, y_balanced, labels_test, labels_balanced, is_regressor)

        print('Summarizing dataset and evaluating Rouge...')
        rouge_scores = evaluate_summarizer(regr, dataset_json, features_to_use, True)
        utilities.print_rouges(rouge_scores)
        print('*****************************************************************************')
    return rouge_scores, model_results


def run_experiments(model_names):
    """
    This version splits original texts in dataset for evaluating summaries
    """
    valid_features = ['cue_words', 'tfisf',
                      'cosine_position', 'relative_len',
                      # 'tf',
                      #'pos_ve_ratio', 'pos_aj_ratio', 'pos_nn_ratio', 'pos_av_ratio', 'pos_num_ratio', 'len', 'position'
                      'doc_words', 'doc_sens',# 'doc_parag', # 'category',
                      #'doc_verbs', 'doc_adjcs', 'doc_advbs', 'doc_nouns',
                      'nnf_isnnf', 'vef_isvef', 'ajf_isajf', 'avf_isavf', 'nuf_isnuf',
                      'political', 'social', 'sport', 'culture', 'economy', 'science'
                      ]
    features, targets, labels, documents, all_vec = load_dataset_splitted('features.json', learning_features)

    X_normal = np.array(all_vec)
    X_normal = select_features(valid_features, X_normal)
    # X_normal = StandardScaler().fit_transform(dataset[0])

    normalize_dataset(X_normal, valid_features, 'learn')

    X_train = np.array(features['train'])
    X_test = np.array(features['test'])
    y_train = np.array(targets['train'])
    y_test = np.array(targets['test'])
    labels_train = np.array(labels['train'])
    labels_test = np.array(labels['test'])

    X_train = select_features(valid_features, X_train)
    normalize_dataset(X_train, valid_features)

    X_test = select_features(valid_features, X_test)
    normalize_dataset(X_test, valid_features)

    print("Dataset size: {}".format(len(all_vec)))

    (X_balanced, y_balanced, labels_balanced) = (X_train, y_train, labels_train)
    #X_balanced, y_balanced, labels_balanced = balance_dataset(X_train, y_train, labels_train, 3)
    print("Train set size: {}".format(len(X_balanced)))
    print("Number of True/False labels: {}/{}".format(sum(labels_balanced), sum(1 for i in labels_balanced if not i)))
    print("Test set size: {}".format(len(X_test)))
    print("Number of True/False labels: {}/{}".format(sum(labels_test), sum(1 for i in labels_test if not i)))
    print("Used features: {}".format(len(X_balanced[0])))

    dataset_json = json.loads(read_file('resources/pasokh/all.json'))
    test_documents = {key: dataset_json[key] for key in documents['test']+documents['train']}
    is_regressor = True
    for model_type in model_names:
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
            regr = SVR(verbose=True, epsilon=0.001, gamma='auto', tol=.00001)
            # Train the model using the training sets
            regr.fit(X_balanced, y_balanced)
            # The coefficients
            print('Coefficients: \n', regr.get_params())
            export_name = 'svm'
        elif model_type == 'dummy':
            regr = RndRegressor()
            export_name = 'dummy'
        elif model_type == 'ideal':
            from IdealRegressor import IdealRegressor
            regr = IdealRegressor(X_normal, targets)
            export_name = 'ideal'
        elif model_type == 'nb':
            # from sklearn import svm
            # regr = svm.SVC(gamma='scale').fit(X_train, labels_train)
            from sklearn.naive_bayes import ComplementNB, GaussianNB
            from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
            from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

            regr = ComplementNB(alpha=0.015)
            regr.fit(X_train, labels_train)
            is_regressor = False
            export_name = 'nb'
        else:
            print("Regression type is undefined:" + model_type)
            continue
        # Make predictions using the testing set

        model_results = evaluate_model(regr, X_test, X_balanced, y_test, y_balanced, labels_test, labels_balanced, is_regressor)

        print('Summarizing dataset and evaluating Rouge...')

        rouge_scores = evaluate_summarizer(regr, test_documents, valid_features)
        utilities.print_rouges(rouge_scores)
        print('*****************************************************************************')
    return rouge_scores, model_results

if len(sys.argv)>1 and sys.argv[1] == 'debug':
    sys.setrecursionlimit(2000)
    models = ['ideal', 'dummy', 'linear', 'svm', 'dtr']
    rouge_scores = {
        'rouge-1': {'p': [], 'f': [], 'r': []},
        'rouge-2': {'p': [], 'f': [], 'r': []},
        'rouge-l': {'p': [], 'f': [], 'r': []}
    }
    model_results = {
        'test_mse': [],
        'test_r2': [],
        'train_mse': [],
        'train_r2': []
    }
    for i in range(0, 5):
        rouge_score, model_result = run_experiments(['svm'])
        normalize_dataset.scalers = None
        for test_type in rouge_scores:
            for param in rouge_scores[test_type]:
                rouge_scores[test_type][param].append(rouge_score[test_type][param])
        for metric in model_results:
            model_results[metric].append(model_result[metric])

    avg_rouges = {
        'rouge-1': {'p': 0, 'f': 0, 'r': 0},
        'rouge-2': {'p': 0, 'f': 0, 'r': 0},
        'rouge-l': {'p': 0, 'f': 0, 'r': 0}
    }
    for test_type in rouge_scores:
        for param in rouge_scores[test_type]:
            avg_rouges[test_type][param] = np.array(rouge_scores[test_type][param]).mean()

    avg_results = {
        'test_mse': [],
        'test_r2': [],
        'train_mse': [],
        'train_r2': []
    }
    for metric in model_results:
        avg_results[metric] = np.array(model_results[metric]).mean()

    print(avg_results)
    print_rouges(avg_rouges)