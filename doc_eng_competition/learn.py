import json
from sklearn import tree
from sklearn import linear_model
from sklearn.svm import SVR
import numpy as np
import utilities
import Learn
import doc_eng_competition.doc_eng as eng
from DummyRegressor import RndRegressor
from rouge import Rouge
from doc_eng_competition.Summ import summ
import _pickle as cPickle


def learn_models(model_names, features_to_use):
    """
    This version splits original texts in dataset for evaluating summaries
    """
    dataset_features = utilities.load_features('CNN')
    features, targets, labels, documents, all_vec = utilities.split_dataset(dataset_features, features_to_use, 0.28, 'CNN')
    #return utilities.write_dataset_csv(dataset_features, '/tmp/test.csv')
    '''
    cPickle.dump(features, open('features.pkl', 'wb'))
    cPickle.dump(targets, open('targets.pkl', 'wb'))
    cPickle.dump(labels, open('labels.pkl', 'wb'))
    cPickle.dump(documents, open('documents.pkl', 'wb'))
    cPickle.dump(all_vec, open('all_vec.pkl', 'wb'))

    
    features = cPickle.load(open('features.pkl', 'rb'))
    targets = cPickle.load(open('targets.pkl', 'rb'))
    labels = cPickle.load(open('labels.pkl', 'rb'))
    documents = cPickle.load(open('documents.pkl', 'rb'))
    all_vec = cPickle.load(open('all_vec.pkl', 'rb'))
    '''

    X_normal = np.array(all_vec)
    #X_normal = utilities.select_features(features_to_use, X_normal)
    # X_normal = StandardScaler().fit_transform(dataset[0])

    utilities.normalize_dataset(X_normal, features_to_use, 'learn')

    X_train = np.array(features['train'])
    X_test = np.array(features['test'])
    y_train = np.array(targets['train'])
    y_test = np.array(targets['test'])
    labels_train = np.array(labels['train'])
    labels_test = np.array(labels['test'])

    #X_train = utilities.select_features(features_to_use, X_train)
    #utilities.normalize_dataset(X_train, features_to_use)

    #X_test = utilities.select_features(features_to_use, X_test)
    #utilities.normalize_dataset(X_test, features_to_use)

    print("Dataset size: {}".format(len(all_vec)))

    #(X_balanced, y_balanced, labels_balanced) = (X_train, y_train, labels_train)
    X_balanced, y_balanced, labels_balanced = utilities.balance_dataset(X_train, y_train, labels_train, 1)
    print("Used features: " + ','.join(features_to_use))
    print("Train set size: {}".format(len(X_balanced)))
    print("Number of True/False labels: {}/{}".format(sum(labels_balanced), sum(1 for i in labels_balanced if not i)))
    print("Test set size: {}".format(len(X_test)))
    print("Number of True/False labels: {}/{}".format(sum(labels_test), sum(1 for i in labels_test if not i)))
    print("Used features: {}".format(len(X_balanced[0])))

    dataset_json = json.loads(utilities.read_file('resources/CNN/documents.json'))
    test_documents = {int(key): dataset_json[key] for key in documents['test']}
    is_regressor = True
    for model_type in model_names:
        print('**********************' + model_type + '**********************')
        if model_type == 'dtr':
            # max_depth=6
            regr = tree.DecisionTreeRegressor(criterion='friedman_mse')
            regr = regr.fit(X_balanced, y_balanced)
            print(regr.get_params())
            export_name = 'dtr'
        elif model_type == 'linear':
            regr = linear_model.LinearRegression()
            # Train the model using the training sets
            regr.fit(X_balanced, y_balanced)
            # The coefficients
            print('Coefficients: \n', regr.coef_)
            export_name = 'linear'
        elif model_type == 'svm':
            regr = SVR(kernel='rbf', degree=7, verbose=False, epsilon=0.000001, gamma='scale', tol=.0000001, shrinking=True)
            # Train the model using the training sets
            regr.fit(X_balanced, y_balanced)
            # The coefficients
            print('Coefficients: \n', regr.get_params())
            export_name = 'svm'
        elif model_type == 'dummy':
            regr = RndRegressor()
            export_name = 'dummy'
            is_regressor = False
        elif model_type == 'ideal':
            from IdealRegressor import IdealRegressor
            regr = IdealRegressor(X_train, y_train)
            #regr.predict(X_train)
            regr.fit(X_test, y_test)
            #regr.predict(X_test)
            export_name = 'ideal'
        elif model_type == 'nb':
            #from sklearn import svm
            #regr = svm.SVC(gamma='scale').fit(X_train, labels_train)
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

        model_results = Learn.evaluate_model(regr, X_test, X_balanced, y_test, y_balanced, labels_test, labels_balanced, is_regressor)

        print('Summarizing dataset and evaluating Rouge...')

        rouge_scores = evaluate_summarizer(regr, test_documents, features_to_use, True)
        utilities.print_rouges(rouge_scores)
        utilities.export_model(regr, export_name)
        print('*****************************************************************************')
    return rouge_scores, model_results


def evaluate_summarizer(clf, dataset, used_features, remove_stopwords=False):
    rouge = Rouge()
    empty_score = {
        'rouge-1': {'p': 0, 'f': 0, 'r': 0},
        'rouge-2': {'p': 0, 'f': 0, 'r': 0},
        'rouge-l': {'p': 0, 'f': 0, 'r': 0}
    }
    total_scores = {
        'rouge-1': {'p': 0, 'f': 0, 'r': 0},
        'rouge-2': {'p': 0, 'f': 0, 'r': 0},
        'rouge-l': {'p': 0, 'f': 0, 'r': 0}
    }
    avg_scores = empty_score
    total_summaries = 0
    for key in sorted(dataset):
    # diff_summs = 0):
        total_summaries += 1
        doc = dataset[key]
        doc['key'] = key
        #print(str(key) + doc['file_name'])
        gold_summaries = dataset[key]['summaries']
        best_score = empty_score
        summary = summ(doc, clf, used_features)
        if len(summary) == 0:
            continue
        #eng.write_xml(summary, key, doc['file_name'])
        summary_sentences = [s for (i, s) in summary]
        summary_text = " ".join(summary_sentences)
        summary_words = eng.full_preprocess(summary_text, remove_stopwords)
        summary_processed_text = ' '.join(summary_words)
        for ref_key in gold_summaries:
            ref_sentences = gold_summaries[ref_key]
            ref_words = eng.full_preprocess(' '.join(ref_sentences), remove_stopwords)
            ref_processed_text = ' '.join(ref_words)
            scores = rouge.get_scores(summary_processed_text, ref_processed_text)[0]
            """f_file = open('/tmp/summaries/' + ref_key + str(scores["rouge-1"]["f"]) + '.txt', '+w')
            f_file.writelines(lines)
            f_file.close()"""
            best_score = Learn.best_rouge_f(best_score, scores)

        for test_type in best_score:
            for param in best_score[test_type]:
                total_scores[test_type][param] += best_score[test_type][param]

    total_docs = len(dataset)
    for test_type in total_scores:
        for param in total_scores[test_type]:
            avg_scores[test_type][param] = total_scores[test_type][param] / total_summaries
    return avg_scores


