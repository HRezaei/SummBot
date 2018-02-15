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
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

def visualize(y, predicted, title):
    plt.figure()
    plt.scatter(range(len(y)), y, s=10, edgecolor="black",
                c="darkorange", label="train")
    plt.plot(range(len(y)), predicted, color="cornflowerblue",
            label="test", linewidth=1)
    #plt.plot(X_test, predicted, color="yellowgreen", label="predicted", linewidth=2)
    plt.xlabel("data")
    plt.ylabel("target")
    plt.title(title)
    plt.legend()
    plt.show()

def evaluate_summarizer(clf):
    datasetJson = json.loads(read_file('resources/pasokh/all.json') )
    rouge = Rouge()
    empty_score = {
        'rouge-1':{'p':0, 'f':0, 'r':0},
        'rouge-2':{'p':0, 'f':0, 'r':0},
        'rouge-l':{'p':0, 'f':0, 'r':0}
        }
    total_scores = {
        'rouge-1':{'p':0, 'f':0, 'r':0},
        'rouge-2':{'p':0, 'f':0, 'r':0},
        'rouge-l':{'p':0, 'f':0, 'r':0}
        }
    avg_scores = empty_score
    total_summaries = 0
    for key in datasetJson:
        total_summaries += 1
        text = datasetJson[key]['text']
        summary = summ(text, clf)
        gold_summaries = datasetJson[key]['summaries']
        best_score = empty_score
        for ref_key in gold_summaries:
            ref = gold_summaries[ref_key]
            scores = rouge.get_scores(ref, summary)[0]
            best_score = best_rouge_f(best_score, scores)
   
        for test_type in best_score:
            for param in best_score[test_type]:
                total_scores[test_type][param] += best_score[test_type][param]
    
    total_docs = len(datasetJson) 
    for test_type in total_scores:
        for param in total_scores[test_type]:
            avg_scores[test_type][param] = total_scores[test_type][param]/total_summaries
    
    print("{:<8} {:<25} {:<25} {:<25}".format('Test','f-measure','precision', 'recall'))
    for k in ['rouge-1', 'rouge-2', 'rouge-l']:
        v = avg_scores[k]
        print("{:<8} {:<25} {:<25} {:<25}".format(k, v['f'], v['p'], v['r']))


def best_rouge_f(score1, score2):
    sumF1 = score1["rouge-1"]["f"] + score1["rouge-2"]["f"] + score1["rouge-l"]["f"]
    sumF2 = score2["rouge-1"]["f"] + score2["rouge-2"]["f"] + score2["rouge-l"]["f"]
    if sumF2 > sumF1:
        return score2
    return score1

dataset = load_dataset('features.json')
#X_normal = StandardScaler().fit_transform(dataset[0])
X_normal = dataset[0]
X_train, X_test, y_train, y_test = train_test_split(X_normal, 
                                                    dataset[1], 
                                                    test_size=0.25,
                                                    random_state=0)

(X_balanced, y_balanced) = (X_train, y_train)
#X_balanced, y_balanced = balance_dataset(X_train, y_train, 2)
print("Train set size: {}".format(len(X_balanced)))
print("Test set size: {}".format(len(X_test)))
print("Used features: {}".format(len(X_balanced[0])))

export_name = 'my_dumped_regressor'
if len(sys.argv) < 2 or sys.argv[1] == 'dtr':
    #max_depth=6
    regr = tree.DecisionTreeRegressor(max_depth=6)
    regr = regr.fit(X_balanced, y_balanced)
    export_name = 'dtr_regressor'
elif sys.argv[1] == 'linear':
    regr = linear_model.LinearRegression()
    # Train the model using the training sets
    regr.fit(X_balanced, y_balanced)
    # The coefficients
    print('Coefficients: \n', regr.coef_)
    export_name = 'linear_regressor'
elif sys.argv[1] == 'dummy':
    from DummyRegressor import RndRegressor
    regr = RndRegressor()
    export_name = 'Dummy'
else:
    print("Regression type is undefined")
    exit()
# Make predictions using the testing set
y_pred = regr.predict(X_test)

y_pred_train = regr.predict(X_balanced)

# The mean squared error
print("Mean squared error: %.5f"
      % mean_squared_error(y_test, y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.5f' % r2_score(y_test, y_pred))
#visualize(y_test, y_pred, "Decision Tree Regression - Test set")

#visualize(y_balanced, y_pred_train, "Decision Tree Regression - Train set")
print("MSE on train: %.5f"
      % mean_squared_error(y_balanced, y_pred_train))
print('Variance score on train: %.5f' % r2_score(y_balanced, y_pred_train))

with open(export_name+'.pkl', 'wb') as fid:
    cPickle.dump(regr, fid)

print('Summarizing dataset and evaluating Rouge...')
evaluate_summarizer(regr)
