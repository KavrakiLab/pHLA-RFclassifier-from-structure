import pickle
import numpy as np
from sklearn.model_selection import KFold
from sklearn import metrics
import sys
import create_contact_features
import matplotlib.pyplot as plt
import math
from joblib import dump, load
from sklearn.metrics import confusion_matrix
from collections import defaultdict
import glob

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
#from sklearn.svm import LinearSVC
#from sklearn.svm import SVC
#from sklearn.naive_bayes import GaussianNB

#import eli5
#from eli5.sklearn import PermutationImportance

def printMetrics(output_label, test_index=None):

    if mode == "allele": test_index = np.logical_or(pos_test, neg_test)
    y_scores = clf.predict_proba(X_test)[:,1]
    probs = defaultdict(lambda: [])
    labels = defaultdict(lambda: [])
    for i in range(len(peptides[test_index])):
        ap_name = alleles[test_index][i] + "-" + peptides[test_index][i]
        probs[ap_name].append(y_scores[i])
        labels[ap_name].append(y_test[i])

    y_scores_collapsed = []
    y_test_collapsed = []

    for allele_peptide in labels.keys(): # order is now different than original
        allele, peptide = allele_peptide.split("-")
        y_scores_collapsed.append(np.mean(probs[allele_peptide]))
        y_test_collapsed.append(labels[allele_peptide][0]) # all should be same anyway

    y_scores_collapsed = np.array(y_scores_collapsed)
    y_test_collapsed = np.array(y_test_collapsed)
    y_pred_collapsed = y_scores_collapsed > 0.5

    train_acc = clf.score(X_train, y_train)
    test_acc = metrics.accuracy_score(y_test_collapsed, y_pred_collapsed)
    auroc = metrics.roc_auc_score(y_test_collapsed, y_scores_collapsed)

    #auprc = metrics.average_precision_score(y_test_collapsed, y_scores_collapsed)
    #print(train_acc, test_acc, auroc, auprc)
    #print(metrics.classification_report(y_test, clf.predict(X_test)))
    tp = np.sum(y_test_collapsed * y_pred_collapsed)
    fp = np.sum((y_test_collapsed == 0) * y_pred_collapsed)
    tn = np.sum((y_test_collapsed == 0) * (y_pred_collapsed==0))
    fn = np.sum(y_test_collapsed * (y_pred_collapsed==0))
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))


    #sorted_indices = np.argsort(y_scores_collapsed)
    #num_1_percent = math.ceil(len(sorted_indices) * 0.25)
    #top_1_percent = sorted_indices[-num_1_percent:]
    pp1 = -1 #clf.score(X_test[top_1_percent], y_test[top_1_percent])
    #oob = clf.oob_score_

    output_vars = [output_label, train_acc, test_acc, auroc, pp1, tp, fp, tn, fn, precision, recall, mcc]
    str_output_vars = (str(v) for v in output_vars)

    #print(output_label+","+str(train_acc)+","+str(test_acc)+","+str(auroc)+","+str(auprc)+","+str(pp1)+","+str(oob))
    print(",".join(str_output_vars))

# To get csv files for leave-one-allele-out tests:
# mode = allele, run this script for datasource in ensemble_{reg,r2,sig}.pkl (make sure files are labeled data*.pkl)

# To get final rf model trained on all ensemble data for a particular featurization, (sig for the one on docker hub)
# mode = full, and run this for datasource in ensemble_{reg,r2,sig}.pkl (make sure files are labeled data*.pkl)

mode = sys.argv[1] # kfold, allele, or full
datasource = sys.argv[2] # folder with the data*.pkl files

print("Reading Data")

for i in range(100):
    print(i)
    data = pickle.load(open(datasource + "/data" + str(i) + ".pkl", 'rb'))
    X_i = data['X']
    y_i = data['y']
    peptides_i = data['peptides']
    alleles_i = data['alleles']
    if len(X_i) == 0: continue
    if i == 0:
        X = X_i
        y = y_i
        peptides = peptides_i
        alleles = alleles_i
    else:
        X = np.concatenate((X, X_i), axis=0)
        y = np.concatenate((y, y_i), axis=0)
        peptides = np.concatenate((peptides, peptides_i), axis=0)
        alleles = np.concatenate((alleles, alleles_i), axis=0)

X = np.array(X)
y = np.array(y)
peptides = np.array(peptides)
alleles = np.array(alleles)

print(X.shape, y.shape, peptides.shape, alleles.shape)

"""
random_subset = np.random.choice(len(y), size=100)
X = X[random_subset]
y = y[random_subset]
peptides = peptides[random_subset]
alleles = alleles[random_subset]
print(X.shape, y.shape, peptides.shape, alleles.shape)
"""


rows_finite = np.isfinite(X).all(axis=1)
X = X[rows_finite]
y = y[rows_finite]
peptides = peptides[rows_finite]
alleles = alleles[rows_finite]
print(X.shape, y.shape, peptides.shape, alleles.shape)


allele_list = {}
for a in alleles: allele_list[a] = 1
allele_list = list(allele_list)
allele_list.sort()

total_num_samples = len(y)
print("Total Num Samples:", total_num_samples)
print("Positive/Negative:", np.sum(y==1)/float(total_num_samples), np.sum(y==0)/float(total_num_samples))
print("Num alleles:", len(allele_list))
print("output_label, train_acc, test_acc, auroc, pp1, tp, fp, tn, fn, precision, recall, mcc")


if mode == "kfold":

    phlas = defaultdict(lambda: [])
    for i in range(len(peptides)):
        allele, peptide = alleles[i], peptides[i]
        phla = allele + "-" + peptide
        phlas[phla].append(i)
    print(len(phlas.keys()))

    phla_names = list(phlas.keys())
    phla_names.sort()
    np.random.shuffle(phla_names)
    kfold_split = np.array_split(phla_names, 5)

    """
    f = open("ensemble_stats.txt", 'w')
    phla_names = list(phlas.keys())
    phla_names.sort()
    for phla in phla_names:
        f.write(phla + " " + str(len(phlas[phla])) + "\n")
    f.close()
    """

    #kf = KFold(n_splits=5, shuffle=True)
    #test_train_splits = kf.split(X)

    k = 0
    #for train_index, test_index in test_train_splits:
    for j in range(5):

        test_split = kfold_split[j]
        train_split = np.array([phla for phla in phla_names if phla not in test_split])

        train_index = []
        test_index = []

        for phla in test_split: test_index += phlas[phla]
        for phla in train_split: train_index += phlas[phla]

        train_index = np.array(train_index)
        test_index = np.array(test_index)

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        num_pos_in_train = np.sum(y_train==1)
        num_neg_in_train = np.sum(y_train==0)
        num_pos_in_test = np.sum(y_test==1)
        num_neg_in_test = np.sum(y_test==0)

        num_train = float(X_train.shape[0])
        num_test = float(X_test.shape[0])
        print("Train Stats:", X_train.shape, num_pos_in_train/num_train, num_neg_in_train/num_train)
        print("Test Stats:", X_test.shape, num_pos_in_test/num_test, num_neg_in_test/num_test)

        clf = RandomForestClassifier(n_estimators=1000, criterion='gini', max_features="log2", n_jobs=-1, class_weight='balanced', oob_score=True)
        clf.fit(X_train, y_train)
        printMetrics(str(k), test_index)


        k += 1

    sys.exit(0)

elif mode == "allele":

    for i, allele in enumerate(allele_list):

        pos_train = np.logical_and(alleles != allele, y == 1)
        neg_train = np.logical_and(alleles != allele, y == 0)

        pos_test = np.logical_and(alleles == allele, y == 1)
        neg_test = np.logical_and(alleles == allele, y == 0)

        num_pos_in_test = np.sum(pos_test)
        num_neg_in_test = np.sum(neg_test)

        num_pos_in_train = np.sum(pos_train)
        num_neg_in_train = np.sum(neg_train)

        if num_pos_in_test == 0 or num_neg_in_test == 0: continue

        X_train_pos, X_train_neg = X[pos_train], X[neg_train]
        X_test_pos, X_test_neg = X[pos_test], X[neg_test]
        y_train_pos, y_train_neg = y[pos_train], y[neg_train]
        y_test_pos, y_test_neg = y[pos_test], y[neg_test]

        X_train = np.concatenate((X_train_pos, X_train_neg), axis=0)
        y_train = np.concatenate((y_train_pos, y_train_neg), axis=0)
        X_test = np.concatenate((X_test_pos, X_test_neg), axis=0)
        y_test = np.concatenate((y_test_pos, y_test_neg), axis=0)

        num_train = float(X_train.shape[0])
        num_test = float(X_test.shape[0])
        print("Train Stats:", X_train.shape, num_pos_in_train/num_train, num_neg_in_train/num_train)
        print("Test Stats:", X_test.shape, num_pos_in_test/num_test, num_neg_in_test/num_test)

        clf = RandomForestClassifier(n_estimators=1000, criterion='gini', max_features='log2', n_jobs=-1, class_weight='balanced', oob_score=True)
        #clf = XGBClassifier(n_jobs=8)

        clf.fit(X_train, y_train)

        printMetrics(allele)


#sys.exit(0)
# final model
elif mode == "full":
    print("Training on whole dataset")
    clf = RandomForestClassifier(n_estimators=1000, criterion='gini', max_features='log2', n_jobs=-1, class_weight='balanced')
    clf.fit(X, y)
    dump(clf, datasource + ".joblib", compress=9, protocol=pickle.HIGHEST_PROTOCOL) 



