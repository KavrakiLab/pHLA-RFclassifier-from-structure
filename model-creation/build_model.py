import pickle
import numpy as np
from sklearn.model_selection import KFold
from sklearn import metrics
import sys
import create_contact_features
import math
from joblib import dump, load
from sklearn.metrics import confusion_matrix
from collections import defaultdict

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
#from sklearn.svm import LinearSVC
#from sklearn.svm import SVC
#from sklearn.naive_bayes import GaussianNB

#import eli5
#from eli5.sklearn import PermutationImportance

def printMetrics(output_label):

    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)
    y_scores = clf.predict_proba(X_test)[:,1] #clf.predict(X_test)
    auroc = metrics.roc_auc_score(y_test, y_scores)
    y_pred = clf.predict(X_test)

    #auprc = metrics.average_precision_score(y_test_collapsed, y_scores_collapsed)
    #print(train_acc, test_acc, auroc, auprc)
    #print(metrics.classification_report(y_test, clf.predict(X_test)))
    tp = np.sum(y_test * y_pred)
    fp = np.sum((y_test == 0) * y_pred)
    tn = np.sum((y_test == 0) * (y_pred==0))
    fn = np.sum(y_test * (y_pred==0))
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))


    sorted_indices = np.argsort(y_scores)
    num_1_percent = math.ceil(len(sorted_indices) * 0.25)
    top_1_percent = sorted_indices[-num_1_percent:]
    pp1 = clf.score(X_test[top_1_percent], y_test[top_1_percent])
    #oob = clf.oob_score_

    output_vars = [output_label, train_acc, test_acc, auroc, pp1, tp, fp, tn, fn, precision, recall, mcc]
    str_output_vars = (str(v) for v in output_vars)

    #print(output_label+","+str(train_acc)+","+str(test_acc)+","+str(auroc)+","+str(auprc)+","+str(pp1)+","+str(oob))
    print(",".join(str_output_vars))

# To get csv files in param_sweep:
# mode = kfold, and for model in {lr, xg, rf}, run this script for datasource in data_singleconf_{reg,r2,sig}.pkl

# To get csv files for leave-one-allele-out tests:
# mode = allele, model = rf, and run this script for datasource in data_singleconf_{reg,r2,sig}.pkl

# To get final rf model trained on all data for a particular featurization, (sig for the one on docker hub)
# mode = full, model = rf, and run this for datasource in data_singleconf_{reg,r2,sig}.pkl

mode = sys.argv[1] # kfold, allele, or full
datasource = sys.argv[2] # pkl file with data
model = sys.argv[3] # lr, xg, or rf - only in kfold or allele mode

print("Reading Data")

data = pickle.load(open(datasource,'rb'))
X = data['X']
y = data['y']
peptides = np.array(data['peptides'])
alleles = np.array(data['alleles'])

print(X.shape, y.shape, peptides.shape, alleles.shape)

"""
random_subset = np.random.choice(len(y), size=100)
X = X[random_subset]
y = y[random_subset]
peptides = peptides[random_subset]
alleles = alleles[random_subset]
print(X.shape, y.shape, peptides.shape, alleles.shape)
"""

"""
rows_finite = np.isfinite(X).all(axis=1)
X = X[rows_finite]
y = y[rows_finite]
peptides = peptides[rows_finite]
alleles = alleles[rows_finite]
print(X.shape, y.shape, peptides.shape, alleles.shape)
"""

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

    kf = KFold(n_splits=5, shuffle=True)
    test_train_splits = kf.split(X)

    k = 0
    for train_index, test_index in test_train_splits:

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

        # RF
        if model == "rf":
            n_est = [100, 500, 1000]
            max_feats = ["log2", "sqrt", 0.1]

            for est in n_est:
                for fe in max_feats:
                    param_name = "-" + str(est) + "-" + str(fe)
                    clf = RandomForestClassifier(n_estimators=est, criterion='gini', max_features=fe, n_jobs=-1, class_weight='balanced', oob_score=True)
                    clf.fit(X_train, y_train)
                    printMetrics(str(k)+param_name)
        
        # xgboost
        elif model == "xg":
            n_est = [100, 500, 1000]
            lrs = [0.01, 0.1, 0.2]

            for est in n_est:
                for lr in lrs:
                    param_name = "-" + str(est) + "-" + str(lr)
                    clf = XGBClassifier(n_jobs=8,n_estimators=est,learning_rate=lr,scale_pos_weight=num_neg_in_train/float(num_pos_in_train))
                    clf.fit(X_train, y_train)
                    printMetrics(str(k)+param_name)

        elif model == "lr":
            Cs = [0.1,1.0,10.0]
            #kernels = ["linear", "rbf"]
            #penalties = ["l1", "l2"]

            for C in Cs:
                param_name = "-" + str(C)
                #clf = SVC(C=C, kernel=ker, probability=True, class_weight='balanced', gamma="scale")
                #clf = LinearSVC(C=C, dual=False, probability=True, class_weight='balanced', gamma="scale")
                clf = LogisticRegression(C=C, class_weight='balanced', n_jobs=-1)
                clf.fit(X_train, y_train)
                printMetrics(str(k)+param_name)

        k += 1

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

        if model == "lr":
            clf = LogisticRegression(C=10.0, class_weight='balanced', n_jobs=-1)
        elif model == "xg":
            clf = XGBClassifier(n_jobs=8,n_estimators=1000,learning_rate=0.2,scale_pos_weight=num_neg_in_train/float(num_pos_in_train))
        elif model == "rf"
            clf = RandomForestClassifier(n_estimators=1000, criterion='gini', max_features='log2', n_jobs=-1, class_weight='balanced', oob_score=True)

        clf.fit(X_train, y_train)

        printMetrics(allele)


#sys.exit(0)
# final model
elif mode == "full":
    print("Training on whole dataset")
    clf = RandomForestClassifier(n_estimators=1000, criterion='gini', max_features='log2', n_jobs=-1, class_weight='balanced')
    clf.fit(X, y)
    dump(clf, datasource + ".joblib", compress=9, protocol=pickle.HIGHEST_PROTOCOL) 




