import pickle
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
import sys
import create_contact_features
import matplotlib.pyplot as plt
import math
from joblib import dump, load
from sklearn.metrics import confusion_matrix
from collections import defaultdict

#from xgboost import XGBClassifier
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


mode = sys.argv[1]
datasource = sys.argv[2]

print("Reading Data")

data = pickle.load(open(datasource,'rb'))
X = data['X']
y = data['y']
peptides = np.array(data['peptides'])
alleles = np.array(data['alleles'])

print(X.shape, y.shape, peptides.shape, alleles.shape)

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

decoys = defaultdict(lambda: [])
#f = open("extra_nonbinders.txt", 'r')
f = open("labeled_extra.txt", 'r')
for line in f:
    allele, peptide, aff = line.split()
    #peptide = peptide.rstrip()
    a_name = allele[4] + allele[6:8] + allele[-2:]
    #if float(aff) > 20000: decoys[a_name+"-"+peptide].append(1)
    decoys[a_name+"-"+peptide].append(1)
f.close()

isdecoy = []
for i in range(len(peptides)):
    pHLA = alleles[i] + "-" + peptides[i]
    if len(decoys[pHLA]) > 0: isdecoy.append(True)
    else: isdecoy.append(False)
isdecoy = np.array(isdecoy)
isnotdecoy = np.logical_not(isdecoy)

total_num_samples = len(y)
print("Total Num Samples:", total_num_samples)
print("Positive/Negative:", np.sum(y==1)/float(total_num_samples), np.sum(y==0)/float(total_num_samples))
print("Num alleles:", len(allele_list))
print("Num decoys:", np.sum(isdecoy))
print("Num nonbinders:", np.sum(isnotdecoy))
print("output_label, train_acc, test_acc, auroc, pp1, tp, fp, tn, fn, precision, recall, mcc")

if mode == "kfold":

    X = X[isnotdecoy]
    y = y[isnotdecoy]

    pos_index = y==1
    neg_index = y==0

    X_pos, y_pos = X[pos_index], y[pos_index]
    X_neg, y_neg = X[neg_index], y[neg_index]

    random_subset = np.random.choice(np.sum(pos_index), size=np.sum(neg_index), replace=False)
    X_pos, y_pos = X_pos[random_subset], y_pos[random_subset]

    X = np.concatenate((X_pos, X_neg), axis=0)
    y = np.concatenate((y_pos, y_neg), axis=0)

    kf = StratifiedKFold(n_splits=5, shuffle=True)
    #kf.get_n_splits(X)
    test_train_splits = kf.split(X, y)

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

        clf = RandomForestClassifier(n_estimators=1000, criterion='gini', max_features='sqrt', n_jobs=-1, class_weight='balanced', oob_score=True)
        #clf = XGBClassifier(n_jobs=8)

        clf.fit(X_train, y_train)

        printMetrics(str(k))
        k += 1

elif mode == "allele":

    for i, allele in enumerate(allele_list):

        pos_train = np.logical_and(alleles != allele, y == 1)
        neg_train = np.logical_and(alleles != allele, np.logical_and(isnotdecoy, y == 0))

        pos_test = np.logical_and(alleles == allele, y == 1)
        neg_test = np.logical_and(alleles == allele, np.logical_and(isnotdecoy, y == 0))

        #train_index = np.logical_and(alleles != allele, np.logical_or(y == 1, isdecoy))
        #test_index = np.logical_and(alleles == allele, np.logical_or(y == 1, isdecoy))
        #test_index = np.logical_and(alleles == allele, isnotdecoy)
        #pos_train = np.logical_and(train_index, y == 1)
        #neg_train = np.logical_and(train_index, y == 0)
        #pos_test = np.logical_and(test_index, y == 1)
        #neg_test = np.logical_and(test_index, y == 0)

        num_pos_in_test = np.sum(pos_test)
        num_neg_in_test = np.sum(neg_test)

        num_pos_in_train = np.sum(pos_train)
        num_neg_in_train = np.sum(neg_train)

        if num_pos_in_test == 0 or num_neg_in_test == 0: continue

        X_train_pos, y_train_pos = X[pos_train], y[pos_train]
        X_train_neg, y_train_neg = X[neg_train], y[neg_train]
        X_test_pos, y_test_pos = X[pos_test], y[pos_test]
        X_test_neg, y_test_neg = X[neg_test], y[neg_test]

        #random_subset = np.random.choice(num_neg_in_train, size=num_pos_in_train, replace=False)
        #X_train_neg = X_train_neg[random_subset]
        #y_train_neg = y_train_neg[random_subset]
        #num_neg_in_train = num_pos_in_train

        random_subset = np.random.choice(num_pos_in_train, size=num_neg_in_train, replace=False)
        X_train_pos = X_train_pos[random_subset]
        y_train_pos = y_train_pos[random_subset]
        num_pos_in_train = num_neg_in_train

        if num_pos_in_test > num_neg_in_test:
            random_subset = np.random.choice(num_pos_in_test, size=num_neg_in_test, replace=False)
            X_test_pos = X_test_pos[random_subset]
            y_test_pos = y_test_pos[random_subset]
            num_pos_in_test = num_neg_in_test
        elif num_neg_in_test > num_pos_in_test:
            random_subset = np.random.choice(num_neg_in_test, size=num_pos_in_test, replace=False)
            X_test_neg = X_test_neg[random_subset]
            y_test_neg = y_test_neg[random_subset]
            num_neg_in_test = num_pos_in_test

        X_train = np.concatenate((X_train_pos, X_train_neg), axis=0)
        y_train = np.concatenate((y_train_pos, y_train_neg), axis=0)
        X_test = np.concatenate((X_test_pos, X_test_neg), axis=0)
        y_test = np.concatenate((y_test_pos, y_test_neg), axis=0)

        num_train = float(X_train.shape[0])
        num_test = float(X_test.shape[0])
        print("Train Stats:", X_train.shape, num_pos_in_train/num_train, num_neg_in_train/num_train)
        print("Test Stats:", X_test.shape, num_pos_in_test/num_test, num_neg_in_test/num_test)

        clf = RandomForestClassifier(n_estimators=1000, criterion='gini', max_features='sqrt', n_jobs=-1, class_weight='balanced', oob_score=True)
        #clf = XGBClassifier(n_jobs=8)

        clf.fit(X_train, y_train)

        printMetrics(allele)


sys.exit(0)
# final model
print("Training on whole dataset")
clf = RandomForestClassifier(n_estimators=1000, criterion='gini', max_features='sqrt', n_jobs=-1, class_weight='balanced')
clf.fit(X, y)
dump(clf, 'models/all_data.joblib') 


print("Train Acc, Test Acc, AUROC, AUPRC, PP1, OOB")

kf = KFold(n_splits=5, shuffle=True)
kf.get_n_splits(X)

allele_splits = []
for i in range(len(allele_indices)):
    if not new_allele_list_bool[i]: continue
    test_indices = allele_indices[i]
    train_indices_list = [allele_indices[j] for j in range(len(allele_indices)) if j != i]
    train_indices = np.concatenate(train_indices_list)
    #print(train_indices.shape, test_indices.shape)
    allele_splits.append((train_indices, test_indices))
print(len(allele_splits))

if mode == "allele": test_train_splits = allele_splits
elif mode == "kfold": test_train_splits = kf.split(X)
else:
    print("Mode not recognized")
    sys.exit(0)

j = 0
for train_index, test_index in test_train_splits:

    if mode == "allele": print("Testing on", new_allele_list[j])
    j += 1
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    clf = RandomForestClassifier(n_estimators=1000, criterion='gini', max_features='sqrt', n_jobs=-1, class_weight='balanced', oob_score=True)
    #clf = XGBClassifier(n_jobs=8)

    clf.fit(X_train, y_train)
    #clf = load('models/model_not' + new_allele_list[j-1] + '.joblib') 

    y_scores = clf.predict_proba(X_test)[:,1]
    preds = defaultdict(lambda: [])
    labels = defaultdict(lambda: [])
    for i in range(len(peptides[test_index])):
        ap_name = alleles[test_index][i] + "-" + peptides[test_index][i]
        preds[ap_name].append(y_scores[i])
        labels[ap_name].append(y_test[i])

    y_scores_collapsed = []
    y_test_collapsed = []

    for allele_peptide in labels.keys():
        allele, peptide = allele_peptide.split("-")
        #X.append(np.mean(allele_peptides[allele_peptide], axis=0))
        y_scores_collapsed.append(np.mean(preds[allele_peptide]))
        y_test_collapsed.append(np.mean(labels[allele_peptide]))
        #peptides_n.append(peptide)
        #alleles_n.append(allele)

    y_scores_collapsed = np.array(y_scores_collapsed)
    y_test_collapsed = np.array(y_test_collapsed)


    #train_acc = clf.score(X_train, y_train)
    #test_acc = clf.score(X_test, y_test)
    #y_scores = clf.predict_proba(X_test)[:,1] #clf.predict(X_test)
    auroc = metrics.roc_auc_score(y_test_collapsed, y_scores_collapsed)

    auprc = metrics.average_precision_score(y_test_collapsed, y_scores_collapsed)
    #print(train_acc, test_acc, auroc, auprc)
    #print(metrics.classification_report(y_test, clf.predict(X_test)))

    sorted_indices = np.argsort(y_scores)
    num_1_percent = math.ceil(len(sorted_indices) * 0.25)
    top_1_percent = sorted_indices[-num_1_percent:]
    pp1 = clf.score(X_test[top_1_percent], y_test[top_1_percent])
    oob = clf.oob_score_

    if mode == "allele": print(new_allele_list[j-1]+","+str(train_acc)+","+str(test_acc)+","+str(auroc)+","+str(auprc)+","+str(pp1)+","+str(oob))
    else: print(str(train_acc)+","+str(test_acc)+","+str(auroc)+","+str(auprc)+","+str(pp1)+","+str(oob))
    #print(new_allele_list[j-1]+","+str(auroc)+","+str(auprc)+","+str(oob)) 
    #dump(clf, 'models/model_not' + new_allele_list[j-1] + '.joblib')  
    #clf = load('models/model_not' + new_allele_list[j-1] + '.joblib') 

    #c = confusion_matrix(y_test, clf.predict(X_test))
    #print(c)

   

sys.exit(0)
# final model
print("Training on whole dataset")
clf = RandomForestClassifier(n_estimators=1000, criterion='gini', max_features='sqrt', n_jobs=-1, class_weight='balanced')
clf.fit(X, y)

dump(clf, 'models/all_data.joblib') 

