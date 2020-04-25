import sys
import os
from subprocess import call

import pickle
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import KFold
from sklearn import metrics
import matplotlib.pyplot as plt
import math

# 2.0.2

allele = sys.argv[1] # ex. A0101
#hla_name_file = "HLA-" + allele[0] + "*" + allele[1:3] + ":" + allele[-2:]
#hla_name_netMHC = "HLA-" + allele[0:3] + allele[-2:]

# extract data from txt file
call(["grep \" 1\" modeled_pHLAs.txt | grep \"" + allele + "\" | awk '{ print $2 }' > temp.txt"], shell=True)
# run mixmhcpred
call(["./MixMHCpred/MixMHCpred -i temp.txt -o temp_out.txt -a " + allele], shell=True)
# extract predicted labels
call(["grep -P \"\\t" + allele + "\" temp_out.txt | awk '{ print $2 }' > out1.txt"], shell=True)

# extract data from txt file
call(["grep \" 0\" modeled_pHLAs.txt | grep \"" + allele + "\" | awk '{ print $2 }' > temp.txt"], shell=True)
#call(["grep \"HLA-" + allele[0] + "\\*" + allele[1:3] + ":" + allele[-2:] + "\" combined_nonbinders.txt | awk '{ print $2 }' > temp.txt"], shell=True)
call(["./MixMHCpred/MixMHCpred -i temp.txt -o temp_out.txt -a " + allele], shell=True)
# extract predicted labels
call(["grep -P \"\\t" + allele + "\" temp_out.txt | awk '{ print $2 }' > out0.txt"], shell=True)

y_scores = []
y_scores_pos = []
y_scores_neg = []
#y_true = []
f = open("out1.txt", 'r')
for line in f:
	y_scores_pos.append( np.max(float(line), 0) )
	#y_true.append(1)
f.close()
f = open("out0.txt", 'r')
for line in f:
	y_scores_neg.append( np.max(float(line), 0) )
	#y_true.append(0)
f.close()
y_scores_pos = np.array(y_scores_pos)
y_scores_neg = np.array(y_scores_neg)

"""
if len(y_scores_pos) > len(y_scores_neg):
	random_subset = np.random.choice(len(y_scores_pos), size=len(y_scores_neg), replace=False)
	y_scores_pos = y_scores_pos[random_subset]
elif len(y_scores_pos) < len(y_scores_neg):
	random_subset = np.random.choice(len(y_scores_neg), size=len(y_scores_pos), replace=False)
	y_scores_neg = y_scores_neg[random_subset]
num_in_class = len(y_scores_neg)
"""

y_scores = np.concatenate((y_scores_pos, y_scores_neg), axis=0)
y_true = len(y_scores_pos)*[1] + len(y_scores_neg)*[0]

y_scores = np.array(y_scores)
y_true = np.array(y_true)
y_pred = y_scores>0.5

#print(y_scores_pos.shape, y_scores_neg.shape, y_scores.shape, y_true.shape, y_pred.shape)

train_acc = -1 #clf.score(X_train, y_train)
test_acc = metrics.accuracy_score(y_true, y_pred) #clf.score(X_test, y_true)
auroc = metrics.roc_auc_score(y_true, y_scores) 

#auprc = metrics.average_precision_score(y_true_collapsed, y_scores_collapsed)
#print(train_acc, test_acc, auroc, auprc)
#print(metrics.classification_report(y_true, clf.predict(X_test)))
tp = np.sum(y_true * y_pred)
fp = np.sum((y_true == 0) * y_pred)
tn = np.sum((y_true == 0) * (y_pred==0))
fn = np.sum(y_true * (y_pred==0))
precision = tp / (tp + fp)
recall = tp / (tp + fn)
mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))


sorted_indices = np.argsort(y_scores)
num_1_percent = math.ceil(len(sorted_indices) * 0.50)
top_1_percent = sorted_indices[-num_1_percent:]
pp1 = metrics.accuracy_score(y_true[top_1_percent], y_pred[top_1_percent])
#oob = clf.oob_score_

output_label = allele
output_vars = [output_label, train_acc, test_acc, auroc, pp1, tp, fp, tn, fn, precision, recall, mcc]
str_output_vars = (str(v) for v in output_vars)

#print(output_label+","+str(train_acc)+","+str(test_acc)+","+str(auroc)+","+str(auprc)+","+str(pp1)+","+str(oob))
print(",".join(str_output_vars))


#print(allele+","+str(acc)+","+str(auroc)+","+str(auprc)+","+str(pp1))

call(["rm temp.txt temp_out.txt out0.txt out1.txt"], shell=True)

