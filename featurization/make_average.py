import pickle
from sklearn.ensemble import RandomForestClassifier
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

#from xgboost import XGBClassifier
#import eli5
#from eli5.sklearn import PermutationImportance


alleles = []
allele_indices = []
running_index = 0
print("Reading Data")

data = pickle.load(open("data_ensemble.pkl",'rb'))
X_o = data['X']
y_o = data['y']
peptides = data['peptides']
alleles = data['alleles']

rows_finite = np.isfinite(X_o).all(axis=1)
X_o = X_o[rows_finite]
y_o = y_o[rows_finite]
peptides = peptides[rows_finite]
alleles = alleles[rows_finite]

allele_peptides = defaultdict(lambda: [])
labels = defaultdict(lambda: [])
for i in range(len(peptides)):
    if (i%10000) == 0: print(i)
    ap_name = alleles[i] + "-" + peptides[i]
    allele_peptides[ap_name].append(X_o[i])
    labels[ap_name].append(y_o[i])

X = []
y = []
peptides_n = []
alleles_n = []
for allele_peptide in allele_peptides.keys():
    allele, peptide = allele_peptide.split("-")
    X.append(np.mean(allele_peptides[allele_peptide], axis=0))
    y.append(labels[allele_peptide][0])
    peptides_n.append(peptide)
    alleles_n.append(allele)

X = np.array(X)
y = np.array(y)

print(X.shape, y.shape, len(peptides_n), len(alleles_n))

data = {'X':X, 'y':y, 'peptides':peptides_n, 'alleles':alleles_n}

f = open("data_ensemble_avg.pkl", 'wb')
pickle.dump(data, f)
f.close()


