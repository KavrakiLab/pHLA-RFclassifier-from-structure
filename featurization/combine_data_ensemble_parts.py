import sys
import create_contact_features
import numpy as np
import glob
import pickle
from threading import Thread
from subprocess import call
import os



for i in range(48):
    print(i)
    data = pickle.load(open("data" + str(i) + ".pkl", 'rb'))
    X_i = data['X']
    y_i = data['y']
    peptides_i = data['peptides']
    alleles_i = data['alleles']

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


    #call(["rm -r " + fol], shell=True)

X = np.array(X)
y = np.array(y)

print(X.shape, y.shape, peptides.shape, alleles.shape)

data = {'X':X, 'y':y, 'peptides':peptides, 'alleles':alleles}

f = open("data_ensemble.pkl", 'wb')
pickle.dump(data, f)
f.close()
