import sys
import create_contact_features
import numpy as np
import glob
import pickle
from threading import Thread
from subprocess import call
import os
import time

label = sys.argv[1]
file_with_confs = "confs" + str(label) + ".txt"
mode = sys.argv[2]

pHLA_to_label = {}

f = open("all_nonbinders.txt", 'r')
for line in f:
    allele, peptide = line.split()
    allele_name = allele[4] + allele[6:8] + allele[-2:]
    pHLA_to_label[allele_name + "-" + peptide] = 0
f.close()

f = open("all_binders.txt", 'r')
for line in f:
    allele, peptide = line.split()
    allele_name = allele[4] + allele[6:8] + allele[-2:]
    pHLA_to_label[allele_name + "-" + peptide] = 1
f.close()


#os.chdir("test")
all_files = []
f = open(file_with_confs, 'r')
for line in f:
    fname = line.rstrip()
    all_files.append(fname)
f.close()
#all_files = glob.glob("*.pdb")
print(len(all_files))


start = time.time()
X = []
y = []
peptides = []
alleles = []
for i in range(len(all_files)):
    print(i)
    #allele, peptide = all_files[i].split()
    #peptide = peptide.rstrip()
    #a_name = allele[4] + allele[6:8] + allele[-2:]
    #fi_name = a_name + "-" + peptide
    #confs = glob.glob("min_confs/" + fi_name + ".pdb")
    fullpdbname = all_files[i]
    phla, pdbname = fullpdbname.split("/")
    fi_name = phla
    allele, peptide = phla.split("-")
    a_name = allele
    conf_name = "ensemble/" + fullpdbname
    confs = glob.glob(conf_name)
    #print(conf_name, confs, fi_name)
    for c in confs:
        try:
            #peptide = f[:-4]
            if mode == "reg": feature_vec = create_contact_features.featurize(c)
            elif mode == "r2": feature_vec = create_contact_features.featurize_r2(c)
            elif mode == "sig": feature_vec = create_contact_features.featurize_sig(c)
            peptides.append(peptide)
            alleles.append(a_name)
        except: continue
        X.append(feature_vec)
        y.append(pHLA_to_label[fi_name])
end = time.time()
print(end-start)

X = np.array(X)
y = np.array(y)

data = {'X':X, 'y':y, 'peptides':peptides, 'alleles':alleles}

f = open("data" + mode + label + ".pkl", 'wb')
pickle.dump(data, f)
f.close()




