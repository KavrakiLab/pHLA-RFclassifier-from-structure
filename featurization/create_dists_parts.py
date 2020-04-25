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

pHLA_to_label = {}

#os.chdir("test")
all_files = []
f = open(file_with_confs, 'r')
for line in f:
    all_files.append(line)
f.close()
#all_files = glob.glob("*.pdb")
print(len(all_files))


start = time.time()
X = []
y = []
peptides = []
alleles = []
interactions_all = []
distances_all = []
for i in range(len(all_files)):
    print(i)
    #fi_name = str(all_files[i]).rstrip()
    #allele, peptide = f[:-4].split("-")
    allele, peptide = all_files[i].split()
    peptide = peptide.rstrip()    
    a_name = allele[4] + allele[6:8] + allele[-2:]
    fi_name = a_name + "-" + peptide
    confs = glob.glob("min_confs/" + fi_name + ".pdb")
    #print(confs)
    for c in confs:
        try:
            #peptide = f[:-4]
            #feature_vec = create_contact_features.featurize(c)
            interactions, resres_dists = create_contact_features.get_dists(c)
            peptides.append(peptide)
            alleles.append(a_name)
        except: continue
        interactions_all.append(interactions)
        distances_all.append(resres_dists)
end = time.time()
print(end-start)

X = np.array(X)
y = np.array(y)
distances_all = np.array(distances_all)
interactions_all = np.array(interactions_all)

data = {'interactions_all':interactions_all,'distances_all':distances_all, 'peptides':peptides, 'alleles':alleles}

f = open("data" + label + ".pkl", 'wb')
pickle.dump(data, f)
f.close()




