import sys
import create_contact_features
import numpy as np
import glob
import pickle
from threading import Thread
from subprocess import call
import os
import time

class ReceptorThread(Thread):

    def __init__(self, loop_indices, files, label):
        self.loop_indices = loop_indices
        self.files = files
        self.label = label
        
        Thread.__init__(self)

    def run(self):
        X = []
        y = []
        peptides = []
        alleles = []
        folder_name = str(self.loop_indices[0])
        call(["mkdir " + folder_name], shell=True)
        for i in self.loop_indices:
            print(i)
            fi_name = str(self.files[i]).rstrip()
            #allele, peptide = f[:-4].split("-")
            allele, peptide = fi_name.split("/")[0].split("-")
            try:
                #peptide = f[:-4]
                feature_vec = create_contact_features.featurize(fi_name)
                peptides.append(peptide)
                alleles.append(allele)
            except: continue
            X.append(feature_vec)
            y.append(pHLA_to_label[fi_name.split("/")[0]])

        X = np.array(X)
        y = np.array(y)

        data = {'X':X, 'y':y, 'peptides':peptides, 'alleles':alleles}

        f = open(folder_name + "/data.pkl", 'wb')
        pickle.dump(data, f)
        f.close()


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

print(len(pHLA_to_label.keys()))

#os.chdir("test")
all_files = []
f = open(sys.argv[1], 'r')
for line in f:
    all_files.append(line)
f.close()
#all_files = glob.glob("*.pdb")
print(len(all_files))


num_cores = min(1, len(all_files))
array_splits = np.array_split(list(range(len(all_files))), num_cores)
folder_names = [str(s[0]) for s in array_splits]

start = time.time()
threads = []
for loop_indices in array_splits:
    t = ReceptorThread(loop_indices, all_files, 0)
    threads.append(t)
    t.start()
for t in threads: t.join()
end = time.time()
print(end-start)

#X = []
#y = []
#peptides = []
for i,fol in enumerate(folder_names):
    data = pickle.load(open(fol + "/data.pkl", 'rb'))
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

data = {'X':X, 'y':y, 'peptides':peptides, 'alleles':alleles}

f = open("data.pkl", 'wb')
pickle.dump(data, f)
f.close()

