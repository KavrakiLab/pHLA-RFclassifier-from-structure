import pickle
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn import metrics
import sys
import create_contact_features
from joblib import dump, load
import mdtraj as md

pdbfile = sys.argv[1]
model = "/rf_classifier/data_singleconf_sig.pkl.joblib" #sys.argv[2]
doInterpretation = int(sys.argv[2]) == 1

print("Featurizing")
pepmhc_names, pepmhc_dists = create_contact_features.get_distances(pdbfile)
feature_vec = create_contact_features.featurize_from_distances(pepmhc_names, pepmhc_dists, "sig")

print("Loading Model")
clf = load(model)

print("Making prediction")
X_i = np.array([feature_vec])
output_prob = clf.predict_proba(X_i)[0][1]
if output_prob >= 0.5: y_pred_i = 1
else: y_pred_i = 0
if y_pred_i == 1: print("Binder:", output_prob)
else: print("Nonbinder:", output_prob)

if not doInterpretation: sys.exit(0)

print("Computing Interpretation")
from treeinterpreter import treeinterpreter as ti

prediction, bias, contributions = ti.predict(clf, X_i)

print("Prediction:", prediction[0][1])
print("Bias (trainset mean)", bias[0][1])

contri = contributions[0, :, 1]

print("Top 5 conformational feature contributions towards the prediction")
print("Interaction, Contribution Value, Percentage towards prediction")
sorted_indices = np.argsort(contri)
if y_pred_i == 1:
    sorted_indices = sorted_indices[::-1]
    tot_c = np.sum(contri[contri>0])
elif y_pred_i == 0:
    tot_c = np.sum(contri[contri<0])

num_printed = 0
for j in sorted_indices:
    feature_value = X_i[0, j]
    if feature_value < 0.00001: continue # feature does not exist in conformation
    print(create_contact_features.new_index_to_interaction[j], contri[j], contri[j]/tot_c)
    num_printed += 1
    if num_printed == 5: break

print("All feature contributions saved in feature_contributions.csv")
f = open("feature_contributions.csv", 'w')
f.write("Feature Index,Interaction Type,Contribution Value,Feature Value,Pos Contribution,Neg Contribution\n")
for j in range(210):
    if contri[j] > 0: 
        pos_c = contri[j]/np.sum(contri[contri>0])
        neg_c = -1
    elif contri[j] < 0:
        pos_c = -1
        neg_c = contri[j]/np.sum(contri[contri<0])
    f.write(str(j) + "," + create_contact_features.new_index_to_interaction[j] + "," + str(contri[j]) + "," + str(X_i[0, j]) + "," + str(pos_c) + "," + str(neg_c) + "\n")
f.close()

print("Decomposing contributions towards individual peptide-mhc distances in contribution_decomp.csv")
contri_per_distance, weights = create_contact_features.decompose_contributions(pepmhc_names, pepmhc_dists, contri, "sig")
f = open("contribution_decomp.csv", 'w')
f.write("Pep Res, MHC Res, Weight, Contribution\n")
for k in range(len(pepmhc_names)):
    r1 = pepmhc_names[k][0]
    r2 = pepmhc_names[k][1]
    c = contri_per_distance[k]
    w = weights[k]
    f.write(r1 + "," + r2 + "," + str(w) + "," + str(c) + "\n")
f.close()

print("Attributing feature contributions toward peptide residues")
pepres = [str(r) for r in md.load(pdbfile).top.residues]
num_pepres = len(md.load(pdbfile).top.select("chainid == 2 and name == 'CA'"))
pepcontri = []
for i in range(num_pepres):
    pepcontri.append(create_contact_features.get_net_contribution(pepmhc_names, contri_per_distance, "pep", i+1))

print("Position, Pos Contribution, Neg Contribution")
for i in range(num_pepres): 
    print(i+1, pepcontri[i][0], pepcontri[i][1])

print("Peptide contributions saved in peptide_contributions.csv")
f = open("peptide_contributions.csv", 'w')
f.write("Position, Pos Contribution, Neg Contribution\n")
for i in range(num_pepres):
    f.write(str(i+1) + "," + str(pepcontri[i][0]) + "," + str(pepcontri[i][1]) + "\n")
f.close()

print("Attributing feature contributions toward MHC residues (sorted by contribution towards prediction)")
num_mhcres = 180
mhccontri = []
for i in range(num_mhcres):
    mhccontri.append(create_contact_features.get_net_contribution(pepmhc_names, contri_per_distance, "mhc", i+1))

print("Position, Pos Contribution, Neg Contribution")
if y_pred_i == 1:
    vals = [m[0] for m in mhccontri]
    sorted_indices = np.argsort(vals)
    sorted_indices = sorted_indices[::-1]
elif y_pred_i == 0:
    vals = [m[1] for m in mhccontri]
    sorted_indices = np.argsort(vals)
#for i in sorted_indices: 
#    print(i+1, mhccontri[i][0], mhccontri[i][1])

print("MHC contributions saved in mhc_contributions.csv")
f = open("mhc_contributions.csv", 'w')
f.write("Position, Pos Contribution, Neg Contribution\n")
for i in sorted_indices: #range(num_mhcres):
    f.write(str(i+1) + "," + str(mhccontri[i][0]) + "," + str(mhccontri[i][1]) + "\n")
f.close()

