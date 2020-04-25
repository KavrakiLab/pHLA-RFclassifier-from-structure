import pickle
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn import metrics
import sys
import create_contact_features
from joblib import dump, load
import glob

pdbfile = sys.argv[1]
model = "/rf_classifier/ensemble_sig.joblib" #sys.argv[2]
#doInterpretation = int(sys.argv[3]) == 1

pdbs = glob.glob(pdbfile + "/*.pdb")
pdbs.sort()

print("Featurizing")
X_i = np.array([create_contact_features.featurize_sig(pdbi) for pdbi in pdbs])
#X_new_i = create_contact_features.featurize_sig(pdbfile)
#num_new_features = 210
print(X_i.shape)


#for i in range(210):
#	print(i+1, create_contact_features.new_index_to_interaction[i], X_new_i[i])

print("Loading Model")
clf = load(model)

print("Making prediction")
y_i = clf.predict_proba(X_i)
scores = y_i[:,1]
pred = np.mean(scores)
if pred >= 0.5: print("Binder:", pred, (np.std(scores)))
else: print("Nonbinder:", pred, (np.std(scores)))

sys.exit(0)
if not doInterpretation: sys.exit(0)

print("Computing Interpretation")
from treeinterpreter import treeinterpreter as ti

X = np.array([X_new_i])
instances = X
print(clf.predict(instances))
prediction, bias, contributions = ti.predict(clf, instances)
print(contributions.shape)

selected_rows = [0]
for i in range(len(selected_rows)):
    print("Row", selected_rows[i])
    print("Prediction:", prediction[i])
    print("Bias (trainset mean)", bias[i])
    print("Feature contributions:")
    # top positive
    contri = contributions[i, :, 1]
    sorted_indices = np.argsort(contri)
    pos_c = contri[contri>0]
    neg_c = contri[contri<0]
    for j in sorted_indices[-15:]:
        print(create_contact_features.new_index_to_interaction[j], contri[j], X[selected_rows[i], j], contri[j]/np.sum(pos_c))
    # top negative
    for j in sorted_indices[:15]:
        print(create_contact_features.new_index_to_interaction[j], contri[j], X[selected_rows[i], j], contri[j]/np.sum(neg_c))

    anchor_vals = 0
    for j in range(210):
    	r1,r2 = create_contact_features.new_index_to_interaction[j].split("-")
    	#if contri[j] < 0: continue
    	#if r1 in ["VAL","TYR"] or r2 in ["VAL","TYR"]: anchor_vals += contri[j]
    	if contri[j] > 0: continue
    	if r1 in ["TRP"] or r2 in ["TRP"]: anchor_vals += contri[j]
    #print(anchor_vals, anchor_vals/np.sum(pos_c))
    print(anchor_vals, anchor_vals/np.sum(neg_c))


print(prediction)
print(bias + np.sum(contributions, axis=1))

