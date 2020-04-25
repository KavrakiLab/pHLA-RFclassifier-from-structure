import mdtraj as md
import sys
import numpy as np
import itertools

one_letter_code = {'ARG':'R', 'HIS':'H', 'LYS':'K', 'ASP':'D', 'GLU':'E', \
                  'SER':'S', 'THR':'T', 'ASN':'N', 'GLN':'Q', 'CYS':'C', \
                  'GLY':'G', 'PRO':'P', 'ALA':'A', 'VAL':'V', 'ILE':'I', \
                  'LEU':'L', 'MET':'M', 'PHE':'F', 'TYR':'Y', 'TRP':'W'}

three_letter_code  = {v: k for k, v in one_letter_code.items()}

residue_names = [k for k in one_letter_code.keys()]
residue_names.sort()

"""
interaction_to_index = {}
index_to_interaction = []
interaction_index = 0
for i in range(len(residue_names)):
	for j in range(len(residue_names)):
		i_to_j = residue_names[i] + "-" + residue_names[j]
		interaction_to_index[i_to_j] = interaction_index
		interaction_index += 1
		index_to_interaction.append(i_to_j)

#print(len(interaction_to_index.keys()))
num_dim = len(index_to_interaction)
#print(num_dim)
"""

new_interaction_to_index = {}
new_index_to_interaction = []
new_interaction_index = 0
for i in range(len(residue_names)):
	for j in range(i, len(residue_names)):
		i_to_j = residue_names[i] + "-" + residue_names[j]
		j_to_i = residue_names[j] + "-" + residue_names[i]
		new_interaction_to_index[i_to_j] = new_interaction_index
		new_interaction_to_index[j_to_i] = new_interaction_index
		new_interaction_index += 1
		new_index_to_interaction.append(i_to_j)

#print(len(interaction_to_index.keys()))
num_dim = len(new_index_to_interaction)
#print(num_dim)

"""
def vector_to_matrix(vec):

	num_residues = len(one_letter_code.keys())
	mat = []
	start_index = 0
	for i in range(num_residues):
		mat.append(vec[start_index:(start_index+num_residues)])
		start_index += num_residues
	return np.array(mat)
"""

def get_net_contribution(resres_names, contri_per_distance, restype, resnum):

    if restype == "pep": resindex = 0
    elif restype == "mhc": resindex = 1

    index_vec = np.array([r[resindex][3:] for r in resres_names])
    #overall_score = np.sum(contri_per_distance[index_vec == str(resnum)]) / np.sum(contri_per_distance)
    pos_score = np.sum(contri_per_distance[np.logical_and(index_vec == str(resnum), contri_per_distance > 0)]) / np.sum(contri_per_distance[contri_per_distance > 0])
    neg_score = np.sum(contri_per_distance[np.logical_and(index_vec == str(resnum), contri_per_distance < 0)]) / np.sum(contri_per_distance[contri_per_distance < 0])

    return pos_score, neg_score

def decompose_contributions(resres_names, distances, contributions, featurize_type):

    feature_vec = featurize_from_distances(resres_names, distances, featurize_type)

    contri_per_distance = []
    weights = []
    for i in range(len(distances)):
        r1 = resres_names[i][0][:3]
        r2 = resres_names[i][1][:3]
        d = distances[i]
        if featurize_type == "reg": unnormalized_weight = 1./(d)
        elif featurize_type == "r2": unnormalized_weight = 1./(d*d)
        elif featurize_type == "sig": unnormalized_weight = 1./(1 + np.exp(d - 5.0))
        weight = unnormalized_weight / feature_vec[new_interaction_to_index[r1+"-"+r2]]
        contri_per_distance.append(weight * contributions[new_interaction_to_index[r1+"-"+r2]])
        weights.append(weight)

    return np.array(contri_per_distance), np.array(weights)

def get_distances(filename):

    conf = md.load(filename)

    num_peptide_residues = len(conf.top.select("name == CA and chainid == 2"))  
    num_binding_site_residues = 180
    num_total_residues = len(conf.top.select("name == CA"))

    peptide_indices = range(num_total_residues)[-num_peptide_residues:]
    binding_site_indices = range(num_binding_site_residues)

    contacts_all = list(itertools.product(peptide_indices, binding_site_indices))
    #print("Num contacts:", len(contacts_all))

    distances, pairs = md.compute_contacts(conf, contacts_all)
    distances = distances[0] # only a single frame

    feature_vec = np.zeros((num_dim,))

    residues = [r for r in conf.top.residues]
    resres_names = []
    pep_mhc_distances = []
    for i,p in enumerate(pairs):
        r1 = str(residues[p[0]])
        r2 = str(residues[p[1]])
        resres_names.append([r1,r2])
        pep_mhc_distances.append(distances[i]*10)
        #feature_vec[new_interaction_to_index[r1+"-"+r2]] += 1./(distances[i]*10)
        #if abs(distances[i]) < 0.001: print(r1, r2, distances[i], p)

    return resres_names, pep_mhc_distances

def featurize_from_distances(resres_names, distances, featurize_type):

    feature_vec = np.zeros((num_dim,))

    for i in range(len(distances)):
        r1 = resres_names[i][0][:3]
        r2 = resres_names[i][1][:3]
        d = distances[i]
        if featurize_type == "reg": feature_vec[new_interaction_to_index[r1+"-"+r2]] += 1./(d)
        elif featurize_type == "r2": feature_vec[new_interaction_to_index[r1+"-"+r2]] += 1./(d*d)
        elif featurize_type == "sig": feature_vec[new_interaction_to_index[r1+"-"+r2]] += 1./(1 + np.exp(d - 5.0))
        #if abs(d) < 0.001: print(r1, r2, d)

    return feature_vec

def featurize(filename):

	conf = md.load(filename)

	num_peptide_residues = len(conf.top.select("name == CA and chainid == 2"))	
	num_binding_site_residues = 180
	num_total_residues = len(conf.top.select("name == CA"))

	peptide_indices = range(num_total_residues)[-num_peptide_residues:]
	binding_site_indices = range(num_binding_site_residues)

	contacts_all = list(itertools.product(peptide_indices, binding_site_indices))
	#print("Num contacts:", len(contacts_all))

	distances, pairs = md.compute_contacts(conf, contacts_all)
	distances = distances[0] # only a single frame

	feature_vec = np.zeros((num_dim,))

	residues = [r for r in conf.top.residues]
	for i,p in enumerate(pairs):
		r1 = str(residues[p[0]])[:3]
		r2 = str(residues[p[1]])[:3]
		feature_vec[new_interaction_to_index[r1+"-"+r2]] += 1./(distances[i]*10)
		if abs(distances[i]) < 0.001: print(r1, r2, distances[i], p)

	return feature_vec

def featurize_r2(filename):

        conf = md.load(filename)

        num_peptide_residues = len(conf.top.select("name == CA and chainid == 2"))
        num_binding_site_residues = 180
        num_total_residues = len(conf.top.select("name == CA"))

        peptide_indices = range(num_total_residues)[-num_peptide_residues:]
        binding_site_indices = range(num_binding_site_residues)

        contacts_all = list(itertools.product(peptide_indices, binding_site_indices))
        #print("Num contacts:", len(contacts_all))

        distances, pairs = md.compute_contacts(conf, contacts_all)
        distances = distances[0] # only a single frame

        feature_vec = np.zeros((num_dim,))

        residues = [r for r in conf.top.residues]
        for i,p in enumerate(pairs):
                r1 = str(residues[p[0]])[:3]
                r2 = str(residues[p[1]])[:3]
                resres_dist = distances[i]*10
                feature_vec[new_interaction_to_index[r1+"-"+r2]] += 1./(resres_dist*resres_dist)
                if abs(distances[i]) < 0.001: print(r1, r2, distances[i], p)

        return feature_vec

def featurize_sig(filename):

        conf = md.load(filename)

        num_peptide_residues = len(conf.top.select("name == CA and chainid == 2"))
        num_binding_site_residues = 180
        num_total_residues = len(conf.top.select("name == CA"))

        peptide_indices = range(num_total_residues)[-num_peptide_residues:]
        binding_site_indices = range(num_binding_site_residues)

        contacts_all = list(itertools.product(peptide_indices, binding_site_indices))
        #print("Num contacts:", len(contacts_all))

        distances, pairs = md.compute_contacts(conf, contacts_all)
        distances = distances[0] # only a single frame

        feature_vec = np.zeros((num_dim,))

        residues = [r for r in conf.top.residues]
        for i,p in enumerate(pairs):
                r1 = str(residues[p[0]])[:3]
                r2 = str(residues[p[1]])[:3]
                resres_dist = distances[i]*10
                alpha = 5 #np.log(99)+4
                feature_vec[new_interaction_to_index[r1+"-"+r2]] += 1./(1 + np.exp(resres_dist - alpha))
                if abs(distances[i]) < 0.001: print(r1, r2, distances[i], p)

        return feature_vec

# NOT USED
def get_dists(filename):

        conf = md.load(filename)

        num_peptide_residues = len(conf.top.select("name == CA and chainid == 2"))
        num_binding_site_residues = 180
        num_total_residues = len(conf.top.select("name == CA"))

        peptide_indices = range(num_total_residues)[-num_peptide_residues:]
        binding_site_indices = range(num_binding_site_residues)

        contacts_all = list(itertools.product(peptide_indices, binding_site_indices))
        #print("Num contacts:", len(contacts_all))

        distances, pairs = md.compute_contacts(conf, contacts_all)
        distances = distances[0] # only a single frame

        interactions = []
        resres_dists = []
        residues = [r for r in conf.top.residues]
        for i,p in enumerate(pairs):
                r1 = str(residues[p[0]])[:3]
                r2 = str(residues[p[1]])[:3]
                interactions.append(r1+"-"+r2)
                resres_dists.append(distances[i]*10)

        return interactions, resres_dists




#conf = md.load(sys.argv[1])
#feature_vec = featurize(conf)
#print(feature_vec)




