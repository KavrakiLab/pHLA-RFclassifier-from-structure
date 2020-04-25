# Structure-based classification of stable peptide-HLA binding using random forests

The following describes how we produced the interpretation analysis in the paper. 
All scripts and modeling software (APE-Gen) is included inside a public-available image in Docker Hub.
Therefore, this tutorial requires installing docker.

## Structure-based classification of EVDPIGHLY onto A0101 with model interpretation

### 1. Download the image from docker hub (Warning: large image - 6.56GB)

```
docker pull jayab867/apegen:rf_classifier
```

### 2. Go to the folder in which the analysis will be performed, and create a container

This will lead you to the environment defined within the image inside a folder called `data`

```
docker run -it --rm -v $(pwd):/data --workdir "/data" jayab867/apegen:rf_classifier
```

### 3. Run APE-Gen to model conformations of the peptide-MHC of interest

This portion can take a few minutes depending on machine. For detailed instructions on using APE-Gen, please visit <https://github.com/KavrakiLab/APE-Gen>

```
mkdir wild
cd wild
python /APE-Gen/APE_Gen.py EVDPIGHLY HLA-A*01:01
cd ..
```

Inside the `wild` folder that we created, there is a folder called `0` which represents the index of the APE-Gen round. 
This is the only folder because only a single round was performed.
Inside every round-folder, all conformations in the ensemble generated are found in `full_system_confs`, which we will use at the end.
For now, we will use the best scoring conformation from the ensemble, which is in `min_energy_system.pdb`

### 4. Run the structure-based classifier

The classification script takes two inputs: 1) the location of the PDB file and 2) whether or not to perform the interpretation analysis (0 or 1).

The following will simply run a prediction. Loading the model may take a few seconds, but the prediction step is almost instant.
The output probability is a value between 0 and 1, where a value greater than 0.5 is predicted to be a stable binder.

```
python /rf_classifier/predict_pdb.py wild/0/min_energy_system.pdb 0
```

The following will run the prediction along with an interpretation analysis. 
As before, loading the model takes a few seconds, the prediction step is instant, but the interpretation step will take a few minutes.

```
python /rf_classifier/predict_pdb.py wild/0/min_energy_system.pdb 1
```

#### Interpretation Analysis

There is a bit of output, both on the terminal and in the form of CSV files.
Any numerical values may be slightly different, due to the randomness in APE-Gen.

a) Each of the 210 features (describing peptide-HLA residue-residue interactions) has a contribution value, such that their sum (plus the bias term) produces the final output probability prediction.
There are both positive values (positive contributions) and negative values (negative contributions).
All features contribute to the prediction even if their corresponding feature value is zero.
All feature contribution values are saved into `feature_contribution.csv`, and the top 5 feature contributions with a nonzero feature value is outputted to the terminal.

```
Feature Index,Interaction Type,Contribution Value,Feature Value,Pos Contribution,Neg Contribution
0,ALA-ALA,0.0008032879761508717,0.0,0.0017331196102232035,-1
1,ALA-ARG,0.0006280059761277354,0.0,0.0013549430651006295,-1
...
208,TYR-VAL,0.0086054418263837,2.9989154645067853,0.018566517147941736,-1
209,VAL-VAL,-0.002658749691519231,0.24076051634568293,-1,0.010828213989461866
```

b) Of the features that have nonzero feature values, we can attribute the contribution value towards each of the residue-residue contacts found in the conformation, since the feature values were constructed in a linear way.
For a 9-mer peptide, there are 1620 such peptide-HLA residue-residue contacts (we only use the first 180 residues in the HLA).
This contribution decomposition is saved into `contribution_decomp.csv`.
Note that sum of these values does not equal the sum of the raw feature contribution values because features with a value of zero do not have such a decomposition.

c) We can group these interactions based on their relation to the peptide or HLA positions. 
The peptide contributions are saved in `peptide_contributions.csv`

```
Position, Pos Contribution, Neg Contribution
1,0.038803568654247125,0.2541146128177117
2,0.18536158907416453,0.04342939558631472
3,0.32164454106554397,0.009031197649165115
4,0.09111192511589242,0.07684658609193454
5,0.05362416743343664,0.026439513474494195
6,0.018015466270475577,0.35339988524524546
7,0.10812721404202479,0.0005744798630328544
8,0.09955133771452278,0.10057822131490501
9,0.08376019062969214,0.13558610795719628
```

In this example, we see that positions 2 and 9 make up about 26% of the position contribution.
But interestingly, position 3 has the greatest positive contribution of 32% (it is known from IEDB that A0101 has D3 as a preferred residue).
We focus on the position contribution since this peptide-HLA was predicted to be a stable binder.

d) The mhc contributions are saved in `mhc_contributions.csv`, sorted by magnitude of contribution.

```
Position, Pos Contribution, Neg Contribution
156,0.11163705347383923,0.00017178518618544457
114,0.07637698791143187,0.008404130853982159
67,0.042622914394730683,0.036801285485007246
9,0.03667684927511764,0.028014901000500376
...
```

Here we see that position 156 of the MHC had the greatest position contribution, which was an ARG that is interacting with D3 of the peptide.

### 5. Another example with modeling a nonbinder

Here we model the same peptide with a destructive mutation in position 2 (V2W).
Before moving on, move the previous CSV files so that they are not overwritten (`mv *.csv wild/`).

So we can run APE-Gen and the classfier.

```
mkdir V2W
cd V2W
python /APE-Gen/APE_Gen.py EWDPIGHLY HLA-A*01:01
cd ..
python /rf_classifier/predict_pdb.py V2W/0/min_energy_system.pdb 1
```

From the peptide contributions, we see that position 2 is making about 42% of the negative contribution.

```
Position, Pos Contribution, Neg Contribution
1 0.1710398219791122 0.13775991391212827
2 0.047342355761051565 0.4199053966468622
3 0.17222247748998917 0.023924675507095004
4 0.1905179817891308 0.05610834696590516
5 0.03384391668464566 0.029831011627153063
6 0.030839174126260197 0.1916648925332388
7 0.1509840205349785 0.011504673906360051
8 0.1070136022818151 0.09051825881096115
9 0.09619664935301696 0.03878283009029633
```

## Classifier trained on ensemble-enriched dataset

Warning: large image - 10.9GB - and requires at least 32 GB of memory when loaded. 
The interpretation analysis has not been incorporated for this ensemble version. 
The script takes a single input: the location of the folder with the ensemble

In a particular folder, run the following to perform a classification using our example from above.

```
docker pull jayab867/apegen:rf_classifier_ensemble
docker run -it --rm -v $(pwd):/data --workdir "/data" jayab867/apegen:rf_classifier_ensemble
python /rf_classifier/predict_ensemble.py wild/0/full_system_confs
```

The output probability (defined as the mean output probability across the ensemble) is printed along with the standard deviation.

## Training the model

The script used to generate the models can be found in `model-creation/build_model.py` and `model-creation/ensemble_build_model.py`.
All cross-validation and leave-one-allele-out tests can be found in those scripts as well.
The raw APE-Gen output along with the featurized conformations is available upon request.


