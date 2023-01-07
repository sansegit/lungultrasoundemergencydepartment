# The Role of Lung Ultrasound in the Emergency Department with Respect to Clinical Variables, Chest X-ray and Chest Computed Tomography: a Machine Learning Study of Patient Outcomes
Supporting code and data for publication.


# Description
The code provides an objective comparison of the capability of different sets of features obtained from clinical and imaging reports to predict clinical outcomes.

Based on a dataframe stored in a *.csv file, the code performs machine learning with a stratified cross-validation approach.
Each row of the *.csv file corresponds to a separate patient instance. Each column contains features used for training the model (binary, integer), together with prediction labels (binary).

The model evaluates best performing machine learning architectures for a set of input features with respect to  binary prediction outcomes.
By performing a comprehensive search of machine learning architectures and hyperparameters, the best performing architecture is identified for each input feature set.
Models include: K-Neighbors, Bayes, Support Vector Machines, Decision Trees, Random Forests, Multi-layer Perceptron, Gaussian Processes, Ada Boost, Quadratic Discriminant Analysis, Logistic Regression, Ridge Classifier, Bagged Decision Trees and Stochastic Gradient Boosting. Hyperparameter optimization is performed with grid search, with 191 models tested in total per feature set.

The code allows comparing predictive performance for different input data feature sets, applying a consistent data split in all cases.
For each input data feature set, a best performing machine learning architecture is identified. 
The code allow tracing one-by-one prediction outcomes for a specific patient instance based on different feature sets.

Unpaired (Spearman's rank correlation coefficient and AUC expressed in terms of mean and standard deviation for cross-validation set) 
and paired (McNemar method) statistics allow assessing performance difference among different feature sets.

## Installation
The code runs in Python 3.8.5 using the open-source machine learning library Scikit-learn 1.02., together with additional libraries NumPy, Pandas, Matplotlib and Plotly for data structure manipulation and results visualization.

## Usage



# Data
In the reference citation, the code is applied to predict clinical outcomes (hospital admission, 2-month mortality and positive SARS-CoV-2 test) in patients presenting with acute respiratory distress ARD in the emergency department (ED).

For this purpose, feature sets obtained from electrical medical reports (EMR), together with reports of lung ultrasound (LUS), chest X-ray and computed tomography (CT) are compared.

The code compares performance differences between each individual set of features and between combinations of EMR with either LUS, X-ray or CT features.

The *.csv file with data utilized is provided in /data

# Authors and acknowledgment
The author list for this work is:
Sanabria SJ^{a,b,*},
Antil N^{b},
Trieu A^{b},
Lutz A^{b},
Dahl J^{b},
Kamaya A^{b},
Anderson K L^{c},
ElKaffas A^{b},

The participating institutions are: 
{a} Deusto Institute of Technology, University of Deusto/ IKERBASQUE, Basque Foundation for Science, Bilbao, Spain.
{b} Department of Radiology, Stanford University, 3155 Porter Drive, Stanford, CA 94305 USA.
{c} Department of Emergency Medicine, Stanford University, 900 Welch Road Ste. 350, Palo Alto, CA 94304 USA.

*Corresponding author: E-mail: sanse@stanford.edu


# License
TO BE DEFINED (open-source software).

Please cite the original work in:
1. Sanabria SJ, Antil N, Trieu A, Lutz A, Dahl J, Kamaya A, Anderson K L, ElKaffas, A. The Role of Lung Ultrasound in the Emergency Department with Respect to Clinical Variables, Chest X-ray and Chest Computed Tomography: a Machine Learning Study of Patient Outcomes. 

