#### Includes all variables necessary for configuration of the code

WHERE_IS_MY_DATA = "./data/LUS_EC_data.csv" # Where is *.csv file for data processing

N_RANDOM_SEEDS = 5 # number of random seeds for repeated analyses

OUTCOME_LIST = ["OUTCOME","RT_PCR_POSITIVE_NEGATIVE","Deceased"] 
OUTCOME_LIST_VERBOSE = ["Hospital admission", "SARS_CoV_2_positive", "2-month mortality"] # For table generation purposes
# Clinical outcome list (hospital admission, SARS_CoV_2 positive, 2-month mortality)

FEATURE_SET_LIST = ["Clinical", "LUS", "X-ray", "CT", "Clinical+LUS", "Clinical+X-ray", "Clinical+CT"] # Feature sets to be compared
## Requires corresponding definition in lung_ultrasound_funs.selectDesiredData of to indicate how to select table columns in each case

REFERENCE_FEATURE_MCNEMAR =  "LUS" # Which feature index from FEATURE_SET_LIST do we use as a reference for McNemar tests
COMPARISON_FEATURE_MCNEMAR = ["X-ray", "CT"] # List of features for which McNemar test is performed

INDEX_COL = "S.NO " # Column containing patient indices in increasing order

PHASE_0_ACTIVE = True   ## Calculates correlations between outcomes and individual features
PHASE_1_ACTIVE = True  ## Trains machine learning models
PHASE_2_ACTIVE = True  ## Determines best machine learning model for each feature set, computes extended statistics, including confusion matrices and AUROC (requires output of PHASE_2 to run) (requires output of PHASE_1_ to run)
PHASE_3_ACTIVE = True   ## Performs McNemar paired comparison test (requires output of PHASE_2 to run)
PHASE_4_ACTIVE = True  ## Generates results figures for manuscript (requires output of PHASE_3 and PHASE_2 to run)
