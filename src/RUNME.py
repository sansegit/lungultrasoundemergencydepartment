from imports import *
from models import *
import config
import stats_funs
import lung_ultrasound_funs


random_seeds = np.arange(0,config.N_RANDOM_SEEDS)  

### Getting main code started
warnings.filterwarnings("ignore")

if config.PHASE_0_ACTIVE == True:  
    ###### Calculation of correlation coefficients between individual features and outcomes #######
    df = pd.read_csv(config.WHERE_IS_MY_DATA)

    # Find most predictive attributes
    df1 = df.copy()
    for OUTCOME in config.OUTCOME_LIST:
        df1 = df1.drop(columns = OUTCOME)
    df1 = lung_ultrasound_funs.preprocessDF_CorrelationAnalysis(df1)  
    for OUTCOME in config.OUTCOME_LIST:
        cor_support1, cor_feature1, cor_values1, p_values1 = stats_funs.cor_selector(df1, df[OUTCOME], len(df1.columns))
        correlation_results = {"cor_support1": cor_support1, "cor_feature": cor_feature1, "cor_values": cor_values1, "p_values": p_values1}
        np.save("./Results/correlation_results_" + OUTCOME + ".npy", correlation_results, allow_pickle = True)

if config.PHASE_1_ACTIVE == True:

    df = pd.read_csv(config.WHERE_IS_MY_DATA)

    # Prepare data frame for ML models 
    df1 = df.copy()

    for OUTCOME in config.OUTCOME_LIST:
        df1 = df1.drop(columns = OUTCOME)
    df1 = lung_ultrasound_funs.preprocessDF_MachineLearning(df1)



    ## LIST OF OUTCOMES WE WANT TO VERIFY
    data ={'Classifier': names}
    classifiers = generateClassifiers(0)
    columns_classifieroutputs = [config.INDEX_COL]
    for outcome in config.OUTCOME_LIST:
        for indexclassifier in range(len(classifiers)):
            columns_classifieroutputs .append(outcome[0:4] + "_" + str(indexclassifier))
            columns_classifieroutputs .append(outcome[0:4] + "_p_" + str(indexclassifier))

    dfclassifieroutputs = pd.DataFrame(np.zeros([len(df1[config.INDEX_COL]), len(columns_classifieroutputs)])*np.nan, 
        columns = columns_classifieroutputs)
    dfclassifieroutputs[config.INDEX_COL] = df1[config.INDEX_COL]

    for seed in random_seeds:

        classifiers = generateClassifiers(seed*42)

        for desireddata in range(1, len(config.FEATURE_SET_LIST) + 1):  #1 - Amnamesis, 2 - Ultrasound, 3 - X-ray, 4- CT, 5- Amnamesis + LUS, 6 - Amnamesis + X-ray, 7 - Amnamesis + CT 
            #desireddata = 2
            #print("Data is " + str(desireddata))
            dfsummary = pd.DataFrame(data)    # dfsummary gives us the quality statistics of a specific classifier, for all classification outcomes, for a single input data type
            mdfclassifieroutputs = dfclassifieroutputs.copy()
            # somehow we need to store the classifier outputs for each patient in an organized way (rows = patients, columns = classifiers for each outcome OUTCOME_1, OUTCOME_2 .... and so on)                                    
            
            for outcome in config.OUTCOME_LIST: # iterate in outcomes
                #print("OUTCOME")
                cor_valuesm_list = []
                mcc_valuesm_list = []
                p_valuesm_list = []
                df2 = df1.copy() #[cor_feature1] ## most correlated variable
                # Select desired data
                df2 = lung_ultrasound_funs.selectDesiredData(df2, config.FEATURE_SET_LIST[desireddata-1]) 
                df2.insert(0, outcome, df[outcome], True)
                df2 = df2.dropna() # drop empty rows  - 79 rows in total
                y = df2[outcome]
                df2 = df2.drop(columns = outcome)
                ### Now we can do a random forest classifier
                X = df2.copy()
                ## Strafied cross validation
                skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = seed*42)
                skf.get_n_splits(X,y)
                index_classifier = 0
                for name, clf in zip(names, classifiers): # iterate in classifiers
                    counter = 0
                    #print("CLASSIFIER:")
                    for train_index, test_index in skf.split(X, y):  # iterate in cross-validation folds
                        #print("TEST:", test_index)
                        #print("TRAIN:", train_index, "TEST:", test_index)
                        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                        X_train = X_train.drop(columns = config.INDEX_COL) 
                        SNO_test = X_test[config.INDEX_COL] # patient identifier
                        X_test = X_test.drop(columns = config.INDEX_COL)
                        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                        # Train the model using the training sets
                        clf.fit(X_train, y_train)
                        y_pred = clf.predict(X_test)
                        try:
                            y_proba = clf.decision_function(X_test)
                        except:
                            try:
                                y_proba = clf.predict_proba(X_test)[:,1]
                            except:
                                y_proba = np.zeros(y_pred.shape)*np.nan
                        mdfclassifieroutputs.loc[SNO_test - 1, outcome[0:4] + "_" + str(index_classifier)] = y_pred
                        mdfclassifieroutputs.loc[SNO_test - 1, outcome[0:4] + "_p_" + str(index_classifier)] = y_proba
                        X_test.insert(0,"OUTCOME_PRED", y_pred)
                        X_test.insert(0,"OUTCOME", y_test)
                        if counter == 0:
                            dfout = pd.DataFrame(X_test)
                        else:
                            dfout = dfout.append(X_test)
                        counter += 1
                    yout = dfout["OUTCOME"]
                    dfout = dfout.drop(columns = "OUTCOME")
                    #matrix = confusion_matrix(yout, dfout["OUTCOME_PRED"])
                    #report = classification_report(yout, dfout["OUTCOME_PRED"])
                    cor_valuesm, p_valuesm = stats_funs.cor_calculator(dfout["OUTCOME_PRED"], yout)
                    ypred = np.array(mdfclassifieroutputs[outcome[0:4] + "_" + str(index_classifier)])
                    ytrue = np.array(df[outcome])
                    # Get rid of NaNs
                    yprednonan = ypred[~np.isnan(ypred) & ~np.isnan(ytrue) ]
                    ytruenonan = ytrue[~np.isnan(ypred) & ~np.isnan(ytrue) ]                
                    mcc = matthews_corrcoef(ytruenonan, yprednonan)
                    mcc_valuesm_list.append(mcc)
                    cor_valuesm_list.append(cor_valuesm)
                    p_valuesm_list.append(p_valuesm) #p_valuesm
                    index_classifier += 1
                dfsummary['Corr: ' + outcome[0:4]] = cor_valuesm_list
                dfsummary['p-v: ' + outcome[0:4]] = p_valuesm_list
                dfsummary['MCC: ' + outcome[0:4]] = mcc_valuesm_list
                mdfclassifieroutputs[outcome[0:4]] = df[outcome]
            print(dfsummary)
            dfsummary.to_csv("./results/dfsummary_" + str(desireddata) + "_seed" + str(seed) + ".csv")

            mdfclassifieroutputs.to_csv("./results/dfclassifieroutputs_" + str(desireddata) + "_seed" + str(seed) + ".csv")
    print("PHASE 1 COMPLETED")




if config.PHASE_2_ACTIVE == True:
    ### Once we reach here, we start analyzing the data for conclusions on best classifier parameters. For these, we compute more detailed metrics

    order_according = "Corr: " # "MCC: "
    for seed in random_seeds:
        list_dfsummary = []
        for desireddata in range(1, len(config.FEATURE_SET_LIST) + 1):
            dfsummary = pd.read_csv("./results/dfsummary_" + str(desireddata)+ "_seed" + str(seed)  + ".csv")
            mdfclassifieroutputs = pd.read_csv("./results/dfclassifieroutputs_" + str(desireddata) + "_seed" + str(seed) + ".csv") #+ "_seed" + str(seed) + 
            ## We get the maximum classifier score for each classifier type
            listClassifiers = dfsummary["Classifier"].str[:-3].drop_duplicates() # find unique list of methods
            listDataFramesOutcomes = []
            for outcome in config.OUTCOME_LIST:
                columns_outcome = ["Classifier " + outcome[0:4], "MCC: " + outcome[0:4], "Corr: " + outcome[0:4], "p-v: " + outcome[0:4], "Index: "+ outcome[0:4]]
                dfsummary_classifier = pd.DataFrame(np.zeros([len(listClassifiers), len(columns_outcome)])*np.nan,
                columns = columns_outcome)
                index_classifier = 0
                for classifier in listClassifiers:
                    dfsummary_classifier["Classifier " + outcome[0:4]].iloc[index_classifier] = classifier
                    ## select rows corresponding to this classifier
                    contain_rows = dfsummary[dfsummary["Classifier"].str.contains(classifier)]
                    bestClassifierSelection = contain_rows[contain_rows[order_according + outcome[0:4]] ==  contain_rows[order_according + outcome[0:4]].max()]
                    if len(bestClassifierSelection ) == 0: # no available data
                        index_classifier += 1
                        continue
                    bestClassifier = bestClassifierSelection["Classifier"].iloc[0]
                    dfsummary_classifier["Classifier " + outcome[0:4]].iloc[index_classifier] = bestClassifier
                    indexBestClassifier = names.index(bestClassifier)
                    #ypred = np.array(mdfclassifieroutputs[outcome[0:4] + "_" + str(indexBestClassifier)])
                    #yref = np.array(mdfclassifieroutputs[outcome[0:4]])
                    ### Now we can start computing metrics such as accuracy and so on... (for the summary just r and p-v)
                    dfsummary_classifier["Corr: " + outcome[0:4]].iloc[index_classifier] = bestClassifierSelection["Corr: " + outcome[0:4]].iloc[0]
                    dfsummary_classifier["MCC: " + outcome[0:4]].iloc[index_classifier] = bestClassifierSelection["MCC: " + outcome[0:4]].iloc[0]
                    dfsummary_classifier["p-v: " + outcome[0:4]].iloc[index_classifier] = bestClassifierSelection["p-v: " + outcome[0:4]].iloc[0]
                    dfsummary_classifier["Index: " + outcome[0:4]].iloc[index_classifier] = indexBestClassifier
                    index_classifier +=1
                # Here we can do some sorting of the table in decreasing order of MCC
                dfsummary_classifier = dfsummary_classifier.sort_values(by=[order_according + outcome[0:4]], ascending = False).reset_index(drop=True)
                listDataFramesOutcomes.append(dfsummary_classifier)
            # Here we can append all this dataframes in one and add a column for desiredata
            dfsummary_classifier_outcomes = pd.concat(listDataFramesOutcomes, axis=1)
            dfsummary_classifier_outcomes["Representation"] = desireddata
            list_dfsummary.append(dfsummary_classifier_outcomes)
        dfsummary_representations = pd.concat(list_dfsummary, axis = 0)
        dfsummary_representations.to_csv("./results/summary_of_representations" + "_seed" + str(seed) + ".csv") #+ "_seed" + str(seed) 



    for seed in random_seeds:
        ### Visualize confusion matrices and ROC curves for top classifiers
        dfsummary_representations = pd.read_csv("./results/summary_of_representations" + "_seed" + str(seed) + ".csv")
        dfsummary_representations_best = dfsummary_representations.iloc[0::len(listClassifiers)]
        ### Find each model and compute confusion matrix and ROC_CURVE
        listDataFramesOutcomes = []
        for outcome in config.OUTCOME_LIST:
            columns_outcome = ["Representation","AUROC: " + outcome[0:4], "MCC: " + outcome[0:4], "Bal.Acc: " + outcome[0:4], "p-v: " + outcome[0:4], "Corr: " + outcome[0:4],
            "CM[0,0]: " + outcome[0:4], "CM[0,1]: " + outcome[0:4], "CM[1,0]: " + outcome[0:4], "CM[1,1]: "+ outcome[0:4]]
            dfconfmatrix_representations = pd.DataFrame(np.zeros([7, len(columns_outcome)])*np.nan,
            columns = columns_outcome)
            for desireddata in range(1, len(config.FEATURE_SET_LIST) + 1):
                mdfclassifieroutputs = pd.read_csv("./results/dfclassifieroutputs_" + str(desireddata) + "_seed" + str(seed) +  ".csv")
                indexBestClassifier = dfsummary_representations_best["Index: "+ outcome[0:4]].iloc[desireddata-1]
                # Load ypred and ytest ... 
                ypred = np.array(mdfclassifieroutputs[outcome[0:4] + "_" + str(int(indexBestClassifier))])
                ypredproba = np.array(mdfclassifieroutputs[outcome[0:4] + "_p_" + str(int(indexBestClassifier))])
                ytrue = np.array(mdfclassifieroutputs[outcome[0:4]])
                # Get rid of NaNs
                yprednonan = ypred[~np.isnan(ypred) & ~np.isnan(ytrue) & ~np.isnan(ypredproba)]
                ypredprobanonan = ypredproba[~np.isnan(ypred) & ~np.isnan(ytrue) & ~np.isnan(ypredproba)]
                ytruenonan = ytrue[~np.isnan(ypred) & ~np.isnan(ytrue) & ~np.isnan(ypredproba)]
                confmatrix = confusion_matrix(ytruenonan, yprednonan)
                balancedaccuracy = balanced_accuracy_score(ytruenonan, yprednonan)
                mcc = matthews_corrcoef(ytruenonan, yprednonan)
                try:
                    auroc = roc_auc_score(ytruenonan,ypredprobanonan) 
                except:
                    auroc = 0
                corr, p_corr = stats_funs.cor_calculator(yprednonan, ytruenonan)

                try:
                    dfconfmatrix_representations["CM[0,0]: " + outcome[0:4]].iloc[desireddata-1] = confmatrix[0,0]
                    dfconfmatrix_representations["CM[0,1]: " + outcome[0:4]].iloc[desireddata-1] = confmatrix[0,1]
                    dfconfmatrix_representations["CM[1,0]: " + outcome[0:4]].iloc[desireddata-1] = confmatrix[1,0]
                    dfconfmatrix_representations["CM[1,1]: " + outcome[0:4]].iloc[desireddata-1] = confmatrix[1,1]
                    dfconfmatrix_representations["Representation"].iloc[desireddata-1] = desireddata
                    dfconfmatrix_representations["Bal.Acc: " + outcome[0:4]].iloc[desireddata-1] = balancedaccuracy
                    dfconfmatrix_representations["Corr: " + outcome[0:4]].iloc[desireddata-1] = corr
                    dfconfmatrix_representations["p-v: " + outcome[0:4]].iloc[desireddata-1] = p_corr
                    dfconfmatrix_representations["MCC: " + outcome[0:4]].iloc[desireddata-1] = mcc
                    dfconfmatrix_representations["AUROC: " + outcome[0:4]].iloc[desireddata-1] = auroc
                except:
                    dfconfmatrix_representations["CM[0,0]: " + outcome[0:4]].iloc[desireddata-1] = np.NaN
                    dfconfmatrix_representations["CM[0,1]: " + outcome[0:4]].iloc[desireddata-1] = np.NaN
                    dfconfmatrix_representations["CM[1,0]: " + outcome[0:4]].iloc[desireddata-1] = np.NaN
                    dfconfmatrix_representations["CM[1,1]: " + outcome[0:4]].iloc[desireddata-1] = np.NaN
                    dfconfmatrix_representations["Representation"].iloc[desireddata-1] = np.NaN
                    dfconfmatrix_representations["Bal.Acc: " + outcome[0:4]].iloc[desireddata-1] = np.NaN
                    dfconfmatrix_representations["Corr: " + outcome[0:4]].iloc[desireddata-1] = np.NaN
                    dfconfmatrix_representations["p-v: " + outcome[0:4]].iloc[desireddata-1] = np.NaN
                    dfconfmatrix_representations["MCC: " + outcome[0:4]].iloc[desireddata-1] = np.NaN
                    dfconfmatrix_representations["AUROC: " + outcome[0:4]].iloc[desireddata-1] = np.NaN


            listDataFramesOutcomes.append(dfconfmatrix_representations)
        dfconfmatrix_representations_outcomes = pd.concat(listDataFramesOutcomes, axis=1)    
        dfconfmatrix_representations_outcomes.to_csv("./results/confusionmatrix_representations" + "_seed" + str(seed) + ".csv")

    ### Build summary per seeds to identify best performing iterations and average statistics
    list_dfconfmatrix_representations_outcomes_seeds = []
    #pd.DataFrame(np.zeros([7*len(random_seeds), len(columns_outcome)])*np.nan
    for desireddata in range(1, len(config.FEATURE_SET_LIST) + 1):
        for seed in random_seeds:
            dfconfmatrix_representations_outcomes = pd.read_csv("./results/confusionmatrix_representations" + "_seed" + str(seed) + ".csv")
            list_dfconfmatrix_representations_outcomes_seeds.append(dfconfmatrix_representations_outcomes[:].iloc[desireddata-1])
    dfconfmatrix_representations_outcomes_seeds = pd.concat(list_dfconfmatrix_representations_outcomes_seeds, axis = 1).transpose()
    dfconfmatrix_representations_outcomes_seeds.to_csv("./results/confusionmatrix_representations_allseeds.csv")
    print("PHASE 2 COMPLETED")






if config.PHASE_3_ACTIVE:

    columns_mcnemar = ["Representation"]
    for outcome in config.OUTCOME_LIST:
        columns_mcnemar.append("OR: " + outcome[0:4])
        columns_mcnemar.append("p-v: " + outcome[0:4])

    df_mcnemar = pd.DataFrame(np.zeros([len(config.COMPARISON_FEATURE_MCNEMAR), len(columns_mcnemar)])*np.nan,
    columns = columns_mcnemar)

    ######### PERFORM McNemarTest for Amnamesis - you can group all repetitions together and average by 5 to obtain a global score
    for outcome in config.OUTCOME_LIST:
        ypred_desireddata_list = []
        ytrue_desireddata_list = []
        for desireddata in range(1, len(config.FEATURE_SET_LIST) + 1):
            ypred_list = []
            ytrue_list = []
            for seed in random_seeds:
                dfsummary = pd.read_csv("./results/dfsummary_" + str(desireddata)+ "_seed" + str(seed)  + ".csv")
                mdfclassifieroutputs = pd.read_csv("./results/dfclassifieroutputs_" + str(desireddata) + "_seed" + str(seed) + ".csv") #+ "_seed" + str(seed) + 
                ## We get the maximum classifier score for each classifier type
                listClassifiers = dfsummary["Classifier"].str[:-3].drop_duplicates() # find unique list of methods

                ### Visualize ROC curves for top classifiers
                dfsummary_representations = pd.read_csv("./results/summary_of_representations" + "_seed" + str(seed) + ".csv")
                dfsummary_representations_best = dfsummary_representations.iloc[0::len(listClassifiers)]
                ### Find each model and compute confusion matrix and ROC_CURVE
                mdfclassifieroutputs = pd.read_csv("./results/dfclassifieroutputs_" + str(desireddata) + "_seed" + str(seed) +  ".csv")
                indexBestClassifier = dfsummary_representations_best["Index: "+ outcome[0:4]].iloc[desireddata-1]
                # Load ypred and ytest ... 
                ypred = np.array(mdfclassifieroutputs[outcome[0:4] + "_" + str(int(indexBestClassifier))])
                ypredproba = np.array(mdfclassifieroutputs[outcome[0:4] + "_p_" + str(int(indexBestClassifier))])
                ytrue = np.array(mdfclassifieroutputs[outcome[0:4]])
                # Group repetitions into one
                ypred_list.append(ypred)
                ytrue_list.append(ytrue)
            ypred_seed = np.concatenate(ypred_list).ravel()
            ytrue_seed = np.concatenate(ytrue_list).ravel()
            ypred_desireddata_list.append(ypred_seed)
            ytrue_desireddata_list.append(ytrue_seed)
        
        
        ## Here we start McNemar test ### 
        referenceIndexMcNemar = (config.FEATURE_SET_LIST).index(config.REFERENCE_FEATURE_MCNEMAR)
        comparisonIndexMcNemar = []
        for comparisonFeature in config.COMPARISON_FEATURE_MCNEMAR:
            comparisonIndexMcNemar.append((config.FEATURE_SET_LIST).index(comparisonFeature))



        
        index = 0
        for desireddata in comparisonIndexMcNemar: # We perform all tests with respect to clinical features
            labels2 = ytrue_desireddata_list[referenceIndexMcNemar]
            labels1 = ytrue_desireddata_list[desireddata]
            predictions2 = ypred_desireddata_list[referenceIndexMcNemar]
            predictions1 = ypred_desireddata_list[desireddata]
            labels2nonan = labels2[~np.isnan(labels1) & ~np.isnan(labels2) & ~np.isnan(predictions1) & ~np.isnan(predictions2)]
            labels1nonan = labels1[~np.isnan(labels1) & ~np.isnan(labels2) & ~np.isnan(predictions1) & ~np.isnan(predictions2)]
            predictions2nonan = predictions2[~np.isnan(labels1) & ~np.isnan(labels2) & ~np.isnan(predictions1) & ~np.isnan(predictions2)]
            predictions1nonan = predictions1[~np.isnan(labels1) & ~np.isnan(labels2) & ~np.isnan(predictions1) & ~np.isnan(predictions2)]
            [results, p_value] = stats_funs.performMcNemarTest (labels1nonan, predictions1nonan, labels2nonan, predictions2nonan, "./results/McNemar_" + str(desireddata+1) + "_ref" + str(referenceIndexMcNemar+1) + "_" + outcome + '.txt')
            df_mcnemar["Representation"][index] = config.COMPARISON_FEATURE_MCNEMAR[index]
            df_mcnemar["OR: " + outcome[0:4]][index] = results
            df_mcnemar["p-v: " + outcome[0:4]][index] = p_value
            index = index + 1

        df_mcnemar.to_csv("./results/mcnemar_test.csv")


        ## Build contigency matrices
        ## Calculate odds ratios
        ## Calculate p-values
        # Get rid of NaNs
        #yprednonan = ypred[~np.isnan(ypred) & ~np.isnan(ytrue) & ~np.isnan(ypredproba)]
        #ypredprobanonan = ypredproba[~np.isnan(ypred) & ~np.isnan(ytrue) & ~np.isnan(ypredproba)]
        #ytruenonan = ytrue[~np.isnan(ypred) & ~np.isnan(ytrue) & ~np.isnan(ypredproba)]
    print("PHASE 3 COMPLETED")




if config.PHASE_4_ACTIVE:    

    ## Generate Table 4 - Most predictive attributes
    table_1 = texttable.Texttable()
    table_1.set_cols_align(["c" , "c"])
    table_1.set_cols_valign(["m", "m"] )
    
    index_outcome = 0
    for outcome in config.OUTCOME_LIST:
        table_1.add_row([config.OUTCOME_LIST_VERBOSE[index_outcome], "r_s (p_value)"])
        data = np.load("./Results/correlation_results_" + outcome + ".npy",allow_pickle = True)
        corrv = data.flatten()[0]["cor_values"]
        featurev = data.flatten()[0]["cor_feature"]
        pv = data.flatten()[0]["p_values"]
        for indexit in range(len(pv)):
            if pv[indexit] <= 0.05:
                table_1.add_row([featurev[indexit], format(corrv[indexit],".3f") + " (" +  format(pv[indexit],".3f") +  ")"] )
            else:
                pass
        index_outcome+=1
    f = open("./figures/Table_corr_features" + ".txt", "w")
    stats_funs.mprint(table_1.draw(), f)
    f = open("./figuresTable_corr_features_LaTEX" + ".txt", "w")
    stats_funs.mprint(latextable.draw_latex(table_1, caption="An example table.", label="table:example_table"), f)



    ## Generate Table 5 - Classification metrics for top performing models
    df_allseeds = pd.read_csv("./results/confusionmatrix_representations_allseeds.csv")
    table_1 = texttable.Texttable()
    table_1.set_cols_align("l" * ( len(config.FEATURE_SET_LIST) + 1))
    table_1.set_cols_valign("m" * ( len(config.FEATURE_SET_LIST) + 1))
    a = config.FEATURE_SET_LIST.copy()
    a.insert(0, "Outcome")
    table_1.add_row(a)

    id_outcome = 0 
    for outcome in config.OUTCOME_LIST:
        str_row = []
        for desireddata in range(1, len(config.FEATURE_SET_LIST) + 1):
            AUROC = (np.mean(df_allseeds["AUROC: " + outcome[0:4]][(desireddata - 1)*config.N_RANDOM_SEEDS:(desireddata)*config.N_RANDOM_SEEDS]))
            MCC = (np.mean(df_allseeds["MCC: " + outcome[0:4]][(desireddata - 1)*config.N_RANDOM_SEEDS:(desireddata)*config.N_RANDOM_SEEDS]))
            AUROC_std = (np.std(df_allseeds["AUROC: " + outcome[0:4]][(desireddata - 1)*config.N_RANDOM_SEEDS:(desireddata)*config.N_RANDOM_SEEDS]))
            MCC_std = (np.std(df_allseeds["MCC: " + outcome[0:4]][(desireddata - 1)*config.N_RANDOM_SEEDS:(desireddata)*config.N_RANDOM_SEEDS]))
            p_v = (np.max(df_allseeds["p-v: " + outcome[0:4]][(desireddata - 1)*config.N_RANDOM_SEEDS:(desireddata)*config.N_RANDOM_SEEDS]))
            str_cell = format(MCC, ".3f")+"±" + format(MCC_std, ".3f") + " (" + format(p_v, ".3f") + ")\n" + format(AUROC, ".3f")+"±" + str(format(AUROC_std, ".3f")) 
            str_row.append(str_cell)
        str_row.insert(0, config.OUTCOME_LIST_VERBOSE[id_outcome]+"\nrs (mean±std)\nAUROC (p-value)")
        table_1.add_row(str_row)
        id_outcome += 1
    f = open("./figures/Table_rs_AUROC" + ".txt", "w")
    stats_funs.mprint(table_1.draw(), f)
    f = open("./figuresTable_rs_AUROC_LaTEX" + ".txt", "w")
    stats_funs.mprint(latextable.draw_latex(table_1, caption="An example table.", label="table:example_table"), f)



    ## Generate Table 6 - McNemar test
    df_mcnemar = pd.read_csv("./results/mcnemar_test.csv")

    table_1 = texttable.Texttable()
    table_1.set_cols_align(["l", "l", "l"])
    table_1.set_cols_valign(["m", "m", "m"])
    table_1.add_rows([["Outcome", "Comparison\nX-ray/LUS",  "Comparison\nCT/LUS"],
                    ["Hospital Admission\nOR (p-value)", 
                    str(format(df_mcnemar["OR: OUTC"][0],".2f") +  " (" + format(df_mcnemar["p-v: OUTC"][0],".2f") + ")"), 
                    str(format(df_mcnemar["OR: OUTC"][1],".2f") +  " (" + format(df_mcnemar["p-v: OUTC"][1],".2f") + ")")],
                    ["2-month Mortality\nOR (p-value)", 
                    str(format(df_mcnemar["OR: Dece"][0],".2f") +  " (" + format(df_mcnemar["p-v: Dece"][0],".2f") + ")"), 
                    str(format(df_mcnemar["OR: Dece"][1],".2f") +  " (" + format(df_mcnemar["p-v: Dece"][1],".2f") + ")")],
                    ["Positive SARS-CoV-2\nOR (p-value)", 
                    str(format(df_mcnemar["OR: RT_P"][0],".2f") +  " (" + format(df_mcnemar["p-v: RT_P"][0],".2f") + ")"), 
                    str(format(df_mcnemar["OR: RT_P"][1],".2f") +  " (" + format(df_mcnemar["p-v: RT_P"][1],".2f") + ")")]])
    f = open("./figures/Table_McNemar" + ".txt", "w")
    stats_funs.mprint(table_1.draw(), f)
    f = open("./figures/Table_McNemar_LaTEX" + ".txt", "w")
    stats_funs.mprint(latextable.draw_latex(table_1, caption="An example table.", label="table:example_table"), f)


    ## Generate Table Supp. S2 - Most predictive attributes
    table_1 = texttable.Texttable()
    table_1.set_cols_align("c" * ( len(config.OUTCOME_LIST) + 1))
    table_1.set_cols_valign("m" * ( len(config.OUTCOME_LIST) + 1))
    a = config.OUTCOME_LIST_VERBOSE.copy()
    a.insert(0, "Data input/Outcome")
    table_1.add_row(a)
    df_allseeds = pd.read_csv("./results/confusionmatrix_representations_allseeds.csv")


    id_outcome = 0 
    for desireddata in range(1, len(config.FEATURE_SET_LIST) + 1):
        str_row = []
        for outcome in config.OUTCOME_LIST:
            id_max= (np.argmax(df_allseeds["MCC: " + outcome[0:4]][(desireddata - 1)*config.N_RANDOM_SEEDS:(desireddata)*config.N_RANDOM_SEEDS]))
            if (sum(id_max.shape)> 0): 
                id_max = id_max[0]
            MCC = ((df_allseeds["MCC: " + outcome[0:4]][(desireddata - 1)*config.N_RANDOM_SEEDS:(desireddata)*config.N_RANDOM_SEEDS]))[(desireddata - 1)*config.N_RANDOM_SEEDS + id_max]
            CM00 = ((df_allseeds["CM[0,0]: " + outcome[0:4]][(desireddata - 1)*config.N_RANDOM_SEEDS:(desireddata)*config.N_RANDOM_SEEDS]))[(desireddata - 1)*config.N_RANDOM_SEEDS + id_max]
            CM01 = ((df_allseeds["CM[0,1]: " + outcome[0:4]][(desireddata - 1)*config.N_RANDOM_SEEDS:(desireddata)*config.N_RANDOM_SEEDS]))[(desireddata - 1)*config.N_RANDOM_SEEDS + id_max]
            CM10 = ((df_allseeds["CM[1,0]: " + outcome[0:4]][(desireddata - 1)*config.N_RANDOM_SEEDS:(desireddata)*config.N_RANDOM_SEEDS]))[(desireddata - 1)*config.N_RANDOM_SEEDS + id_max]
            CM11 = ((df_allseeds["CM[1,1]: " + outcome[0:4]][(desireddata - 1)*config.N_RANDOM_SEEDS:(desireddata)*config.N_RANDOM_SEEDS]))[(desireddata - 1)*config.N_RANDOM_SEEDS + id_max]
            p_v = ((df_allseeds["p-v: " + outcome[0:4]][(desireddata - 1)*config.N_RANDOM_SEEDS:(desireddata)*config.N_RANDOM_SEEDS]))[(desireddata - 1)*config.N_RANDOM_SEEDS + id_max]
            str_cell =   "[" + str(CM00) + ", " +  str(CM01) + ", \n" + str(CM10) +  ", " + str(CM11) +  "] \n r_s = " +  format(MCC, ".3f") + " (" + format(p_v, ".3f") +  ")\n"              
            str_row.append(str_cell)
        str_row.insert(0, config.FEATURE_SET_LIST[desireddata - 1])
        table_1.add_row(str_row)
        id_outcome += 1
    f = open("./figures/Table_rs_CONFMATRIX" + ".txt", "w")
    stats_funs.mprint(table_1.draw(), f)
    f = open("./figuresTable_rs_CONFMATRIX_LaTEX" + ".txt", "w")
    stats_funs.mprint(latextable.draw_latex(table_1, caption="An example table.", label="table:example_table"), f)

    # Generate table S3 - most predictive models
    table_1 = texttable.Texttable()
    table_1.set_cols_align("c" * ( len(config.OUTCOME_LIST) + 1))
    table_1.set_cols_valign("m" * ( len(config.OUTCOME_LIST) + 1))
    a = config.OUTCOME_LIST_VERBOSE.copy()
    a.insert(0, "Data input/Outcome")
    table_1.add_row(a)
    df_allseeds = pd.read_csv("./results/confusionmatrix_representations_allseeds.csv")


    id_outcome = 0 
    classifiers = generateClassifiers(0)
    for desireddata in range(1, len(config.FEATURE_SET_LIST) + 1):
        str_row = []
        for outcome in config.OUTCOME_LIST:
            id_max= (np.argmax(df_allseeds["MCC: " + outcome[0:4]][(desireddata - 1)*config.N_RANDOM_SEEDS:(desireddata)*config.N_RANDOM_SEEDS]))
            if (sum(id_max.shape)> 0): 
                id_max = id_max[0]
            df_models = pd.read_csv("./results/summary_of_representations" + "_seed" + str(id_max) + ".csv")
            repeats = df_models.shape[0]/len(config.FEATURE_SET_LIST)
            best_model = df_models["Index: " + outcome[0:4]][(desireddata - 1)*repeats]
            str_cell = str(classifiers[int(best_model)]) + ", Id: " + str(int(best_model))
            str_row.append(str_cell)
        str_row.insert(0, config.FEATURE_SET_LIST[desireddata - 1])
        table_1.add_row(str_row)
        id_outcome += 1
    f = open("./figures/Table_rs_BestModel" + ".txt", "w")
    stats_funs.mprint(table_1.draw(), f)
    f = open("./figuresTable_rs_BestModel_LaTEX" + ".txt", "w")
    stats_funs.mprint(latextable.draw_latex(table_1, caption="An example table.", label="table:example_table"), f)







    a = config.FEATURE_SET_LIST.copy()   



    ## Generate Figure ROC-Curves

    ### Build ROC curves with confidence intervals w.r.t. 5-fold cross validation
    ### Amnamesis, X-ray, Amnamesis + X-ray for deceased
    # https://towardsdatascience.com/pooled-roc-with-xgboost-and-plotly-553a8169680c

    #c_fill      = 'rgba(52, 152, 219, 0.2)'
    #c_line      = 'rgba(52, 152, 219, 0.5)'
    #c_line_main = 'rgba(41, 128, 185, 1.0)'
    #c_grid      = 'rgba(189, 195, 199, 0.5)'
    #c_annot     = 'rgba(149, 165, 166, 0.5)'
    #c_highlight = 'rgba(192, 57, 43, 1.0)'
    c_grid = 'rgba(195, 195, 195, 0.5)'

    #c_line_main_list = ['black', 'cornflowerblue', 'darkseagreen', 'salmon', 'navy', 'darkgreen', 'darkmagenta']
    
    c_line_main_list = ['rgba(0,0,0,1.0)', 'rgba(0.2,0.2,0.2,1.0)', 'rgba(0.4,0.4,0.4,1.0)', 'rgba(0.6,0.6,0.6,1.0)', 'rgba(0.2,0.2,0.2,1.0)', 'rgba(0.4,0.4,0.4,1.0)', 'rgba(0.6,0.6,0.6,1.0)']
    marker_list = ['cross-thin', 'circle-open', 'triangle-up-open', 'square-open', 'circle', 'triangle-up', 'square']
    labels_desireddata = ['Clinical', 'LUS', 'X-ray', 'CT', 'Clinical + LUS', 'Clinical + X-ray', 'Clinical + CT']

    for outcome in config.OUTCOME_LIST:
        fig = go.Figure()
        for desireddata in range(1, len(config.FEATURE_SET_LIST) + 1):
            fpr_list = []
            tpr_list = []
            thresholds_list = []
            auroc_list = []
            for seed in random_seeds:
                ### Visualize ROC curves for top classifiers
                dfsummary_representations = pd.read_csv("./results/summary_of_representations" + "_seed" + str(seed) + ".csv")

                dfsummary = pd.read_csv("./results/dfsummary_" + str(desireddata)+ "_seed" + str(seed)  + ".csv")
                listClassifiers = dfsummary["Classifier"].str[:-3].drop_duplicates() # find unique list of methods

                dfsummary_representations_best = dfsummary_representations.iloc[0::len(listClassifiers)]
                ### Find each model and compute confusion matrix and ROC_CURVE
                mdfclassifieroutputs = pd.read_csv("./results/dfclassifieroutputs_" + str(desireddata) + "_seed" + str(seed) +  ".csv")
                indexBestClassifier = dfsummary_representations_best["Index: "+ outcome[0:4]].iloc[desireddata-1]
                # Load ypred and ytest ... 
                ypred = np.array(mdfclassifieroutputs[outcome[0:4] + "_" + str(int(indexBestClassifier))])
                ypredproba = np.array(mdfclassifieroutputs[outcome[0:4] + "_p_" + str(int(indexBestClassifier))])
                ytrue = np.array(mdfclassifieroutputs[outcome[0:4]])
                # Get rid of NaNs
                yprednonan = ypred[~np.isnan(ypred) & ~np.isnan(ytrue) & ~np.isnan(ypredproba)]
                ypredprobanonan = ypredproba[~np.isnan(ypred) & ~np.isnan(ytrue) & ~np.isnan(ypredproba)]
                ytruenonan = ytrue[~np.isnan(ypred) & ~np.isnan(ytrue) & ~np.isnan(ypredproba)]
                try:
                    fpr, tpr, thresholds = roc_curve(ytruenonan, ypredprobanonan)
                    auroc = roc_auc_score(ytruenonan,ypredprobanonan) 
                    auroc_list.append(auroc)
                    fpr_list.append(fpr)
                    tpr_list.append(tpr)
                    thresholds_list.append(thresholds)
                except:
                    pass



            ## Plot ROC curve here and hold on for next iteration
            # Precondition fpr
            fpr_mean = np.linspace(0,1,100)
            interp_tprs = []
            for i in range(len(fpr_list)):
                fpr           = fpr_list[i]
                tpr           = tpr_list[i]
                interp_tpr    = np.interp(fpr_mean, fpr, tpr)
                interp_tpr[0] = 0.0
                interp_tprs.append(interp_tpr)
            tpr_mean     = np.mean(interp_tprs, axis=0)
            auroc_mean = np.mean(auroc_list)
            tpr_mean[-1] = 1.0
            tpr_std      = 2*np.std(interp_tprs, axis=0)
            tpr_upper    = np.clip(tpr_mean+tpr_std, 0, 1)
            tpr_lower    = tpr_mean-tpr_std


            '''
            fig.add_trace(go.Scatter(
                x          = fpr_mean,
                y          = tpr_upper,
                line       = dict(color=c_line, width=1),
                hoverinfo  = "skip",
                showlegend = False,
                name       = 'upper'))
            fig.add_trace(go.Scatter(
                x          = fpr_mean,
                y          = tpr_lower,
                fill       = 'tonexty',
                fillcolor  = c_fill,
                line       = dict(color=c_line, width=1),
                hoverinfo  = "skip",
                showlegend = False,
                name       = 'lower'))
            '''
            if desireddata == 1:
                fig.add_trace(go.Scatter(
                    x          = fpr_mean,
                    y          = tpr_mean,
                    line       = dict(color=c_line_main_list[desireddata-1], width=2),
                    hoverinfo  = "skip",
                    showlegend = True,
                    name       = labels_desireddata[desireddata-1] + f', AUC: {auroc_mean:.3f}')),
            else:
                fig.add_trace(go.Scatter(
                    x          = fpr_mean,
                    y          = tpr_mean,
                    line       = dict(color=c_line_main_list[desireddata-1], width=2),
                    hoverinfo  = "skip",
                    showlegend = False,
                    name       = labels_desireddata[desireddata-1] + f', AUC: {auroc_mean:.3f}')),

            if desireddata == 1:
                fig.add_trace(go.Scatter(
                    x          = fpr_mean[::5],
                    y          = tpr_mean[::5],
                    mode = "markers",
                    line       = dict(color=c_line_main_list[desireddata-1], width=2),
                    marker     = dict(symbol = marker_list[desireddata - 1], size = 10),
                    hoverinfo  = "skip",
                    showlegend = False,
                    name       = labels_desireddata[desireddata-1] + f', AUC: {auroc_mean:.3f}')),
            else:
                fig.add_trace(go.Scatter(
                    x          = fpr_mean[::5],
                    y          = tpr_mean[::5],
                    mode = "markers",
                    line       = dict(color=c_line_main_list[desireddata-1], width=2),
                    marker     = dict(symbol = marker_list[desireddata - 1], size = 10),
                    hoverinfo  = "skip",
                    showlegend = True,
                    name       = labels_desireddata[desireddata-1] + f', AUC: {auroc_mean:.3f}')),                


            fig.add_shape(
                type ='line', 
                line =dict(dash='dash'),
                x0=0, x1=1, y0=0, y1=1
            )
            fig.update_layout(
                template    = 'plotly_white', 
                title_x     = 0.5,
                xaxis_title = "1 - Specificity",
                yaxis_title = "Sensitivity",
                width       = 800,
                height      = 800,
                legend      = dict(
                    yanchor="bottom", 
                    xanchor="right", 
                    x=0.95,
                    y=0.01),
                font = dict(
                   family="Arial, monospace",
                    size=22,
                    color="black"
                ))

            fig.update_yaxes(
                range       = [0, 1],
                gridcolor   = c_grid,
                scaleanchor = "x", 
                scaleratio  = 1,
                linecolor   = 'black')
            fig.update_xaxes(
                range       = [0, 1],
                gridcolor   = c_grid,
                constrain   = 'domain',
                linecolor   = 'black')
            fig.write_image("./figures/" + "figure_rocplots_" + outcome + ".png", scale=5)
        fig.show()
    print("PHASE 4 COMPLETED")