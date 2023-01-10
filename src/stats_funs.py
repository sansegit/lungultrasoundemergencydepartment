
from imports import *

def cor_selector(X, y,num_feats):
    cor_list = []
    pval_list = []
    feature_name = X.columns.tolist()
    # calculate the correlation with y for each feature
    for i in X.columns.tolist():
        xdata = X[i]
        ydata = y
        xdatanonan = xdata[~np.isnan(ydata) & ~np.isnan(xdata) ]
        ydatanonan = ydata[~np.isnan(ydata) & ~np.isnan(xdata) ]
        rho, pval = stats.spearmanr(xdatanonan, ydatanonan)
        #cor = np.corrcoef(xdatanonan, ydatanonan)[0, 1]
        cor_list.append(rho)
        pval_list.append(pval)
    cor_list_np = np.array(cor_list)
    pval_list_np = np.array(pval_list)
    # feature name
    cor_feature = X.iloc[:,np.argsort(np.abs(cor_list_np))[-1:-(num_feats+1):-1]].columns.tolist()
    cor_values = cor_list_np[np.argsort(np.abs(cor_list_np))[-1:-(num_feats+1):-1]]
    pval_values = pval_list_np[np.argsort(np.abs(cor_list_np))[-1:-(num_feats+1):-1]]
    # feature selection? 0 for not select, 1 for select
    cor_support = [True if i in cor_feature else False for i in feature_name]
    return cor_support, cor_feature, cor_values, pval_values

def cor_calculator(X, y):
    xdata = X
    ydata = y
    xdatanonan = xdata[~np.isnan(ydata) & ~np.isnan(xdata) ]
    ydatanonan = ydata[~np.isnan(ydata) & ~np.isnan(xdata) ]
    rho, pval = stats.spearmanr(xdatanonan, ydatanonan)
    return rho, pval

def score_cor_calculator(X, y):
    xdata = X
    ydata = y
    xdatanonan = xdata[~np.isnan(ydata) & ~np.isnan(xdata) ]
    ydatanonan = ydata[~np.isnan(ydata) & ~np.isnan(xdata) ]
    rho, _ = stats.spearmanr(xdatanonan, ydatanonan)
    return rho

def performMcNemarTest (labels1, predictions1, labels2, predictions2, destination_results):
# def performMcNemarTest (labels1, predictions1, labels2, predictions2)
# performs McNemar test to compare merits of two methods
# can be applied to frames or patient cases
    f = open(destination_results + ".txt", "w")

    contingency_Table = np.array([[0,0],[0,0]])
    classifier1_correct = labels1 == predictions1
    classifier2_correct = labels2 == predictions2
    contingency_Table[0, 0] = np.count_nonzero ((classifier1_correct == True) & (classifier2_correct == True))
    contingency_Table[0, 1] = np.count_nonzero ((classifier1_correct == True) & (classifier2_correct == False))
    contingency_Table[1, 0] = np.count_nonzero ((classifier1_correct == False) & (classifier2_correct == True))
    contingency_Table[1, 1] = np.count_nonzero ((classifier1_correct == False) & (classifier2_correct == False))
    contingency_Table = np.round(contingency_Table/5) # Due to 5-repeated k-cross validations

    mprint("             2 Correct,       2 Incorre", f)
    mprint("1 Correct    %10d       %10d" % (contingency_Table[0,0], contingency_Table[0,1]), f)
    mprint("1 Incorre    %10d       %10d" % (contingency_Table[1,0], contingency_Table[1,1]), f)
   
    mprint(contingency_Table, f)
    total_num_cases = contingency_Table[0,0] + contingency_Table[0,1]  + contingency_Table[1,0] + contingency_Table[1,1]
    mprint("No elements: %d" % total_num_cases, f)
    contingency_Table_ratios = np.array(contingency_Table)/total_num_cases
    mprint("Contigency table in proportions (%)", f)
    mprint("             2 Correct,       2 Incorre", f)
    mprint("1 Correct    %10.3f       %10.3f" % (contingency_Table_ratios[0,0], contingency_Table_ratios[0,1]), f)
    mprint("1 Incorre    %10.3f       %10.3f" % (contingency_Table_ratios[1,0], contingency_Table_ratios[1,1]), f)
   

    odds_ratio_1_2 = contingency_Table[0,1]/contingency_Table[1,0]
    mprint("Odd ratios OR = 1Correct_2Incorre/1Incorre_1Correct", f) 
    mprint("OR = %10.3f" % (odds_ratio_1_2), f)


    if (np.min(contingency_Table) <= 25):
        isexact = True
    else:
        isexact = False

    results = mcnemar(contingency_Table, exact=isexact)


    mprint(results,f) # we use standard function to avoid issues
    return odds_ratio_1_2, results.pvalue
    '''
    if exact == False:
        correction = True
        if correction:
            mcnemar_statistic = (np.abs(contingency_Table[0,1] - contingency_Table[1,0]) - 1) ** 2/(contingency_Table[0,1] + contingency_Table[1,0])
        else:
            mcnemar_statistic = (np.abs(contingency_Table[0,1] - contingency_Table[1,0])) ** 2/(contingency_Table[0,1] + contingency_Table[1,0])
        mprint("Mc Nemar statistic %10.3f" % mcnemar_statistic, f)
        pvalue_mcnemar = chi2.sf(mcnemar_statistic, df=1)
    else:
        #pvalue_mcnemar = 2*binom.sf(contingency_Table[0,1] - 1, contingency_Table[0,1] + contingency_Table[1,0], 0.5)
        n = contingency_Table[0,1] + contingency_Table[1,0]
        i = contingency_Table[0,1]
        i_n = np.arange(i+1, n+1)
        pvalue_mcnemar = 1 - np.sum(comb(n, i_n) * 0.5 ** i_n * (1 - 0.5) ** (n - i_n)) 
        pvalue_mcnemar *= 2
        #mid_p_value = pvalue_mcnemar - binom.pmf(i, n, 0.5)
        #pvalue_mcnemar = mid_p_value


    mprint ("p-value = %10.8f" % pvalue_mcnemar, f)
    '''



def mprint (what_to_print, file):
    import sys
    print(what_to_print)  # print to screen
    original_stdout = sys.stdout
    sys.stdout = file
    print(what_to_print) # print to file
    sys.stdout = original_stdout
    return
