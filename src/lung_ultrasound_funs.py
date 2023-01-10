from imports import *
import config

def preprocessDF_CorrelationAnalysis(df):
    df["LUNG_US_01_NORMAL"] = np.nanmin(   [ df["RIGHT_LUNG_US_01_NORMAL"] ,    df["LEFT_LUNG_US_01_NORMAL"] ], axis = 0)
    df=df.drop(["LEFT_LUNG_US_01_NORMAL","RIGHT_LUNG_US_01_NORMAL" ], axis =1)
    df["LUNG_US_02_BLINE"] = np.nanmax(   [ df["RIGHT_LUNG_US_02_BLINE"] ,    df["LEFT_LUNG_US_02_BLINE"] ], axis = 0)
    df=df.drop(["LEFT_LUNG_US_02_BLINE","RIGHT_LUNG_US_02_BLINE" ], axis =1)
    df["LUNG_US_03_EFFUSION"] = np.nanmax(   [ df["RIGHT_LUNG_US_03_EFUSSION"] ,    df["LEFT_LUNG_US_03_EFUSSION"] ], axis = 0)
    df=df.drop(["LEFT_LUNG_US_03_EFUSSION","RIGHT_LUNG_US_03_EFUSSION" ], axis =1)
    df["LUNG_US_04_PNEUMOTHORAX"] = np.nanmax(   [ df["RIGHT_LUNG_US_04_PNEUMOTHORAX"] ,    df["LEFT_LUNG_US_04_PNEUMOTHORAX"] ], axis = 0)
    df=df.drop(["LEFT_LUNG_US_04_PNEUMOTHORAX","RIGHT_LUNG_US_04_PNEUMOTHORAX" ], axis =1)
 
    df["LUNG_CT_01_NORMAL"] = np.nanmin(   [ df["RIGHT_LUNG_CT_01_NORMAL"] ,    df["LEFT_LUNG_CT_01_NORMAL"] ], axis = 0)
    df=df.drop(["LEFT_LUNG_CT_01_NORMAL","RIGHT_LUNG_CT_01_NORMAL" ], axis =1)
    df["LUNG_CT_02_GGO"] = np.nanmax(   [ df["RIGHT_LUNG_CT_02_GGO"] ,    df["LEFT_LUNG_CT_02_GGO"] ], axis = 0)
    df=df.drop(["LEFT_LUNG_CT_02_GGO","RIGHT_LUNG_CT_02_GGO" ], axis =1)
    df["LUNG_CT_03_CONSOLIDATION"] = np.nanmax(   [ df["RIGHT_LUNG_CT_03_CONSOLIDATION"] ,    df["LEFT_LUNG_CT_03_CONSOLIDATION"] ], axis = 0)
    df=df.drop(["LEFT_LUNG_CT_03_CONSOLIDATION","RIGHT_LUNG_CT_03_CONSOLIDATION" ], axis =1)
    df["LUNG_CT_04_GGOplusCONSOLIDATION"] = np.nanmax(   [ df["RIGHT_LUNG_CT_04_GGOplusCONSOLIDATION"] ,    df["LEFT_LUNG_CT_04_GGOplusCONSOLIDATION"] ], axis = 0)
    df=df.drop(["LEFT_LUNG_CT_04_GGOplusCONSOLIDATION","RIGHT_LUNG_CT_04_GGOplusCONSOLIDATION" ], axis =1)
    df["LUNG_CT_05_PLEURALEFFUSION"] = np.nanmax(   [ df["RIGHT_LUNG_CT_05_PLEURALEFFUSION"] ,    df["LEFT_LUNG_CT_05_PLEURALEFFUSION"] ], axis = 0)
    df=df.drop(["LEFT_LUNG_CT_05_PLEURALEFFUSION","RIGHT_LUNG_CT_05_PLEURALEFFUSION" ], axis =1)
    df["LUNG_CT_06_PNEUMOTHORAX"] = np.nanmax(   [ df["RIGHT_LUNG_CT_06_PNEUMOTORAX"] ,    df["LEFT_LUNG_CT_06_PNEUMOTORAX"] ], axis = 0)
    df=df.drop(["LEFT_LUNG_CT_06_PNEUMOTORAX","RIGHT_LUNG_CT_06_PNEUMOTORAX" ], axis =1)
    df["LUNG_CT_07_ATELACTASIS"] = np.nanmax(   [ df["RIGHT_LUNG_CT_07_ATELACTASIS"] ,    df["LEFT_LUNG_CT_07_ATELACTASIS"] ], axis = 0)
    df=df.drop(["LEFT_LUNG_CT_07_ATELACTASIS","RIGHT_LUNG_CT_07_ATELACTASIS" ], axis =1)
    df["LUNG_CT_08_CALCIFIED_OLDHEALED"] = np.nanmax(   [ df["RIGHT_LUNG_CT_08_CALCIFIED_OLDHEALED"] ,    df["LEFT_LUNG_CT_08_CALCIFIED_OLDHEALED"] ], axis = 0)
    df=df.drop(["LEFT_LUNG_CT_08_CALCIFIED_OLDHEALED","RIGHT_LUNG_CT_08_CALCIFIED_OLDHEALED" ], axis =1)
    df["LUNG_CT_09_OTHERSLIKE"] = np.nanmax(   [ df["RIGHT_LUNG_CT_09_OTHERSLIKE"] ,    df["LEFT_LUNG_CT_09_OTHERSLIKE"] ], axis = 0)
    df=df.drop(["LEFT_LUNG_CT_09_OTHERSLIKE","RIGHT_LUNG_CT_09_OTHERSLIKE" ], axis =1)
    df["LUNG_CT_10_TREE_IN_NODULES"] = np.nanmax(   [ df["RIGHT_LUNG_CT_10_TREE_IN_NODULES"] ,    df["LEFT_LUNG_CT_10_TREE_IN_NODULES"] ], axis = 0)
    df=df.drop(["LEFT_LUNG_CT_10_TREE_IN_NODULES","RIGHT_LUNG_CT_10_TREE_IN_NODULES" ], axis =1)


    return df







def preprocessDF_MachineLearning(df1):
    valuesAge = np.array(df1["AGE "])
    valuesAge = (valuesAge - np.nanmean(valuesAge))/(3*np.nanstd(valuesAge))
    valuesAge = 0.5*(valuesAge + 1)
    valuesAge[valuesAge < 0] = 0
    valuesAge[valuesAge >= 1] = 1
    df1["AGE "] = valuesAge #normalization
    df1["SEX"]  = df1["SEX"] - 1
    return df1

def selectDesiredData(df, DATA_SELECTOR):
    if DATA_SELECTOR == config.FEATURE_SET_LIST[0]: # EMR
        df = pd.concat([df["S.NO "], df.filter(regex='Comorbidities'), df.filter(regex='Complaints'), df.filter(regex='AGE'), df.filter(regex='SEX')], axis=1)
    elif DATA_SELECTOR ==  config.FEATURE_SET_LIST[1]: # Ultrasound
        df = pd.concat([df["S.NO "], df.filter(regex='US_')], axis = 1)
    elif DATA_SELECTOR == config.FEATURE_SET_LIST[2]: # X-ray
        df = pd.concat([df["S.NO "], df.filter(regex='XRAY')], axis = 1)
    elif DATA_SELECTOR == config.FEATURE_SET_LIST[3]: # CT
        df = pd.concat([df["S.NO "],df.filter(regex='LUNG_CT')], axis = 1)
    elif DATA_SELECTOR == config.FEATURE_SET_LIST[4]: # EMR + Ultrasound
        df = pd.concat([df["S.NO "], df.filter(regex='US_'), df.filter(regex='Comorbidities'), df.filter(regex='Complaints'), df.filter(regex='AGE'), df.filter(regex='SEX')], axis=1)
    elif DATA_SELECTOR == config.FEATURE_SET_LIST[5]: # EMR + X-ray
        df = pd.concat([df["S.NO "], df.filter(regex='XRAY'), df.filter(regex='Comorbidities'), df.filter(regex='Complaints'), df.filter(regex='AGE'), df.filter(regex='SEX')], axis=1)
    elif DATA_SELECTOR == config.FEATURE_SET_LIST[6]: # EMR + CT
        df = pd.concat([df["S.NO "], df.filter(regex='LUNG_CT'), df.filter(regex='Comorbidities'), df.filter(regex='Complaints'), df.filter(regex='AGE'), df.filter(regex='SEX')], axis=1)
    else:
        df = df
    return df