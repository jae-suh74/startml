# import all required modules
import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import sys
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, 
    brier_score_loss, roc_curve, auc
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier,LGBMRegressor
from missforest.missforest import MissForest
import shap


INNER_CV_LOOPS = 10

def split_and_impute(df, seed, factor_types):
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df['outcome'],
        random_state = seed
    )
        
    classifier = LGBMClassifier(random_state=seed, verbosity=-1)
                               # max_depth=50, n_estimators=10)
    regressor = LGBMRegressor(random_state=seed, verbosity=-1)
                             # max_depth=50, n_estimators=10)
    
    mf = MissForest(clf=classifier, rgr=regressor)
    train_df_imputed = mf.fit_transform(train_df, categorical = factor_types)
    test_df_imputed = mf.fit_transform(test_df, categorical = factor_types)

    return train_df_imputed, test_df_imputed


def one_hot_encode(df):
    oh = OneHotEncoder(
        sparse_output=False,
        drop='first'
    )
    
    dff = df.copy()
    
    for c in dff.columns:
        if c == "outcome":
            continue
        
        # Create dummy variable with appropriate names
        if dff[c].dtype == 'category':
            dummy_variables = oh.fit_transform(
                   dff[c].values.reshape(-1, 1)
            )
            renamed_features = oh.get_feature_names_out([c])

            for i, renamed_feat in enumerate(renamed_features):
                dff[renamed_feat] = dummy_variables[:, i]

            dff.drop(columns=[c], inplace=True)
            
    return dff


def run_rf(train_df, test_df, all_except_outcome):
    rf = ExtraTreesClassifier()
    grid_search_params = {
        "criterion": ["entropy"],
         "max_features": [0.01,0.02,0.03,0.05,0.07,0.09,0.1],
         "max_depth": [3,5,7,9,11],
         "n_estimators": [100,200,300,400,500]
        
    }
    fitter = GridSearchCV(
        rf,
        grid_search_params,
        cv=INNER_CV_LOOPS,
        n_jobs=-1
    )

    fitter.fit(
        X=train_df[all_except_outcome], # This is where we specify predictors
        y=train_df["outcome"]
    )

    # Best model
    rf = ExtraTreesClassifier(
        criterion="entropy",
        max_depth=fitter.best_estimator_.max_depth,
        max_features=fitter.best_estimator_.max_features,
        n_estimators=fitter.best_estimator_.n_estimators
    )
    rf.fit(
        train_df[all_except_outcome],
        train_df["outcome"].astype(int)
    )
    
    rf_predictions = rf.predict(test_df[all_except_outcome])
    rf_probs = rf.predict_proba(test_df[all_except_outcome])[:,1]
    truth = test_df['outcome']
    
    metrics = get_metrics(predictions = rf_predictions, truth = truth, probs = rf_probs)
    metrics['max_depth'] = fitter.best_estimator_.max_depth
    metrics['max_features'] = fitter.best_estimator_.max_features
    metrics['n_estimators'] = fitter.best_estimator_.n_estimators
    
    
    train_predictions = rf.predict(train_df[all_except_outcome])
    train_probabilities = rf.predict_proba(train_df[all_except_outcome])[:,1]
    train_truth = train_df['outcome']
    train_metrics = get_metrics(train_predictions, train_truth, train_probabilities)
    
    for k,v in train_metrics.items():
        metrics[f'train_{k}'] = v
    
    explainer = shap.TreeExplainer(rf) # Attribute error 
    shap_values = explainer(train_df[all_except_outcome])
    
    for i,c in enumerate(all_except_outcome):
        metrics[f'train_shap_{c}'] = np.mean(np.abs(shap_values.values[:,i]))
    
    explainer = shap.TreeExplainer(rf) # Attribute error 
    shap_values = explainer(test_df[all_except_outcome])
    
    for i,c in enumerate(all_except_outcome):
        metrics[f'test_shap_{c}'] = np.mean(np.abs(shap_values.values[:,i]))
    
    
    return metrics


def run_gbm(train_df, test_df, all_except_outcome):
    gbm = XGBClassifier()
    grid_search_params = {
            'n_estimators': [50, 100, 300, 500],
            'learning_rate': [0.01, 0.03, 0.1], # 2-10/number of trees
            'max_depth': [4,6,8,10],
            'subsample': [0.5, 0.75, 1.0],
            'max_depth': [1, 3, 5, 7]
        
    }
    # print("Setting up the GridSearch")        
    fitter = GridSearchCV(
        gbm,
        grid_search_params,
        cv=INNER_CV_LOOPS,
        n_jobs=-1
    )
    # print("Fitting the model")
    fitter.fit(
        X=train_df[all_except_outcome], # This is where we specify predictors
        y=train_df["outcome"]
    )
    
    #Best model
    gbm = XGBClassifier(
        n_estimators=fitter.best_estimator_.n_estimators,
        learning_rate=fitter.best_estimator_.learning_rate,
        max_depth=fitter.best_estimator_.max_depth,
        subsample=fitter.best_estimator_.subsample
    )

    gbm.fit(
        train_df[all_except_outcome],
        train_df["outcome"].astype(int)
    )
    
    gbm_predictions = gbm.predict(test_df[all_except_outcome])
    gbm_probs = gbm.predict_proba(test_df[all_except_outcome])[:,1]
    truth = test_df['outcome']
    
    metrics = get_metrics(predictions=gbm_predictions, probs=gbm_probs, truth=truth)
    
    metrics['n_estimators']=fitter.best_estimator_.n_estimators
    metrics['learning_rate']=fitter.best_estimator_.learning_rate
    metrics['max_depth']=fitter.best_estimator_.max_depth
    metrics['subsample']=fitter.best_estimator_.subsample
    
    train_predictions = gbm.predict(train_df[all_except_outcome])
    train_probabilities = gbm.predict_proba(train_df[all_except_outcome])[:,1]
    train_truth = train_df['outcome']
    train_metrics = get_metrics(train_predictions, train_truth, train_probabilities)
    
    for k,v in train_metrics.items():
        metrics[f'train_{k}'] = v
    
    explainer = shap.TreeExplainer(gbm) # Attribute error 
    shap_values = explainer(train_df[all_except_outcome])
    
    for i,c in enumerate(all_except_outcome):
        metrics[f'train_shap_{c}'] = np.mean(np.abs(shap_values.values[:,i]))
    
    explainer = shap.TreeExplainer(gbm) # Attribute error 
    shap_values = explainer(test_df[all_except_outcome])
    
    for i,c in enumerate(all_except_outcome):
        metrics[f'test_shap_{c}'] = np.mean(np.abs(shap_values.values[:,i]))
    
    return metrics

def run_elasticnet(train_df, test_df, all_except_outcome):
    
    grid_search_params = {
        "l1_ratio": [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.03, 0.05], 
        "C": 1/np.linspace(0.01, 2, num=10) 
        
    }
    en = LogisticRegression(
        penalty='elasticnet',
        solver='saga',
        tol=1e-3,
        max_iter=5000
    )
    fitter = GridSearchCV(
        en,
        grid_search_params,
        cv=INNER_CV_LOOPS,
        n_jobs=-1
    )
    fitter.fit(
        X=train_df[all_except_outcome], # This is where we specify predictors
        y=train_df["outcome"]
    )
    
    en = LogisticRegression(
        penalty='elasticnet',
        solver='saga',
        l1_ratio=fitter.best_estimator_.l1_ratio,
        tol=1e-3,
        max_iter=5000,
        C=fitter.best_estimator_.C
    )
    
    en.fit(
         X=train_df[all_except_outcome],
         y=train_df["outcome"]
    )
    
    en_predictions = en.predict(test_df[all_except_outcome])
    en_probabilities = en.predict_proba(test_df[all_except_outcome])[:,1]
    truth = test_df['outcome']
    
    metrics = get_metrics(en_predictions, truth, en_probabilities)
    metrics['l1_ratio']=fitter.best_estimator_.l1_ratio
    metrics['C']=fitter.best_estimator_.C
    
    train_predictions = en.predict(train_df[all_except_outcome])
    train_probabilities = en.predict_proba(train_df[all_except_outcome])[:,1]
    train_truth = train_df['outcome']
    train_metrics = get_metrics(train_predictions, train_truth, train_probabilities)
    
    for k,v in train_metrics.items():
        metrics[f'train_{k}'] = v
    
    explainer = shap.Explainer(en.predict, train_df[all_except_outcome]) # Attribute error 
    shap_values = explainer(train_df[all_except_outcome])
    
    for i,c in enumerate(all_except_outcome):
        metrics[f'train_shap_{c}'] = np.mean(np.abs(shap_values.values[:,i]))
    
    explainer = shap.Explainer(en.predict, test_df[all_except_outcome]) # Attribute error 
    shap_values = explainer(test_df[all_except_outcome])
    
    for i,c in enumerate(all_except_outcome):
        metrics[f'test_shap_{c}'] = np.mean(np.abs(shap_values.values[:,i]))

    return metrics

def get_metrics(predictions, truth, probs):

    cm = confusion_matrix(truth, predictions)
    ppv = precision_score(truth, predictions, pos_label=1)
    npv = cm[0,0]/(cm[0,0] + cm[1,0])
    spec = cm[0,0]/(cm[0,0] + cm[0,1])
    recall = recall_score(truth, predictions, pos_label=1)
    brier = brier_score_loss(truth, predictions, pos_label=1)
    fpr, tpr, thresholds = roc_curve(
        truth,
        probs,
    )
    roc_auc = auc(fpr, tpr)
    
    tn, fp, fn, tp = cm.ravel()
    
    
    return {
        "PPV": ppv,
        "NPV": npv,
        "Specificity": spec,
        "Recall": recall,
        "BrierScore": brier,
        "AUC": roc_auc,
        "tn":tn,
        "fp":fp,
        "fn":fn,
        "tp":tp, 
    }


def main(df, seed, factor_types, numeric_types, baseline=False):
    
    print("Splitting and imputing data")
    # this is where we split and impute the data using MissForest
    train_df_imputed, test_df_imputed = split_and_impute(df,seed,factor_types)

    for c in factor_types:
        train_df_imputed[c] = train_df_imputed[c].astype('category')
        test_df_imputed[c] = test_df_imputed[c].astype('category')

    for c in numeric_types:
        train_df_imputed[c] = train_df_imputed[c].astype('float')
        test_df_imputed[c] = test_df_imputed[c].astype('float')
        
    # one-hot encode
    print("Onehot encoding...")
    train_df_imputed_oh = one_hot_encode(train_df_imputed)
    test_df_imputed_oh = one_hot_encode(test_df_imputed)

    # Create a condition where, after one-hot encoding, if dummy variables are not shared
    # between train and test sets, ignore that seed and start a new one
    if len(set(train_df_imputed_oh.columns)^set(test_df_imputed_oh.columns))>0:
        return {"status": False}
    
    if baseline:
        status = {"Non-Offender": 0, "Offender":1}
        
        train_metrics = get_metrics(train_df_imputed['Off_NOff'].apply(status.get), train_df_imputed['outcome'], train_df_imputed['Off_NOff'].apply(status.get))
        test_metrics = get_metrics(test_df_imputed['Off_NOff'].apply(status.get), test_df_imputed['outcome'], test_df_imputed['Off_NOff'].apply(status.get))
        
        for k,v in train_metrics.items():
            test_metrics[f'train_{k}'] = v
        
        test_metrics['status'] = True
        return test_metrics

    all_except_outcome = [c for c in train_df_imputed_oh.columns if c != 'outcome']

    # Fit single variable GLM
    print(f"Fitting a single variable GLM For seed {seed}")
    glm = LogisticRegression(max_iter=5000)
    glm.fit(
        X=train_df_imputed_oh['alloffs_pre'].values.reshape(-1, 1),
        y=train_df_imputed_oh["outcome"].astype(int)
    )
    glm_predictions = glm.predict(test_df_imputed_oh['alloffs_pre'].values.reshape(-1, 1))
    glm_probabilities = glm.predict_proba(test_df_imputed_oh['alloffs_pre'].values.reshape(-1, 1))

    glm_metrics = get_metrics(
        predictions=glm_predictions,
        truth=test_df_imputed_oh["outcome"],
        probs = glm_probabilities[:,1]
    )

    glm_train_predictions = glm.predict(train_df_imputed_oh['alloffs_pre'].values.reshape(-1, 1))
    glm_train_probabilities = glm.predict_proba(train_df_imputed_oh['alloffs_pre'].values.reshape(-1, 1))

    glm_train_metrics = get_metrics(
        predictions=glm_train_predictions,
        truth=train_df_imputed_oh['outcome'],
        probs = glm_train_probabilities[:,1]
    )

    for k,v in glm_train_metrics.items():
        glm_metrics['train_'+k] = v
    
    print(f"Fitting a multi variable GLM For seed {seed}")
    # Fit multivariable GLM
    multiglm = LogisticRegression(max_iter=5000)
    multiglm.fit(
        X=train_df_imputed_oh[all_except_outcome],
        y=train_df_imputed_oh["outcome"].astype(int)
    )
    multiglm_predictions = multiglm.predict(test_df_imputed_oh[all_except_outcome])
    multiglm_probabilities = multiglm.predict_proba(test_df_imputed_oh[all_except_outcome])

    multiglm_metrics = get_metrics(
        predictions=multiglm_predictions,
        truth=test_df_imputed_oh["outcome"],
        probs = multiglm_probabilities[:,1]
    )

    multiglm_train_predictions = multiglm.predict(train_df_imputed_oh[all_except_outcome])
    multiglm_train_probabilities = multiglm.predict_proba(train_df_imputed_oh[all_except_outcome])

    multiglm_train_metrics = get_metrics(
        predictions=multiglm_train_predictions,
        truth=train_df_imputed_oh['outcome'],
        probs = multiglm_train_probabilities[:,1]
    )

    for k,v in multiglm_train_metrics.items():
        multiglm_metrics['train_'+k] = v

    explainer = shap.Explainer(multiglm.predict, train_df_imputed_oh[all_except_outcome]) # Attribute error 
    shap_values = explainer(train_df_imputed_oh[all_except_outcome])
    
    for i,c in enumerate(all_except_outcome):
        multiglm_metrics[f'train_shap_{c}'] = np.mean(np.abs(shap_values.values[:,i]))
    
    explainer = shap.Explainer(multiglm.predict, test_df_imputed_oh[all_except_outcome]) # Attribute error 
    shap_values = explainer(test_df_imputed_oh[all_except_outcome])
    
    for i,c in enumerate(all_except_outcome):
        multiglm_metrics[f'test_shap_{c}'] = np.mean(np.abs(shap_values.values[:,i]))
    
    # Fit Elastic net
    print(f"Fitting a elastic net For seed {seed}")
    elastic_net_metrics = run_elasticnet(train_df_imputed_oh, test_df_imputed_oh, all_except_outcome)
    
    # Run RF/ExtraTrees
    print(f"Fitting an ExtraTrees classifier for seed {seed}")
    rf_metrics = run_rf(train_df_imputed_oh, test_df_imputed_oh, all_except_outcome)
    
    print(f"Fitting a GBM classifier for seed {seed}")
    gbm_metrics = run_gbm(train_df_imputed_oh, test_df_imputed_oh, all_except_outcome)
    
    glm_metrics['seed'] = seed
    multiglm_metrics['seed'] = seed
    gbm_metrics['seed'] = seed
    rf_metrics['seed'] = seed
    elastic_net_metrics['seed'] = seed
    
    glm_metrics['model'] = 'SingleGLM'
    multiglm_metrics['model'] = 'MultiGLM'
    gbm_metrics['model'] = 'GBM'
    rf_metrics['model'] = 'RF'
    elastic_net_metrics['model'] = 'ElasticNet'
    
    return {
        "glm_metrics": glm_metrics,
        "multiglm_metrics": multiglm_metrics,
        "en_metrics": elastic_net_metrics,
        "rf_metrics": rf_metrics,
        "gbm_metrics": gbm_metrics,
        "status": True
    }


# Read in imputed data from R
print("Reading in data...")
df = pd.read_stata("../data/clean_data.dta")


factor_types = ['Site',
  'Referrer',
  'OnsetCD',
  'soffend',
  'poffend',
  'yeduc_main',
  'yeduc_spec',
  'ysen',
  'gender',
  'ethnicity',
  'married',
  'PQual',
  'pemploy',
  'SES',
  'yaccom',
  'time',
  'dawba_dep',
  'dawba_adhd',
  'dawba_cdt',
  'Off_NOff',
  'outcome']


numeric_types = ['Age',
  'IQ',
  'numchild',
  'P_ICU_TotalT1',
  'YP_SMF_TotalT1',
  'YP_ICU_TotalT1',
  'NoExclT1',
  'TotalDaysExclT1',
  'P_CONN_LEARLANGTscoreT1',
  'P_CONN_ADHDTscoreT1',
  'P_SDQ_TotalImpactT1',
  'P_SDQ_EmotT1',
  'P_SDQ_CDT1',
  'P_SDQ_HyperT1',
  'P_SDQ_PeerRelT1',
  'P_SDQ_ProSocT1',
  'YP_SRD_Del_ExSib_VolT1',
  'YP_SRD_SubMis_VolT1',
  'YP_SRD_PeerIllSubT1',
  'YP_SRD_PeerDelT1',
  'YP_SDQ_TotalImpactT1',
  'YP_SDQ_EmotT1',
  'YP_SDQ_CDT1',
  'YP_SDQ_HyperT1',
  'YP_SDQ_PeerRelT1',
  'YP_SDQ_ProSocT1',
  'UnauthorisedAbsT1',
  'P_GHQ_TotalT1',
  'CTS25TotalT1',
  'P_ALAB_PosParentT1',
  'P_ALAB_ParInvT1',
  'P_ALAB_MonT1',
  'P_ALAB_CorPunT1',
  'P_ALAB_IncDisT1',
  'P_LOEB_TotalT1',
  'YP_ABAS_TotalT1',
  'P_FACE_CohesionDimensionT1',
  'P_FACE_FlexibilityDimensionT1',
  'P_FACE_FCommT1',
  'P_FACE_FSatT1',
  'YP_LEE_TotalT1',
  'YP_YouthMatScaleT1',
  'alloffs_pre',
  'YP_SRD_Del_ExSib_VarT1',
  'YP_SRD_SubMis_VarT1']

allvars = ['Site',
  'Referrer',
  'OnsetCD',
  'soffend',
  'poffend',
  'yeduc_main',
  'yeduc_spec',
  'ysen',
  'gender',
  'ethnicity',
  'married',
  'PQual',
  'pemploy',
  'SES',
  'yaccom',
  'time',
  'dawba_dep',
  'dawba_adhd',
  'dawba_cdt',
  'Age',
  'IQ',
  'numchild',
  'P_ICU_TotalT1',
  'YP_SMF_TotalT1',
  'YP_ICU_TotalT1',
  'NoExclT1',
  'TotalDaysExclT1',
  'P_CONN_LEARLANGTscoreT1',
  'P_CONN_ADHDTscoreT1',
  'P_SDQ_TotalImpactT1',
  'P_SDQ_EmotT1',
  'P_SDQ_CDT1',
  'P_SDQ_HyperT1',
  'P_SDQ_PeerRelT1',
  'P_SDQ_ProSocT1',
  'YP_SRD_Del_ExSib_VolT1',
  'YP_SRD_SubMis_VolT1',
  'YP_SRD_PeerIllSubT1',
  'YP_SRD_PeerDelT1',
  'YP_SDQ_TotalImpactT1',
  'YP_SDQ_EmotT1',
  'YP_SDQ_CDT1',
  'YP_SDQ_HyperT1',
  'YP_SDQ_PeerRelT1',
  'YP_SDQ_ProSocT1',
  'UnauthorisedAbsT1',
  'P_GHQ_TotalT1',
  'CTS25TotalT1',
  'P_ALAB_PosParentT1',
  'P_ALAB_ParInvT1',
  'P_ALAB_MonT1',
  'P_ALAB_CorPunT1',
  'P_ALAB_IncDisT1',
  'P_LOEB_TotalT1',
  'YP_ABAS_TotalT1',
  'P_FACE_CohesionDimensionT1',
  'P_FACE_FlexibilityDimensionT1',
  'P_FACE_FCommT1',
  'P_FACE_FSatT1',
  'YP_LEE_TotalT1',
  'YP_YouthMatScaleT1',
  'alloffs_pre',
  'Off_NOff',
  'outcome',
  'YP_SRD_Del_ExSib_VarT1',
  'YP_SRD_SubMis_VarT1']

print("Setting data types...")
for c in factor_types:
    df[c] = df[c].astype('category')
for c in numeric_types:
    df[c] = df[c].astype('float')


# toggle for getting clinician model
# clinician_vars = ['Site',
#   'Referrer',
#   'soffend',
#   'poffend',
#   'gender',
#   'ethnicity',
#   'married',
#   'PQual',
#   'pemploy',
#   'SES',
#   'yaccom',
#   'time',
#   'dawba_dep',
#   'dawba_adhd',
#   'dawba_cdt',
#   'Age',
#   'IQ',
#   'numchild',
#   'alloffs_pre',
#   'Off_NOff',
#   'outcome']
# df = df[clinician_vars].copy()
# factor_types = [c for c in factor_types if c in clinician_vars]
# numeric_types = [c for c in numeric_types if c in clinician_vars]

    
glm_container = []
multiglm_container = []
rf_container = []
en_container = []
gbm_container = []
baseline_container = []

# this is our outer loop
SEED = 0
NUM_RESULTS = 100

baseline = False

while len(glm_container) < NUM_RESULTS:
    print(f"Running the analysis for SEED {SEED}")
    
    # within main we will perform the inner loop
    metric_suite = main(df, SEED, factor_types, numeric_types, baseline=baseline)
    
    if metric_suite["status"] is False:
        print(f"Ignoring SEED {SEED} due to data processing issues")
        SEED += 1
        continue

    if baseline:
        # dummy object to allow the while loop to work
        glm_container.append(1)
        baseline_container.append(metric_suite)
        SEED += 1
        continue
    
    glm_container.append(metric_suite['glm_metrics'])
    multiglm_container.append(metric_suite['multiglm_metrics'])
    rf_container.append(metric_suite['rf_metrics'])
    en_container.append(metric_suite['en_metrics'])
    gbm_container.append(metric_suite['gbm_metrics'])
    
    SEED += 1

# # sys.exit()
    

glm_metrics_df = pd.concat([pd.DataFrame.from_dict(c,orient='index') for c in glm_container], axis=1)
multiglm_metrics_df = pd.concat([pd.DataFrame.from_dict(c,orient='index') for c in multiglm_container], axis=1)
rf_metrics_df = pd.concat([pd.DataFrame.from_dict(c,orient='index') for c in rf_container], axis=1)
en_metrics_df = pd.concat([pd.DataFrame.from_dict(c,orient='index') for c in en_container], axis=1)
gbm_metrics_df = pd.concat([pd.DataFrame.from_dict(c,orient='index') for c in gbm_container], axis=1)
# baseline_metrics_df = pd.concat([pd.DataFrame.from_dict(c,orient='index') for c in baseline_container], axis=1)
# baseline_metrics_df.to_csv("baseline_metrics.csv")

glm_metrics_df.to_csv("glm_metrics_shap.csv")
rf_metrics_df.to_csv("rf_metrics_shap.csv")
gbm_metrics_df.to_csv("gbm_metrics_shap.csv")
en_metrics_df.to_csv("en_metrics_shap.csv")
multiglm_metrics_df.to_csv("multiglm_metrics_shap.csv")
