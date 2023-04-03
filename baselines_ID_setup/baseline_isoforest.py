import numpy as np
import pandas as pd
import sys 
import os
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score, auc
from sklearn.metrics import precision_recall_fscore_support, precision_recall_curve

pd.options.mode.chained_assignment = None 


# usage python baseline_isoforest.py anoshift_db_path full_set 
#
# anoshift_db_path      - path to the AnoShift dataset (parquet files)
# full_set              - 0/1 - 1 means all years are used, 0 means only first 5 years (corresponding to our original IID split) are used

# Note: we have considered only 5% of the train data 


def load_train_year(year, anoshift_db_path):
    df = pd.read_parquet(os.path.join(anoshift_db_path, f'subset/{year}_subset.parquet'),  engine='fastparquet')
    df = df.reset_index(drop=True)
    return df

def load_test_year(year, anoshift_db_path):
    df = pd.read_parquet(os.path.join(anoshift_db_path, f'subset/{year}_subset_valid.parquet'),  engine='fastparquet')
    df = df.reset_index(drop=True)
    return df

def rename_columns(df):    
    categorical_cols = ["0", "1", "2", "3", "13"]
    numerical_cols = ["4", "5", "6", "7", "8", "9", "10", "11", "12"]
    additional_cols = ["14", "15", "16", "17", "19"]
    label_col = ["18"]

    new_names = []
    for col_name in df.columns.astype(str).values:
        if col_name in numerical_cols:
            df[col_name] = pd.to_numeric(df[col_name])
            new_names.append((col_name, "num_" + col_name))
        elif col_name in categorical_cols:
            new_names.append((col_name, "cat_" + col_name))
        elif col_name in additional_cols:
            new_names.append((col_name, "bonus_" + col_name))
        elif col_name in label_col:
            df[col_name] = pd.to_numeric(df[col_name])
            new_names.append((col_name, "label"))
        else:
            new_names.append((col_name, col_name))
    df.rename(columns=dict(new_names), inplace=True)
    
    return df

def preprocess(df, train_data, enc=None):
    if not enc:
        enc = OneHotEncoder(handle_unknown='ignore')
        enc.fit(df.loc[:,['cat_' in i for i in df.columns]])
    
    df.loc[df['label'] < 0, 'label'] = -1
    df['label'].replace({1:0}, inplace=True)
    df['label'].replace({-1:1}, inplace=True)
    
    if train_data:
        df = df[df['label']==0]

    num_cat_features = enc.transform(df.loc[:,['cat_' in i for i in df.columns]]).toarray()

    df_catnum = pd.DataFrame(num_cat_features)
    df_catnum = df_catnum.add_prefix('catnum_')

    df = df.reset_index(drop=True)
    num_cols = df.columns.to_numpy()[['num_' in i for i in df.columns]]

    df_catnum[num_cols] = df[num_cols]
    df_catnum['label'] = df['label']
    
    return df_catnum, enc


def print_results(labels, preds, text="?", normalize="true", th=0.5):
    precision_anom, recall_anom, th_anom = precision_recall_curve(labels, preds, pos_label=1)
    precision_norm, recall_norm, th_norm = precision_recall_curve(labels, 1-np.array(preds), pos_label=0)
    
    prec, recall, _, _ = precision_recall_fscore_support(labels, np.array(preds)>=th)
    
    # Use AUC function to calculate the area under the curve of precision recall curve
    pr_auc_norm = auc(recall_norm, precision_norm)
    pr_auc_anom = auc(recall_anom, precision_anom)
    
    roc_auc = roc_auc_score(labels, preds)
    
    print("[%s] ROC-AUC     %.2f%% | PR-AUC-norm    %.2f%% | PR-AUC-anom    %.2f%%" % (text, roc_auc*100, pr_auc_norm*100, pr_auc_anom*100))
    return roc_auc*100, pr_auc_norm*100, pr_auc_anom*100

def get_train(train_years, anoshift_db_path):
    dfs = []

    for year in train_years:
        df_year = load_train_year(year, anoshift_db_path)
        count_norm = df_year[df_year["18"] == "1"].shape[0]
        count_anomal = df_year[df_year["18"] != "1"].shape[0]
        print(year, "normal:", count_norm, "anomalies:", count_anomal)
        dfs.append(df_year)
    
    print("Preprocess train data...")
    df_all_years = pd.concat(dfs, ignore_index=True)
    df_all_years = rename_columns(df_all_years)
    df_new, ohe_enc = preprocess(df_all_years, True)

    # select numerical features
    numerical_cols = df_new.columns.to_numpy()[['num_' in i for i in df_new.columns]]

    X_train_clear = df_new[df_new["label"] == 0]
    X_train_clear = X_train_clear.sample(frac=0.05, random_state=42)
    X_train_num = X_train_clear[numerical_cols].to_numpy()

    return X_train_num, numerical_cols, ohe_enc

if __name__=='__main__':

    anoshift_db_path = sys.argv[1]
    full_set = int(sys.argv[2])

    pair_years = []
    if full_set == 1:
        pair_years.append(([2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015], [2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015]))
    else:
        pair_years.append(([2006, 2007, 2008, 2009, 2010], [2006, 2007, 2008, 2009, 2010]))
 
    for train_years, test_years in pair_years:
        rocs, pr_norms, pr_anoms = [], [], []
        print("Train_years:", train_years)
        sys.stdout.flush()
        X_train_num, numerical_cols, ohe_enc = get_train(train_years, anoshift_db_path)
        print(X_train_num.shape)
        print("Fit IsolationForest...")
        sys.stdout.flush()
        # we have varied random state in range: [42, 44]
        clf = IsolationForest(random_state=42,
                            n_estimators=101,
                            max_samples=1.0,
                            max_features=1.0,
                            verbose=1,
                            n_jobs=10)
        clf.fit(X_train_num)

        del X_train_num
        print("Done fitting.")
        sys.stdout.flush()
        print("Test years:", test_years)
        for year in test_years:
            df_year = load_test_year(year, anoshift_db_path)
            df_year = rename_columns(df_year)
            df_test, _ = preprocess(df_year, False, ohe_enc)
                
            X_test = df_test[numerical_cols].to_numpy()
            print(X_test.shape)
            sys.stdout.flush()
            y_test = df_test["label"].to_numpy()

            X_test = np.nan_to_num(X_test)
            predict_test = (-1) * clf.score_samples(X_test)
            predict_test = np.nan_to_num(predict_test, 0)
            y_test = np.nan_to_num(y_test, 0)
            roc, pr_norm, pr_anom = print_results(y_test, predict_test, text=str(year), normalize=None, th=0.35)
            sys.stdout.flush()
            rocs.append(roc)
            pr_norms.append(pr_norm)
            pr_anoms.append(pr_anom)
            del df_test, df_year, X_test