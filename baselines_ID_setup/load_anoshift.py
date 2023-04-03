import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import os 

train_years_subset = [2006, 2007, 2008, 2009, 2010]
test_years_subset = [2006, 2007, 2008, 2009, 2010]

train_years = [2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015]
test_years = [2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015]

def load_train_year(anoshift_db_path, year):
    df = pd.read_parquet(os.path.join(anoshift_db_path, f'subset/{year}_subset.parquet'))
    df = df.reset_index(drop=True)
    return df

def load_test_year(anoshift_db_path, year):
    df = pd.read_parquet(os.path.join(anoshift_db_path, f'subset/{year}_subset_valid.parquet'))
    df = df.reset_index(drop=True)
    return df 

def rename_columns(df):
    categorical_cols = ["0", "1", "2", "3", "13"]
    numerical_cols = ["4", "5", "6", "7", "8", "9", "10", "11", "12"]
    additional_cols = ["14", "15", "16", "17", "19"]
    label_col = ["18"]

    new_names = []
    for col_name in df.columns.values:
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

def get_train(anoshift_db_path, full_set, train_data_percent=1):
  
    dfs = []

    if full_set == 1:
        current_train_years = train_years 
    else:
        current_train_years = train_years_subset

    for year in current_train_years: 
        df_year = load_train_year(anoshift_db_path, year)
        dfs.append(df_year)
    
    df_all_years = pd.concat(dfs, ignore_index=True)
    df_all_years = rename_columns(df_all_years)

    X_train, ohe_enc = preprocess(df_all_years, True)

    num_cols = X_train.columns.to_numpy()[['num_' in i for i in X_train.columns]]

    X_train = X_train.sample(frac = train_data_percent, random_state=42)

    X_train = X_train[num_cols].to_numpy()
    data_mean = X_train.mean(0)[None,:]
    data_std = X_train.std(0)[None,:]

    data_std[data_std==0] = 1

    return X_train, ohe_enc, data_mean, data_std

def get_n_test_splits(full_set):
    if full_set == 1:
        return len(test_years)
    else:
        return len(test_years_subset)
    
def get_test(anoshift_db_path, full_set, idx, ohe_enc):
  
    if full_set == 1:
        current_test_years = test_years 
    else:
        current_test_years = test_years_subset

    year = current_test_years[idx]
    
    df_year = load_test_year(anoshift_db_path, year)
    df_year = rename_columns(df_year)
    df_test, _ = preprocess(df_year, False, ohe_enc)
    isoforest_cols = df_test.columns.to_numpy()[['num_' in i for i in df_test.columns]]
    X_test = df_test[isoforest_cols].to_numpy()
    y_test = df_test["label"].to_numpy()

    X_test = np.nan_to_num(X_test)
 
    return X_test, y_test