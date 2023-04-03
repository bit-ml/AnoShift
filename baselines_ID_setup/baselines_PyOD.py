# We have considered the PyOD Python library for running part of our baselines 
#      https://pyod.readthedocs.io/en/latest/index.html
#
# In order to run this baselines you need to install pyod along with its dependencies 
#    (models are implemented either in PyTorch or Tensorflow, so both should be available)
#
#       pip install pyd 
#       (check https://pyod.readthedocs.io/en/latest/install.html for details)
# 

import sys 
import numpy as np 
import load_anoshift
import gc 
from pyod.models.ecod import ECOD
from pyod.models.copod import COPOD 
from pyod.models.lunar import LUNAR
from pyod.models.auto_encoder_torch import AutoEncoder
from pyod.models.so_gaal import SO_GAAL
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve

# usage:
#       python baselines_PyOD.py method_name anoshift_db_path train_data_percent full_set
# 
# method_name           - can be one of 'accepted_methods' below      
# anoshift_db_path      - path to the AnoShift dataset (parquet files)
# train_data_percent    - [0,1] - ratio of TRAIN data considered for training the current model
# full_set              - 0/1 - 1 means all years are used, 0 means only first 5 years (corresponding to our original IID split) are used

# Note: for 'lunar' method, we have used 5% of the train data 

accepted_methods = ['ecod', 'copod', 'lunar', 'ae', 'so_gaal']

def get_clf(method_name):
    standardize_data = False
    if method_name == 'ecod':    
        clf = ECOD()
    if method_name == 'copod':   
        clf = COPOD()
    if method_name == 'lunar':    
        clf = LUNAR(verbose=1)       
    if method_name == 'ae':
        clf = AutoEncoder(batch_size=10000, preprocessing=False)
        standardize_data = True 
    if method_name == 'so_gaal': 
        clf = SO_GAAL()
    
    return clf, standardize_data 
    
if __name__=="__main__":
    method_name = sys.argv[1]
    anoshift_db_path = sys.argv[2]
    train_data_percent = float(sys.argv[3])
    full_set = int(sys.argv[4])

    assert (method_name in accepted_methods), 'invalid method name'

    X_train, enc, data_mean, data_std = load_anoshift.get_train(anoshift_db_path, full_set, train_data_percent=train_data_percent)
    print('train split: ',X_train.shape)
    sys.stdout.flush()

    clf, standardize_data = get_clf(method_name)
   
    if standardize_data:
        X_train = (X_train - data_mean) / data_std
 
    clf.fit(X_train)
    del X_train
    gc.collect()
   
    n_test_splits = load_anoshift.get_n_test_splits(full_set)
    for idx in range(n_test_splits):
        print('split %d'%idx)
        X_test, y_test = load_anoshift.get_test(anoshift_db_path, full_set, idx, enc)
        if standardize_data:
            X_test = (X_test - data_mean) / data_std
        print(X_test.shape)
        sys.stdout.flush()
        
        half_n_samples = int(X_test.shape[0] * 0.5)
        y_test_scores_0 = clf.decision_function(X_test[0:half_n_samples])
        y_test_scores_1 = clf.decision_function(X_test[half_n_samples:])
        y_test_scores = np.concatenate((y_test_scores_0, y_test_scores_1))
             
        roc_auc = roc_auc_score(y_test, y_test_scores)
        precision, recall, thresholds = precision_recall_curve(y_test, y_test_scores)
        pr_auc_out = auc(recall, precision)
        precision, recall, thresholds = precision_recall_curve(1-y_test, -y_test_scores)
        pr_auc_in = auc(recall, precision)
        
        print(f'ROC-AUC {roc_auc} -- PR-AUC-in {pr_auc_in} -- PR-AUC-out {pr_auc_out}')
        sys.stdout.flush()

        del X_test 
        del y_test 
        gc.collect()
 