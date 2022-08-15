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
from pyod.models.vae import VAE
from pyod.models.so_gaal import SO_GAAL
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve

# usage:
#       python baselines_PyOD.py method_name anoshoft_db_path train_data_percent
# 
# method_name           - can be one of 'accepted_methods' below      
# anoshift_db_path      - path to the AnoShift dataset (parquet files)
# train_data_percent    - [0,1] - ratio of TRAIN data considered for training the current model

accepted_methods = ['ecod', 'copod', 'lunar', 'vae', 'so_gaal']

def get_clf(method_name):

    if method_name == 'ecod':    
        clf = ECOD()
    if method_name == 'copod':   
        clf = COPOD()
    if method_name == 'lunar':    
        clf = LUNAR(verbose=1)       
    if method_name == 'vae':
        clf = VAE(verbose=2, batch_size=10000)
    if method_name == 'so_gaal': 
        clf = SO_GAAL()
    
    return clf 
    
if __name__=="__main__":
    method_name = sys.argv[1]
    anoshift_db_path = sys.argv[2]
    train_data_percent = float(sys.argv[3])

    assert (method_name in accepted_methods), 'invalid method name'

    X_train, enc = load_anoshift.get_train(anoshift_db_path, train_data_percent=train_data_percent)
    print('train split: ',X_train.shape)
    sys.stdout.flush()
 
    clf = get_clf(method_name)
   
    clf.fit(X_train)
    del X_train
    gc.collect()
   
    n_test_splits = load_anoshift.get_n_test_splits()
    for idx in range(n_test_splits):
        print('split %d'%idx)
        X_test, y_test = load_anoshift.get_test(anoshift_db_path, idx, enc)
        print(X_test.shape)
        sys.stdout.flush()
        
      
        y_test_scores = clf.decision_function(X_test)
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
 