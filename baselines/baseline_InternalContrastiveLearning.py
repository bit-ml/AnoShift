# Code for testing Internal Contrastive Learning approach [1] approach on AnoShift

# [1] Shenkar T, Wolf L - “Anomaly detection for tabular data with internal contrastive learning” - ICLR 2022

# We have considered the original implementation available as supplementary material of https://openreview.net/forum?id=_hszZbt46bT, which we have modified to accomodate our setup and metrics. 
# Please consider the evaluation results displayed in command line
#
# The modified code is available in our main repo https://github.com/bit-ml/AnoShift, in 'baselines/InternalContrastiveLearning' folder
# Please refer to the original implementation for the unchanged source codes.

import os 
import sys 

# usage:
#   python baseline_InternalContrastiveLearning.py anoshift_db_path 
#
# anoshift_db_path      - path to the AnoShift dataset (parquet files)

main_internalcontrastivelearning_script_path = os.path.join(os.path.dirname(__file__), 'baseline_InternalContrastiveLearning', 'main.py')

if __name__=='__main__':
    anoshift_db_path = sys.argv[1]

    command = 'python \"%s\" --dataset anoshift --dataset_path %s'%(main_internalcontrastivelearning_script_path, anoshift_db_path)
    os.system(command)
