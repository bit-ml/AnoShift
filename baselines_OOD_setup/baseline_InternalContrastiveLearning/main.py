import torch
from data_loader import Data_Loader
from train import trainer
import argparse
import numpy as np 
import os 
import sys 
path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

def main(dataset, db_path):
    args.dataset = dataset 
    
    trainer_object=trainer(args)
    
    f_score,auc_score,pr_auc_in, pr_auc_out,df_=trainer_object.train_and_evaluate(db_path)
   
    return (f_score,auc_score, pr_auc_in, pr_auc_out)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1000, help='batch size for training')   
    parser.add_argument('--dataset', type=str, default='anoshift', help='name of dataset')
    parser.add_argument('--faster_version', type=str, default='no', help='faster version with a lower number of repeats')
    parser.add_argument('--dataset_path', type=str, default='', help='db path')
    args = parser.parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    dataset = args.dataset
    db_path = args.dataset_path
    _, auc, pr_auc_in, pr_auc_out = main(dataset, db_path)
    for i in range(len(auc)):
        print('test %d: ROC-AUC %6.4f'%(i, auc[i]))
        print('PR_AUC-IN -- %6.4f -- PR-AUC-OUT -- %6.4f'%(pr_auc_in[i], pr_auc_out[i]))
        sys.stdout.flush()

