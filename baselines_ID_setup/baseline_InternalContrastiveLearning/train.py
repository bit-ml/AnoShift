import torch
import os 
import sys 
from torch import nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve
import helper_functions
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
print(path)
sys.path.append(path)
#import align_uniform.align_uniform

import load_anoshift

class DatasetBuilder(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return (self.data.shape[0])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # return sample & its index in the dataset
        sample = {'data': self.data[idx], 'index': idx}
        return sample


class encoder_a(nn.Module):
    def __init__(self, kernel_size,hdn_size,d):
        super(encoder_a, self).__init__()
        # F - network processing b (d-kernel_size features)
        # G - network processing a (kernel_size features)

        self.fc1 = nn.Linear(d-kernel_size, hdn_size) #F network
        self.activation1 = nn.Tanh()
        self.fc2 = nn.Linear(hdn_size, hdn_size*2)
        self.activation2 = nn.LeakyReLU(0.2)
        self.fc3 = nn.Linear(hdn_size*2, hdn_size)
        self.activation3 = nn.LeakyReLU(0.2)
        self.batchnorm_1 = nn.BatchNorm1d(d-kernel_size+1)
        self.batchnorm_2 = nn.BatchNorm1d(d-kernel_size+1)

        self.fc1_y = nn.Linear(kernel_size, int(hdn_size/4)) #G network
        self.activation1_y = nn.LeakyReLU(0.2)
        self.fc2_y = nn.Linear(int(hdn_size/4), int(hdn_size/2))
        self.activation2_y = nn.LeakyReLU(0.2)
        self.fc3_y = nn.Linear(int(hdn_size/2), hdn_size)
        self.activation3_y = nn.LeakyReLU(0.2)
        self.batchnorm1_y=nn.BatchNorm1d(d-kernel_size+1)

        self.kernel_size = kernel_size

    def forward(self, x):
       
        x = x.permute(0, 2, 1) # n_samples x n_features x 1
        # y - n_samples x n_possible_as x k  => all as
        # x - n_samples x n_possible_as x (d-k) => all bs
        y,x = helper_functions.positive_matrice_builder(x, self.kernel_size)
        
        # y - n x m x k 
        # x - n x m x (d-k)

        # x - is b - now passing b through network F
        x = self.activation1(self.fc1(x))
        x = self.batchnorm_1(x)
        x = self.activation2(self.fc2(x))
        x = self.batchnorm_2(x)
        x = self.activation3(self.fc3(x))
        # y - is a - now passing a through network G 
        y = self.activation1_y(self.fc1_y(y))
        y = self.batchnorm1_y(y)
        y = self.activation2_y(self.fc2_y(y))
        y = self.activation3_y(self.fc3_y(y))
        
        # two normalization steps 
        # normalization across all possible as 
        # across each dimension of u 
        x=nn.functional.normalize(x,dim=1)
        y=nn.functional.normalize(y,dim=1)
        # normalization across all features
        # s.t. each u-dim vector has norm 1  
        x=nn.functional.normalize(x,dim=2)
        y=nn.functional.normalize(y,dim=2)

        return (x, y)

class trainer():
    def __init__(self,args):
        self.num_epochs = 2000
        self.no_btchs = args.batch_size
        # if you wish to have less negatives than m => set this param 
        self.no_negatives=1000  
        self.temperature=0.01
        self.lr=0.001
        self.faster_version=args.faster_version
    
    def train_and_evaluate(self, db_path, full_set):
        # train & test tensors
        train, enc = load_anoshift.get_train(db_path, full_set)
        #train = load_anoshift.get_ano_shift_train_data()
        train = torch.as_tensor(train, dtype=torch.float)
        print('Train size: ',train.shape)
       
        #test = torch.as_tensor(test, dtype=torch.float)
        # all 0 gpu tensor for contrastloss 
       
        # d - nr features 
        # n - nr samples 
        d = train.shape[1]
        n = train.shape[0]

        if self.faster_version=='yes':
            num_permutations = min(int(np.floor(100 / (np.log(n) + d)) + 1),2)
        else:
            num_permutations = int(np.floor(100/(np.log(n)+d))+1)
        num_permutations = 1
        print("going to run for: ", num_permutations, ' permutations')
        sys.stdout.flush()
        hiddensize = 200
        # kernel_size is k from the paper 
        if d <= 40:
            kernel_size = 2
            stop_crteria = 0.001
        if 40 < d and d <= 160:
            kernel_size = 10
            stop_crteria = 0.01
        if 160 < d:
            kernel_size = d - 150
            stop_crteria = 0.01
     
        num_permutations = 1
        for permutations in range(num_permutations):
            '''
            if num_permutations > 1:
                # shuffle feature indexes 
                random_idx = torch.randperm(train.shape[1])
                train = train[:, random_idx]
                for i in range(len(tests)):
                    tests[i] = tests[i][:, random_idx]
            '''
           
            dataset_train = DatasetBuilder(train)

            model_a = encoder_a(kernel_size, hiddensize, d).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer_a = torch.optim.Adam(model_a.parameters(), lr=self.lr)
            trainloader = DataLoader(dataset_train, batch_size=self.no_btchs,
                                     shuffle=True, num_workers=0, pin_memory=True)
             
           

            ### training
         
            for epoch in range(self.num_epochs):
                model_a.train()
                running_loss = 0
                for i, sample in enumerate(trainloader, 0):
                    model_a.zero_grad()

                    # n_samples x n_features (n x d)
                    pre_query = sample['data'].to(device)
                    pre_query = torch.unsqueeze(pre_query, 1)   # n x 1 x d
                    
                    # pre_query - bs - n_samples x n_possible_as x u (n x m x u)
                    # positive_matrice - as - n_samples x n_possible_as x u (n x m x u)
                    pre_query, positives_matrice = model_a(pre_query)

                    # scores_internal - n_samples x m x (m+1) - for each a, positive and negative pairs scores 
                    # already divided by temperature
                    scores_internal = helper_functions.scores_calc_internal(pre_query, positives_matrice,self.no_negatives,self.temperature).to(device)
                    # => scores_internal - n_samples x (m+1) x m
                    scores_internal = scores_internal.permute(0, 2, 1)
                    
                    # correct_class - n_samples x m
                    # all 0, as in train everything is normal
                    correct_class = torch.zeros((np.shape(scores_internal)[0], np.shape(scores_internal)[2]),
                                                dtype=torch.long).to(device)
                    
                    loss = criterion(scores_internal, correct_class).to(device)
                    loss.backward()
                    optimizer_a.step()
                    running_loss += loss.item()

                    if i%100 == 0:
                        print('[%d, %5d]  loss: %.3f' % (epoch + 1, i, running_loss / (i+1)))
                        sys.stdout.flush()
                # if loss lower than threshold => stop 
                if (running_loss / (i + 1) < stop_crteria):
                    break
  
                if (epoch + 1) % 2 == 0:
                    print('[%d, %5d]  loss: %.3f' % (epoch + 1, i + 1, running_loss / (i + 1)))
                    sys.stdout.flush()
                
            ### testing
            model_a.eval()
            criterion_test = nn.CrossEntropyLoss(reduction='none')
            all_f1 = []
            all_auc = []
            all_pr_auc_in = []
            all_pr_auc_out = []
            n_tests = load_anoshift.get_n_test_splits(full_set)
            for i in range(n_tests):
                test, categories = load_anoshift.get_test(db_path, full_set, i, enc)
             
                print('Evaluate split %d'%i)
                print('split size: ',test.shape)
                sys.stdout.flush()
                test = torch.as_tensor(test, dtype=torch.float)

                #categories = tests_labels[i]
                dataset_test = DatasetBuilder(test)
                testloader = DataLoader(dataset_test, batch_size=self.no_btchs,
                                    shuffle=True, num_workers=0, pin_memory=True)

                test_losses_contrastloss = torch.zeros(test.shape[0],dtype=torch.float).to(device)
                with torch.no_grad():
                    for i, sample in enumerate(testloader, 0):
                        # n_samples x n_features
                        pre_query = sample['data'].to(device)
                        indexes = sample['index'].to(device)
                        # n_samples x 1 x n_features 
                        pre_query_test = torch.unsqueeze(pre_query, 1)  # batch X feature X 1
                        # pre_query - n_samples x n_possible_as x u (n x m x u)        --- all bs
                        # positive_matrice - n_samples x n_possible_as x u (n x m x u) --- all as
                        pre_query_test, positives_matrice_test = model_a(pre_query_test)

                        # scores_internal_test - n_samples x m x (m+1) - for each a, positive and negative pairs scores 
                        # already divided by temperature
                        scores_internal_test = helper_functions.scores_calc_internal(pre_query_test, positives_matrice_test,self.no_negatives,self.temperature).to(device)
                        # => scores_internal_test - n_samples x (m+1) x m
                        scores_internal_test = scores_internal_test.permute(0, 2, 1)
                        
                        correct_class = torch.zeros((np.shape(scores_internal_test)[0], np.shape(scores_internal_test)[2]),
                                                    dtype=torch.long).to(device)
                        loss_test = criterion_test(scores_internal_test, correct_class).to(device)
                        test_losses_contrastloss[indexes] += loss_test.mean(dim=1).to(device)
       
                
                test_losses_contrastloss = test_losses_contrastloss.cpu()  
                # this is actually precision 
                f1 = helper_functions.f1_calculator(categories, test_losses_contrastloss)
                y_labels_boolean_modified = np.array(categories) == 0
                #test_losses_contrastloss = test_losses_contrastloss.cpu()
               
                precision, recall, thresholds = precision_recall_curve(y_labels_boolean_modified, -test_losses_contrastloss)
                pr_auc_in = auc(recall, precision) 
                precision, recall, thresholds = precision_recall_curve(1-y_labels_boolean_modified, test_losses_contrastloss)
                pr_auc_out = auc(recall, precision)
                roc_auc = roc_auc_score(y_labels_boolean_modified, -test_losses_contrastloss)
                print('F1 -- %6.4f -- ROC-AUC %6.4f'%(f1, roc_auc))
                print('PR_AUC-IN -- %6.4f -- PR-AUC-OUT -- %6.4f'%(pr_auc_in, pr_auc_out))
                sys.stdout.flush()
                all_f1.append(f1)
                all_auc.append(roc_auc)
                all_pr_auc_in.append(pr_auc_in)
                all_pr_auc_out.append(pr_auc_out)
        
        return (all_f1,all_auc,all_pr_auc_in, all_pr_auc_out, 0)
        
