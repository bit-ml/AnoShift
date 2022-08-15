import torch
import numpy as np
import random
import pandas as pd
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def scores_calc_internal(query, positive,no_negatives,tau):
    # query - n x m x u - bs 
    # positive - n x m x u - as

    # n x m x 1 - dot product btw pairs of as and bs 
    pos_multiplication = (query * positive).sum(dim=2).unsqueeze(2).to(device)

    # shuffle negative samples (the as) and potentially select only part of them
    if no_negatives <= query.shape[1]:
        negative_index = random.sample(range(0, query.shape[1]), no_negatives)
    else:
        negative_index = random.sample(range(0, query.shape[1]), query.shape[1])

    # n x m x m - dot product btw bs and all the as considered for negatives 
    neg_multiplication = torch.matmul(query, positive.permute(0, 2, 1)[:, :,negative_index])
    
    # diagonal of neg_multiplication contains same elements as pos_multiplications 
    # Removal of the diagonals - keep in mind that bs are shuffled 
    identity_matrix = torch.eye(np.shape(query)[1]).unsqueeze(0).repeat(np.shape(query)[0], 1,
                                                                        1)[:, :, negative_index].to(device)
    neg_multiplication.masked_fill_(identity_matrix == 1, -float('inf'))  # exp of -inf=0
    
    # logits - n_samples x m x (m+1) - for each sample, positive and negative pairs scroes 
    logits = torch.cat((pos_multiplication, neg_multiplication), dim=2).to(device)
    logits=logits/tau
    return (logits)


def take_per_row_complement(A, indx, num_elem=3):
    # A - n_samples x n_features x n_features 
    # indx - j's - starting points for a 
    # num_elem - k - dimension of a 

    # each row, indexes of the elements of a for corresponding j
    all_indx = indx[:,None] + np.arange(num_elem)

    all_indx_complement=[]
    for row in all_indx:
        # get complement indexes - indexes corresponding to b elements 
        complement=a_minus_b(np.arange(A.shape[2]),row)
        all_indx_complement.append(complement)
    all_indx_complement=np.array(all_indx_complement)

    # all_indx - n_possible_as x k 
    # all_indx_complement - n_possible_as x (d-k)

    # matrice - n_samples x n_possible_as x k 
    # complement_matrice - n_samples x n_possible_as x (d-k)
    matrice = A[:, np.arange(all_indx.shape[0])[:,None], all_indx]
    complement_matrice = A[:,np.arange(all_indx.shape[0])[:,None],all_indx_complement]
    
    return (matrice, complement_matrice)
    #return (A[:,np.arange(all_indx.shape[0])[:,None],all_indx], A[:,np.arange(all_indx.shape[0])[:,None],all_indx_complement])

def positive_matrice_builder(dataset, kernel_size):
    # dataset - n_samples x n_features x 1 => n_samples x n_features 
    dataset = torch.squeeze(dataset, 2)
    # indices - all posible j's  - starting points for a 
    if kernel_size != 1:
        indices = np.array((range(dataset.shape[1])))[:-kernel_size + 1]
    else:
        indices = np.array((range(dataset.shape[1])))
    # dataset - n_samples x n_features => n_samples x 1 x n_features 
    dataset = torch.unsqueeze(dataset, 1)
    # dataset - n_samples x n_features x n_features - through repeat 
    dataset = dataset.repeat(1, dataset.shape[2], 1)

    matrice,complement_matrice = take_per_row_complement(dataset, indices, num_elem=kernel_size)
    return (matrice,complement_matrice)


def take_per_row(A, indx, num_elem=2):
    all_indx = indx[:, None] + np.arange(num_elem)
    return A[:, np.arange(all_indx.shape[0])[:, None], all_indx]


def f1_calculator(classes, losses):
    # classes - n_samples x 1 - 0/1 (0 - normal, 1 - anomalies)
    # losses - n_samples 
    classes=classes.numpy()
    losses=losses.numpy()
    
    df_version_classes = pd.DataFrame(data=classes)
    df_version_losses = pd.DataFrame(losses).astype(np.float64)
    
    # get nr anomalies 
    Na = df_version_classes[df_version_classes.iloc[:, 0] == 1].shape[0]
    
    # get indexes of top Na losses (that would be classified as anomalies knowing the corruption ratio) 
    anomaly_indices = df_version_losses.nlargest(Na, 0).index.values
    # extract all samples classified as anomalies 
    picked_anomalies = df_version_classes.iloc[anomaly_indices]
    
    true_pos = picked_anomalies[picked_anomalies.iloc[:, 0] == 1].shape[0]
    false_pos = picked_anomalies[picked_anomalies.iloc[:, 0] == 0].shape[0]
    # considering how they establish the threshold, TP+FP = n_total_pos => FP == FN => precision = recall = f1 
    f1 = true_pos / (true_pos + false_pos + 0.00001)
    '''
    n_total_pos = df_version_classes[df_version_classes.iloc[:,0]==1].shape[0]
    false_neg = n_total_pos - true_pos
    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
    my_f1 = (2*precision*recall) / (precision+recall)
    '''
    return (f1)#, precision, recall, my_f1)


def a_minus_b (a,b):
    sidx = b.argsort()
    idx = np.searchsorted(b, a, sorter=sidx)
    idx[idx == len(b)] = 0
    out = a[b[sidx[idx]] != a]
    return out