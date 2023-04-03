import torch
import numpy as np
import random
import pandas as pd
import scipy.io
from scipy.io import arff
import os
import sys
import zipfile

import load_anoshift

class Data_Loader:

    def __init__(self, n_trains=None):
        self.n_train = n_trains

    def get_dataset(self, dataset_name):
        if dataset_name == 'anoshift':
            return load_anoshift.get_ano_shift_data()
        
        # get dataset file name 
        script_dir = os.path.dirname(__file__)
        rel_path = os.path.join("Data/",dataset_name)
        abs_file_path = os.path.join(script_dir, rel_path)

        # list of datasets with generic mat files
        mat_files=['annthyroid','arrhythmia','breastw','cardio','forest_cover','glass','ionosphere','letter','lympho','mammography','mnist','musk',
                   'optdigits','pendigits','pima','satellite','satimage','shuttle','speech','thyroid','vertebral','vowels','wbc','wine']
        
        # load specific dataset 
        if dataset_name in mat_files :
            print ('generic mat file')
            return self.build_train_test_generic_matfile(abs_file_path)

        if dataset_name == 'seismic':
            print('seismic')
            return self.build_train_test_seismic(abs_file_path+'.arff')

        if dataset_name == 'mulcross':
            print('mullcross')
            return self.build_train_test_mulcross(abs_file_path+'.arff')

        if dataset_name == 'abalone':
            print('abalone')
            return self.build_train_test_abalone(abs_file_path+'.data')

        if dataset_name == 'ecoli':
            print('ecoli')
            return self.build_train_test_ecoli(abs_file_path+'.data')

        if dataset_name == 'kdd':
            print ('kdd')
            return self.build_train_test_kdd(script_dir+'/Data/kddcup.data_10_percent_corrected.zip')

        if dataset_name == 'kddrev':
            print ('kddrev')
            return self.build_train_test_kdd_rev(script_dir+'/Data/kddcup.data_10_percent_corrected.zip')

        sys.exit ('No such dataset!')

    def build_train_test_generic_matfile(self,name_of_file):  # takes a matrice that contains the entire dataset, and creates a trainset with 50% of the data of all normals, and the rest are test
        dataset = scipy.io.loadmat(name_of_file)
        # data - n_samples x n_features 
        X = dataset['X']
      
        #X = X - X.mean(axis=0)
        #X = X / X.std(axis=0)
        # labels - n_samples x 1
        # 0 - normals 
        # 1 - anomalies 
        classes = dataset['y']
        # join features and classes 
        jointXY = torch.cat((torch.tensor(X,dtype=torch.double), torch.tensor(classes,dtype=torch.double)), dim=1)
        # select normals and anomalies 
        normals=jointXY[jointXY[:,-1]==0]
        anomalies=jointXY[jointXY[:,-1]==1]
        # shuffle normal samples 
        normals = normals[torch.randperm(normals.shape[0])]
        # split normals in train & test - 50% each
        train, test_norm = torch.split(normals, int(normals.shape[0] / 2) + 1)
        # test will contain both normals and anomalies 
        test = torch.cat((test_norm, anomalies))
        # shuffle test data 
        test = test[torch.randperm(test.shape[0])]
        # shuffle train data
        train = train[torch.randperm(train.shape[0])]

        test_classes = test[:, -1].view(-1, 1)
        # remove label, keep only features 
        train = train[:, 0:train.shape[1] - 1]
        test = test[:, 0:test.shape[1] - 1]
        # train - n_samples x n_features 
        # test - n_samples x n_features 
        # test_classes - n_samples x 1 
        #   0 - normals 
        #   1 - anomalies
        return (train, test, test_classes)

    def build_train_test_seismic(self,name_of_file):  # takes a matrice that contains the entire dataset, and creates a trainset with 50% of the data of all normals, and the rest are test
        dataset, meta = arff.loadarff(name_of_file)
        dataset = pd.DataFrame(dataset)
        classes = dataset.iloc[:, -1]
        dataset = dataset.iloc[:, :-1]
        dataset = pd.get_dummies(dataset.iloc[:, :-1])
        dataset = pd.concat((dataset, classes), axis=1)
        normals = dataset[dataset.iloc[:, -1] == b'0'].values
        anomalies = dataset[dataset.iloc[:, -1] == b'1'].values
        normals = normals[torch.randperm(normals.shape[0])]
        normals = torch.tensor(normals[:, :-1].astype('float32'))
        anomalies = torch.tensor(anomalies[:, :-1].astype('float32'))
        normals = torch.cat((normals, torch.zeros(normals.shape[0]).view(-1, 1)), dim=1)
        anomalies = torch.cat((anomalies, torch.ones(anomalies.shape[0]).view(-1, 1)), dim=1)
        train, test_norm = torch.split(normals, int(normals.shape[0] / 2) + 1)
        test = torch.cat((test_norm, anomalies))
        test = test[torch.randperm(test.shape[0])]
        train = train[torch.randperm(train.shape[0])]
        test_classes = test[:, -1].view(-1, 1)
        train = train[:, 0:train.shape[1] - 1]
        test = test[:, 0:test.shape[1] - 1]
        return (train, test, test_classes)

    def build_train_test_mulcross(self,name_of_file):  # takes a matrice that contains the entire dataset, and creates a trainset with 50% of the data of all normals, and the rest are test
        dataset, _ = arff.loadarff(name_of_file)
        dataset = pd.DataFrame(dataset)
        normals = dataset[dataset.iloc[:, -1] == b'Normal'].values
        anomalies = dataset[dataset.iloc[:, -1] == b'Anomaly'].values
        normals = normals[torch.randperm(normals.shape[0])]
        normals = torch.tensor(normals[:, :-1].astype('float32'))
        anomalies = torch.tensor(anomalies[:, :-1].astype('float32'))
        normals = torch.cat((normals, torch.zeros(normals.shape[0]).view(-1, 1)), dim=1)
        anomalies = torch.cat((anomalies, torch.ones(anomalies.shape[0]).view(-1, 1)), dim=1)
        train, test_norm = torch.split(normals, int(normals.shape[0] / 2) + 1)
        test = torch.cat((test_norm, anomalies))
        test = test[torch.randperm(test.shape[0])]
        train = train[torch.randperm(train.shape[0])]
        test_classes = test[:, -1].view(-1, 1)
        train = train[:, 0:train.shape[1] - 1]
        test = test[:, 0:test.shape[1] - 1]
        return (train, test, test_classes)

    def build_train_test_ecoli(self,name_of_file):  # takes a matrice that contains the entire dataset, and creates a trainset with 50% of the data of all normals, and the rest are test
        dataset = pd.read_csv(name_of_file, header=None, sep='\s+')
        dataset = dataset.iloc[:, 1:]
        anomalies = np.array(
            dataset[(dataset.iloc[:, 7] == 'omL') | (dataset.iloc[:, 7] == 'imL') | (dataset.iloc[:, 7] == 'imS')])[:,
                    :-1]
        normals = np.array(dataset[(dataset.iloc[:, 7] == 'cp') | (dataset.iloc[:, 7] == 'im') | (
                    dataset.iloc[:, 7] == 'pp') | (dataset.iloc[:, 7] == 'imU') | (dataset.iloc[:, 7] == 'om')])[:, :-1]
        normals = torch.tensor(normals.astype('double'))
        anomalies = torch.tensor(anomalies.astype('double'))
        normals = torch.cat((normals, torch.zeros(normals.shape[0], 1,dtype=torch.double)), dim=1)
        anomalies = torch.cat((anomalies, torch.ones(anomalies.shape[0], 1,dtype=torch.double)), dim=1)
        normals = normals[torch.randperm(normals.shape[0])]
        anomalies = anomalies[torch.randperm(anomalies.shape[0])]
        train, test_norm = torch.split(normals, int(normals.shape[0] / 2) + 1)
        test = torch.cat((test_norm, anomalies))
        test = test[torch.randperm(test.shape[0])]
        test_classes = test[:, -1].view(-1, 1)
        train = train[:, :-1]
        test = test[:, :-1]
        return (train, test, test_classes)

    def build_train_test_abalone(self,path):  # takes a matrice that contains the entire dataset, and creates a trainset with 50% of the data of all normals, and the rest are test

        data = pd.read_csv(path, header=None, sep=',')
        data = data.rename(columns={8: 'y'})
        data['y'].replace([8, 9, 10], -1, inplace=True)
        data['y'].replace([3, 21], 0, inplace=True)
        data.iloc[:, 0].replace('M', 0, inplace=True)
        data.iloc[:, 0].replace('F', 1, inplace=True)
        data.iloc[:, 0].replace('I', 2, inplace=True)
        test = data[data['y'] == 0]
        normal = data[data['y'] == -1].sample(frac=1)
        num_normal_samples_test = normal.shape[0] // 2
        test_data = np.concatenate((test.drop('y', axis=1), normal[:num_normal_samples_test].drop('y', axis=1)), axis=0)
        train = normal[num_normal_samples_test:]
        train_data = train.drop('y', axis=1).values
        test_labels = np.concatenate((test['y'], normal[:num_normal_samples_test]['y'].replace(-1, 1)), axis=0)
        for i in range(test_labels.shape[0]):
            if test_labels[i] == 0:
                test_labels[i] = 1
            else:
                test_labels[i] = 0
        train_data=torch.tensor(train_data.astype('double'))
        test_data=torch.tensor(test_data.astype('double'))
        test_labels=torch.tensor(test_labels.astype('double'))
        return (train_data, test_data, test_labels)

    def build_train_test_kdd(self,name_of_file):  # takes a matrice that contains the entire dataset, and creates a trainset with 50% of the data of all normals, and the rest are test
        zf = zipfile.ZipFile(name_of_file)
        kdd_loader = pd.read_csv(zf.open('kddcup.data_10_percent_corrected'), delimiter=',')
        entire_set = np.array(kdd_loader)
        revised_pd = pd.DataFrame(entire_set)
        revised_pd = pd.concat((revised_pd, pd.get_dummies(revised_pd.iloc[:, 1], prefix='new1')), axis=1)
        revised_pd = pd.concat((revised_pd, pd.get_dummies(revised_pd.iloc[:, 2], prefix='new2')), axis=1)
        revised_pd = pd.concat((revised_pd, pd.get_dummies(revised_pd.iloc[:, 3], prefix='new3')), axis=1)
        revised_pd = pd.concat((revised_pd, pd.get_dummies(revised_pd.iloc[:, 6], prefix='new6')), axis=1)
        revised_pd = pd.concat((revised_pd, pd.get_dummies(revised_pd.iloc[:, 11], prefix='new11')), axis=1)
        revised_pd = pd.concat((revised_pd, pd.get_dummies(revised_pd.iloc[:, 21], prefix='new21')), axis=1)
        revised_pd.drop(revised_pd.columns[[1, 2, 3, 6, 11, 20, 21]], inplace=True, axis=1)
        new_columns = [0, 'new1_icmp', 'new1_tcp', 'new1_udp', 'new2_IRC', 'new2_X11', 'new2_Z39_50', 'new2_auth',
                       'new2_bgp',
                       'new2_courier', 'new2_csnet_ns', 'new2_ctf', 'new2_daytime', 'new2_discard', 'new2_domain',
                       'new2_domain_u', 'new2_echo', 'new2_eco_i', 'new2_ecr_i', 'new2_efs', 'new2_exec', 'new2_finger',
                       'new2_ftp', 'new2_ftp_data', 'new2_gopher', 'new2_hostnames', 'new2_http', 'new2_http_443',
                       'new2_imap4',
                       'new2_iso_tsap', 'new2_klogin', 'new2_kshell', 'new2_ldap', 'new2_link', 'new2_login', 'new2_mtp',
                       'new2_name', 'new2_netbios_dgm', 'new2_netbios_ns', 'new2_netbios_ssn', 'new2_netstat', 'new2_nnsp',
                       'new2_nntp', 'new2_ntp_u', 'new2_other', 'new2_pm_dump', 'new2_pop_2', 'new2_pop_3', 'new2_printer',
                       'new2_private', 'new2_red_i', 'new2_remote_job', 'new2_rje', 'new2_shell', 'new2_smtp',
                       'new2_sql_net',
                       'new2_ssh', 'new2_sunrpc', 'new2_supdup', 'new2_systat', 'new2_telnet', 'new2_tftp_u', 'new2_tim_i',
                       'new2_time', 'new2_urh_i', 'new2_urp_i', 'new2_uucp', 'new2_uucp_path', 'new2_vmnet', 'new2_whois',
                       'new3_OTH', 'new3_REJ', 'new3_RSTO', 'new3_RSTOS0', 'new3_RSTR', 'new3_S0', 'new3_S1', 'new3_S2',
                       'new3_S3', 'new3_SF', 'new3_SH', 4, 5, 'new6_0', 'new6_1', 7, 8, 9, 10, 'new11_0', 'new11_1', 12, 13,
                       14,
                       15, 16, 17, 18, 19, 'new21_0', 'new21_1', 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
                       35, 36, 37, 38, 39, 40, 41]
        revised_pd = revised_pd.reindex(columns=new_columns)
        revised_pd.loc[revised_pd[41] != 'normal.', 41] = 0.0
        revised_pd.loc[revised_pd[41] == 'normal.', 41] = 1.0
        kdd_normal = np.array(revised_pd.loc[revised_pd[41] == 0.0], dtype=np.double)
        kdd_anomaly = np.array(revised_pd.loc[revised_pd[41] == 1.0], dtype=np.double)
        kdd_normal = torch.tensor(kdd_normal)
        kdd_anomaly = torch.tensor(kdd_anomaly)
        kdd_normal = kdd_normal[
            torch.randperm(kdd_normal.shape[0])]
        kdd_anomaly = kdd_anomaly[torch.randperm(kdd_anomaly.shape[0])]
        train, test_norm = torch.split(kdd_normal, int(kdd_normal.shape[0] / 2) + 1)
        test = torch.cat((test_norm, kdd_anomaly))
        test = test[torch.randperm(test.shape[0])]
        test_classes = test[:, -1].view(-1, 1)
        train = train[:, 0:train.shape[1] - 1]
        test = test[:, 0:test.shape[1] - 1]
        return (train, test, test_classes)

    def build_train_test_kdd_rev(self,name_of_file):  # takes a matrice that contains the entire dataset, and creates a trainset with 50% of the data of all normals, and the rest are test
        zf = zipfile.ZipFile(name_of_file)
        kdd_loader = pd.read_csv(zf.open('kddcup.data_10_percent_corrected'), delimiter=',')
        entire_set = np.array(kdd_loader)
        revised_pd = pd.DataFrame(entire_set)
        revised_pd = pd.concat((revised_pd, pd.get_dummies(revised_pd.iloc[:, 1], prefix='new1')), axis=1)
        revised_pd = pd.concat((revised_pd, pd.get_dummies(revised_pd.iloc[:, 2], prefix='new2')), axis=1)
        revised_pd = pd.concat((revised_pd, pd.get_dummies(revised_pd.iloc[:, 3], prefix='new3')), axis=1)
        revised_pd = pd.concat((revised_pd, pd.get_dummies(revised_pd.iloc[:, 6], prefix='new6')), axis=1)
        revised_pd = pd.concat((revised_pd, pd.get_dummies(revised_pd.iloc[:, 11], prefix='new11')), axis=1)
        revised_pd = pd.concat((revised_pd, pd.get_dummies(revised_pd.iloc[:, 21], prefix='new21')), axis=1)
        revised_pd.drop(revised_pd.columns[[1, 2, 3, 6, 11, 20, 21]], inplace=True, axis=1)
        new_columns = [0, 'new1_icmp', 'new1_tcp', 'new1_udp', 'new2_IRC', 'new2_X11', 'new2_Z39_50', 'new2_auth',
                       'new2_bgp',
                       'new2_courier', 'new2_csnet_ns', 'new2_ctf', 'new2_daytime', 'new2_discard', 'new2_domain',
                       'new2_domain_u', 'new2_echo', 'new2_eco_i', 'new2_ecr_i', 'new2_efs', 'new2_exec', 'new2_finger',
                       'new2_ftp', 'new2_ftp_data', 'new2_gopher', 'new2_hostnames', 'new2_http', 'new2_http_443',
                       'new2_imap4',
                       'new2_iso_tsap', 'new2_klogin', 'new2_kshell', 'new2_ldap', 'new2_link', 'new2_login', 'new2_mtp',
                       'new2_name', 'new2_netbios_dgm', 'new2_netbios_ns', 'new2_netbios_ssn', 'new2_netstat', 'new2_nnsp',
                       'new2_nntp', 'new2_ntp_u', 'new2_other', 'new2_pm_dump', 'new2_pop_2', 'new2_pop_3', 'new2_printer',
                       'new2_private', 'new2_red_i', 'new2_remote_job', 'new2_rje', 'new2_shell', 'new2_smtp',
                       'new2_sql_net',
                       'new2_ssh', 'new2_sunrpc', 'new2_supdup', 'new2_systat', 'new2_telnet', 'new2_tftp_u', 'new2_tim_i',
                       'new2_time', 'new2_urh_i', 'new2_urp_i', 'new2_uucp', 'new2_uucp_path', 'new2_vmnet', 'new2_whois',
                       'new3_OTH', 'new3_REJ', 'new3_RSTO', 'new3_RSTOS0', 'new3_RSTR', 'new3_S0', 'new3_S1', 'new3_S2',
                       'new3_S3', 'new3_SF', 'new3_SH', 4, 5, 'new6_0', 'new6_1', 7, 8, 9, 10, 'new11_0', 'new11_1', 12, 13,
                       14,
                       15, 16, 17, 18, 19, 'new21_0', 'new21_1', 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
                       35, 36, 37, 38, 39, 40, 41]
        revised_pd = revised_pd.reindex(columns=new_columns)
        revised_pd.loc[revised_pd[41] != 'normal.', 41] = 1.0
        revised_pd.loc[revised_pd[41] == 'normal.', 41] = 0.0
        kdd_anomaly = np.array(revised_pd.loc[revised_pd[41] == 1.0], dtype=np.double)
        kdd_normal = np.array(revised_pd.loc[revised_pd[41] == 0.0], dtype=np.double)
        kdd_normal = torch.tensor(kdd_normal)
        kdd_anomaly = torch.tensor(kdd_anomaly)
        kdd_anomaly = kdd_anomaly[random.sample(range(kdd_anomaly.shape[0]), int(kdd_normal.shape[0] / 4)), :]
        kdd_normal = kdd_normal[ torch.randperm(kdd_normal.shape[0])]
        train, test_norm = torch.split(kdd_normal, int(kdd_normal.shape[0] / 2) + 1)
        test = torch.cat((test_norm, kdd_anomaly))
        test = test[torch.randperm(test.shape[0])]
        test_classes = test[:, -1].view(-1, 1)
        train = train[:, 0:train.shape[1] - 1]
        test = test[:, 0:test.shape[1] - 1]
        return (train, test, test_classes)
