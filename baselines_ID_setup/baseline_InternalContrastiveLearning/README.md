# Self Contrastive Anomaly Detection

This repository contains a PyTorch implementation of the method presented in "Anomaly detection with self contrastive loss".


## Training

To replicate the results of the paper run:
```
python main.py --dataset='dataset_name'
```
'dataset_name' can be any of the following:
```
'abalone'
'annthyroid'
'arrhythmia'
'breastw'
'cardio'
'ecoli'
'forest_cover'
'glass'
'ionosphere'
'kdd'
'kddrev'
'letter'
'lympho'
'mammography'
'mnist'
'mulcross'
'musk'
'optdigits'
'pendigits'
'pima'
'satellite'
'satimage'
'seismic'
'shuttle'
'speech'
'thyroid'
'vertebral'
'vowels'
'wbc'
'wine'                                                       
```

If the faster version is preferred, the argument --faster_version='yes' should be added.
