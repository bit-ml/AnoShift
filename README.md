# AnoShift

Preprocessed Kyoto-2006+ data available at:
https://share.bitdefender.com/apps/files/?dir=/Deeplearning/Kyoto-2016_subsets&fileid=716592


## Data preprocessing
* Download original data from https://www.takakura.com/Kyoto_data/ and copy it under ./datasets/kyoto-2016/
* Preprocess the data by running the following command:
    ```python preprocessing/parse_kyoto_logbins.py```
    
## Load data year
* Use the following example to load a sample year (2010) using the Pandas library
    ```df = pd.read_csv('../datasets/preprocessed/logbins_labels_dupskyoto-2016_2010_subset_300000.csv',  engine="python", index_col=[0])```
* Check the sample notebook under ```notebooks/``` for supplementary details
