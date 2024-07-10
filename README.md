# AnoShift

Accepted at **NeurIPS 2022** - Datasets and Benchmarks Track
- **Title**: AnoShift: A Distribution Shift Benchmark for Unsupervised Anomaly Detection
- **Authors**: Marius Dragoi, Elena Burceanu, Emanuela Haller, Andrei Manolache, Florin Brad
- **[ArXiv Preprint](https://arxiv.org/abs/2206.15476)**

### :boom::boom: AD benchmark for both In-Distribution (ID) and Out-Of-Distribution (OOD) Anomaly Detection tasks ([full results](https://github.com/bit-ml/AnoShift/blob/main/full_results_OOD_and_ID.pdf)) 


  
**Out-Of-Distribution Anomaly Detection** 
    
  <table>
  <thead>
   <tr>
    <th rowspan=2>Method</th>
    <th colspan=3>ROC-AUC ( $\uparrow$ )</th>
   </tr>
   <tr>
   <th>IID</th>
   <th>NEAR</th>
   <th>FAR</th>
   </tr>
  </thead>
  <tbody>
   <tr>
    <td>OC-SVM <a href="https://proceedings.neurips.cc/paper/1999/hash/8725fb777f25776ffa9076e44fcfd776-Abstract.html">[3]</a></td>
    <td>76.86</td>
    <td>71.43</td> 
    <td>49.57</td>
   </tr>
   <tr>
    <td>IsoForest <a href="https://dl.acm.org/doi/abs/10.1145/2133360.2133363">[10]</a></td>
    <td>86.09</td>
    <td>75.26</td> 
    <td>27.16</td>
   </tr>
   <tr>
    <td>ECOD <a href="https://ieeexplore.ieee.org/abstract/document/9737003?casa_token=vxIxJZ9P9twAAAAA:EOuK0kyHQazQKw4mJtQxJcGzGScWAzMyIiG1N4PxJBYuIIf97G5lkUCmf5kR8egrTiIfraviiSQpNA">[6]</a></td>
    <td>84.76</td>
    <td>44.87</td> 
    <td>49.19</td>
   </tr>          
   <tr>
    <td>COPOD <a href="https://ieeexplore.ieee.org/abstract/document/9338429?casa_token=HOBV8tDoG2IAAAAA:r5LYcMreBldmxYiRXG3Cf7Lm_Hh9M6z1rDiJfAen6f6XkSARYxMHtdk9csPOOpCN-rKN3i2CtZvtmg">[8]</a></td>
    <td>85.62</td>
    <td>54.24</td> 
    <td>50.42</td>
   </tr>  
   <tr>
    <td>LOF <a href="https://dl.acm.org/doi/abs/10.1145/342009.335388">[11]</a></td>
    <td>91.50</td>
    <td>79.29</td> 
    <td>34.96</td>
   </tr>         
   <tr>
    <td>SO-GAAL <a href="https://ieeexplore.ieee.org/abstract/document/8668550?casa_token=-s1yCDFu0MMAAAAA:LD5xNTM79f5dkKt6H4zyBhPqx1hHZL1jZg2p1vIPMQzA-36ARrp83syLz-JxIlABk5iFAUFTFo3hOQ">[1]</a></td>
    <td>50.48</td> 
    <td>54.55</td> 
    <td>49.35</td>
   </tr>
   <tr>
    <td>deepSVDD <a href="https://proceedings.mlr.press/v80/ruff18a.html">[2]</a></td>
    <td>92.67</td>
    <td>87.00</td> 
    <td>34.53</td>
   </tr>
   <tr>
    <td>AE for anomalies <a href="https://link.springer.com/chapter/10.1007/978-3-319-47578-3_1">[4]</a></td>
    <td>81.00</td>
    <td>44.06</td> 
    <td>19.96</td>
   </tr>
   <tr>
    <td>LUNAR <a href="https://ojs.aaai.org/index.php/AAAI/article/view/20629">[9]</a></td>
    <td>85.75</td>
    <td>49.03</td> 
    <td>28.19</td>
   </tr>
   <tr>
    <td>InternalContrastiveLearning <a href="https://openreview.net/forum?id=_hszZbt46bT">[7]</a></td>
    <td>84.86</td>
    <td>52.26</td> 
    <td>22.45</td>
   </tr>
   <tr>
    <td>BERT for anomalies <a href="https://arxiv.org/abs/1810.04805">[5]</a></td>
    <td>84.54 </td>
    <td>86.05</td> 
    <td>28.15</td>
   </tr>
  </tbody>
 </table>
 
  * Average results over multiple runs
  * Train data `files "[year]_subset.parquet" with year in {2006, 2007, 2008, 2009, 2010}`
  * IID test data `files "[year]_subset_valid.parquet" with year in {2006, 2007, 2008, 2009, 2010} `
  * NEAR test data `files "[year]_subset.parquet" with year in {2011, 2012, 2013} `
  * FAR test data `files "[year]_subset.parquet" with year in {2014, 2015} `
  * Results for each split are reported as an average over the performance on each year
  * Scripts for repoducing the results are available in 'baselines_OOD_setup/' (check [**Baselines**](#baselines) section for more details).

**In-Distribution Anomaly Detection** 
           
 <table>
  <thead>
   <tr>
    <th>Method</th>
    <th>ROC-AUC ( $\uparrow$ )</th>
   </tr>
  </thead>
  <tbody>
   <tr>
    <td>OC-SVM <a href="https://proceedings.neurips.cc/paper/1999/hash/8725fb777f25776ffa9076e44fcfd776-Abstract.html">[3]</a></td>
    <td>68.73</td>
   </tr>
   <tr>
    <td>IsoForest <a href="https://dl.acm.org/doi/abs/10.1145/2133360.2133363">[10]</a></td>
    <td>81.27</td>
   </tr>          
   <tr>
    <td>ECOD <a href="https://ieeexplore.ieee.org/abstract/document/9737003?casa_token=vxIxJZ9P9twAAAAA:EOuK0kyHQazQKw4mJtQxJcGzGScWAzMyIiG1N4PxJBYuIIf97G5lkUCmf5kR8egrTiIfraviiSQpNA">[6]</a></td>
    <td>79.41</td>
   </tr>
   <tr>
    <td>COPOD <a href="https://ieeexplore.ieee.org/abstract/document/9338429?casa_token=HOBV8tDoG2IAAAAA:r5LYcMreBldmxYiRXG3Cf7Lm_Hh9M6z1rDiJfAen6f6XkSARYxMHtdk9csPOOpCN-rKN3i2CtZvtmg">[8]</a></td>
    <td>80.89</td>
   </tr>
   <tr>
    <td>LOF <a href="https://dl.acm.org/doi/abs/10.1145/342009.335388">[11]</a></td>
    <td>87.61</td>
   </tr>                   
   <tr>
    <td>SO-GAAL <a href="https://ieeexplore.ieee.org/abstract/document/8668550?casa_token=-s1yCDFu0MMAAAAA:LD5xNTM79f5dkKt6H4zyBhPqx1hHZL1jZg2p1vIPMQzA-36ARrp83syLz-JxIlABk5iFAUFTFo3hOQ">[1]</a></td>
    <td>49.90</td>
   </tr>
   <tr>
    <td>deepSVDD <a href="https://proceedings.mlr.press/v80/ruff18a.html">[2]</a></td>
    <td>88.24</td>
   </tr>
   <tr>
    <td>AE for anomalies <a href="https://link.springer.com/chapter/10.1007/978-3-319-47578-3_1">[4]</a></td>
    <td>64.08</td>
   </tr>
   <tr>
    <td>LUNAR <a href="https://ojs.aaai.org/index.php/AAAI/article/view/20629">[9]</a></td>
    <td>78.53</td>
   </tr>
   <tr>
    <td>InternalContrastiveLearning <a href="https://openreview.net/forum?id=_hszZbt46bT">[7]</a></td>
    <td>66.99</td>
   </tr>
   <tr>
    <td>BERT for anomalies <a href="https://arxiv.org/abs/1810.04805">[5]</a></td>
    <td>79.62</td>
   </tr>
  </tbody>
 </table>
 
  * Average results over multiple runs 
  * Train data `files "[year]_subset.parquet" with year in {2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015}`
  * Test data `files "[year]_subset_valid.parquet" with year in {2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015} `
  * Scripts for repoducing the results are available in 'baselines_ID_setup/' (check [**Baselines**](#Baselines) section for more details).
## AnoShift Protocol

- We introduce an unsupervised anomaly detection benchmark with data that shifts over time, built over Kyoto-2006+, a traffic dataset for network intrusion detection. This type of data meets the premise of shifting the input distribution: it covers a large time span (from 2006 to 2015), with naturally occurring changes over time. In AnoShift, we split the data in **IID**, **NEAR**, and **FAR** testing splits. We validate the performance degradation over time with diverse models (MLM to classical Isolation Forest).

![AnoShift overview - Kyoto-2006+](resources/feat_shift_normals_abnormals.png)

- With all tested baselines, we notice a significant decrease in performance on **FAR** years for inliers, showing that there might be a particularity with those years. We observe a large distance in the Jeffreys divergence on **FAR** and the rest of the years for 2 features: service type and the number of bytes sent by the source IP.

- From the OTDD analysis we observe that: first, the inliers from **FAR** are very distanced to training years; and second, the outliers from **FAR** are quite close to the training inliers.

- We propose a BERT model for MLM and compare several training regimes: **iid**, **finetune** and a basic **distillation** technique, and show that acknowledging the distribution shift leads to better test performance on average.

## Kyoto-2006+ data

- Original Kyoto-2006+ data available at: https://www.takakura.com/Kyoto_data/ (in AnoShift we used the New data (Nov. 01, 2006 - Dec. 31, 2015))

- Preprocessed Kyoto-2006+ data available at: https://storage.googleapis.com/bitdefender_ml_artifacts/anoshift/Kyoto-2016_AnoShift.tar

- The result is obtained by applying the preprocessing script `data_processor/parse_kyoto_logbins.py` to the original data.

- The preprocessed dataset is available in pandas parquet format and available as both `full sets` and `subsets` with 300k inlier samples, with equal outlier proportions to the original data.

- In the notebook tutorials, we use the subsets for fast experimentation. In our experiments, the subset results are consistent with the full sets.

* Label column (18) has value 1 for inlier class (normal traffic) and -1 (known type) and -2 (unknown type) for anomalies.

## Prepare data

```
curl https://storage.googleapis.com/bitdefender_ml_artifacts/anoshift/Kyoto-2016_AnoShift.tar --output AnoShift.zip

mkdir datasets

mv AnoShift.zip datasets

unzip datasets/AnoShift.zip -d datasets/

rm datasets/AnoShift.zip
```

## Prepare environment
* Create a new conda environment: `conda create --name anoshift`
* Activate the new environment: `conda activate anoshift`
* Install pip: `conda install -c anaconda pip`
* Upgrade pip: `pip install --upgrade pip`
* Install dependencies: `pip install -r requirements.txt`

## Baselines

We provide numeros baselines in the `baselines_OOD_setup/` directory, which are a good entrypoint for familiarizing with the protocol:

- `baseline_*.ipynb`: isoforest/ocsvm/LOF baselines on AnoShift
- `baseline_deep_svdd/baseline_deepSVDD.py`: deppSVDD baseline on AnoShift
- `baseline_BERT_train.ipynb`: BERT baseline on AnoShift
- `baseline_InternalContrastiveLearning.py`: InternalContrastiveLearning baseline on AnoShift
- `baselines_PyOD.py`: ['ecod', 'copod', 'lunar', 'ae', 'so_gaal'] baselines on AnoShift using PyOD
- `iid_finetune_distill_comparison.ipynb`: compare the IID, finetune and distillation training strategies for the BERT model, on AnoShift

* run the notebooks from the `root` of the project: `jupyter-notebook .`

If you intend to use AnoShift in the ID setup, please consider the code provided in 'baselines_ID_setup/'. You can use either the full set (`full_set=1` => all ten years) or the years corresponding to our original IID split (`full_set=0` => first five years) (check usage instructions for each baseline in order to switch between them).

## Please cite this project as:

```
@article{druagoi2022anoshift,
  title={AnoShift: A Distribution Shift Benchmark for Unsupervised Anomaly Detection},
  author={Dr{\u{a}}goi, Marius and Burceanu, Elena and Haller, Emanuela and Manolache, Andrei and Brad, Florin},
  journal={Neural Information Processing Systems {NeurIPS}, Datasets and Benchmarks Track},
  year={2022}
}
```
