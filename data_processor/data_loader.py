from sklearn.utils import shuffle
import gc
import pandas as pd
pd.options.mode.chained_assignment = None


# name of the label column in different datasets
label_col_names = {
    "kyoto-2016": "18",
}

# value of the positive class for different datasets
label_col_pos_vals = {
    "kyoto-2016": "1",
}


def load_kyoto_principal(contamination=0.0, experiment_type="", experiment_year="", ds_size="subset", random_seed=102):
    """
    Load Kyoto for principal experiments
    Experiment types: iid, finetune, distil

    experiment_type = iid -> train set is [2006-2011], test set is [2012-2015]
    experiment_type = finetune or distil
        train set is [experiment_year], test set is [2012-2015]

    Train set contains only clean (contamination=0) or contamination fraction anomalies
    """

    df_train_set = pd.DataFrame()
    df_test = []

    label_col_name = label_col_names["kyoto-2016"]
    label_col_pos_val = label_col_pos_vals["kyoto-2016"]

    cols = [str(i) for i in range(14)] + [label_col_name, ]

    if experiment_type == "iid":
        train_st = 2006
        train_end = 2011
        train_years = [str(y)
                       for y in range(train_st, train_end + 1)]
        ret_train_ds_name = f"principal_{experiment_type}_trainon_{train_st}-{train_end}"
        if experiment_year != "":
            test_years = [experiment_year, ]
        else:
            test_years = range(2012, 2015 + 1)
    elif experiment_type == "distil" or experiment_type == "finetune":
        train_years = [str(y) for y in range(
            int(experiment_year), int(experiment_year) + 1)]
        ret_train_ds_name = f"principal_{experiment_type}_{experiment_year}"
        test_years = range(2011, 2016)

    for train_year in train_years:
        train_year_path = f"./datasets/Kyoto-2016_AnoShift/{ds_size}/{train_year}_{ds_size}.parquet"
        print("Loading train set:", train_year_path)
        df_train_year = pd.read_parquet(train_year_path)
        df_train_year.drop(columns=list(
            (set(df_train_year.columns) - set(cols))), inplace=True)

        subset_size = df_train_year.shape[0]

        df_train_year_clean = df_train_year[df_train_year[label_col_name]
                                            == label_col_pos_val]
        df_train_year_anom = df_train_year[df_train_year[label_col_name]
                                           != label_col_pos_val]

        num_clean = int(subset_size * (1-contamination))
        num_anom = int(subset_size * contamination)

        if num_clean < subset_size:
            df_train_year_clean = df_train_year_clean.sample(
                n=num_clean, random_state=random_seed)
        df_train_year_anom = df_train_year_anom.sample(
            n=num_anom, random_state=random_seed)

        df_train_set = pd.concat(
            [df_train_set, df_train_year_clean, df_train_year_anom])
        df_train_set.drop_duplicates(keep='first', inplace=True)
        df_train_set.dropna(inplace=True)

        del df_train_year_clean
        del df_train_year_anom

        print(f"Train {train_year} shape {df_train_set.shape}")

    gc.collect()
    df_train = [(ret_train_ds_name, df_train_set)]

    for test_year in test_years:
        test_year_path = f"./datasets/Kyoto-2016_AnoShift/{ds_size}/{test_year}_{ds_size}.parquet"
        print("Loading test set:", test_year_path)
        df_test_year = pd.read_parquet(test_year_path)
        df_test_year.drop(columns=list(
            (set(df_test_year.columns) - set(cols))), inplace=True)
        df_test.append((test_year, df_test_year))

    gc.collect()

    return df_train, df_test


def split_set(df, label_col_name, label_col_pos_val):
    """
    Splits a labeled dataframe in inlier and outlier subsets
    """

    df_inlier = df[df[label_col_name] == label_col_pos_val]
    df_outlier = df[df[label_col_name] != label_col_pos_val]

    df_inlier.drop(
        [
            label_col_name,
        ],
        axis=1,
        inplace=True
    )
    df_inlier.drop_duplicates(keep='first', inplace=True)

    df_outlier.drop(
        [
            label_col_name,
        ],
        axis=1,
        inplace=True
    )
    df_outlier.drop_duplicates(keep='first', inplace=True)

    return df_inlier, df_outlier


def load_local_dataset(
    ds_name="kyoto-2016",
    experiment_year="",
    experiment_set="",
    experiment_type="basic",
    contamination=0.0,
    ds_size="subset",
    random_seed=102
):
    """
    Umbrella function for loading different datasets and transform them in a standard format
    """
    if ds_name == "kyoto-2016":
        if experiment_set == "principal":
            dfs_train, dfs_test = load_kyoto_principal(
                experiment_type=experiment_type,
                experiment_year=experiment_year,
                contamination=contamination,
                ds_size=ds_size,
                random_seed=random_seed)

    label_col_name = label_col_names[ds_name]
    label_col_pos_val = label_col_pos_vals[ds_name]

    df_train_ret = []
    df_test_ret = []

    for df_name, df_train_part in dfs_train:
        df_train_part = shuffle(df_train_part, random_state=random_seed)
        df_train_ret.append(
            (df_name, df_train_part.drop([label_col_name, ], axis=1)))

    for df_name, df_test_part in dfs_test:
        df_test_part = shuffle(df_test_part, random_state=random_seed)

        df_test_part_inlier, df_test_part_outlier = split_set(
            df_test_part, label_col_name=label_col_name, label_col_pos_val=label_col_pos_val
        )
        df_test_ret.append(
            (df_name, df_test_part_inlier, df_test_part_outlier))

    return df_train_ret, df_test_ret
