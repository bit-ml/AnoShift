import pandas as pd
import os


ds_path = "./datasets/kyoto-2016/"
ds_years = [
    "2006",
    "2007",
    "2008",
    "2009",
    "2010",
    "2011",
    "2012",
    "2013",
    "2014",
    "2015",
]

df = pd.DataFrame()
cols = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 11, 12]

bins = [1.1 ** i - 1 for i in range(233)]
bins[0] -= 0.01
bins[-1] += 0.01

print(bins)

col_bounds = {
    0: {"min": 0.0, "max": 86397.394358},
    2: {"min": 0.0, "max": 4294967295.0},
    3: {"min": 0.0, "max": 4294967295.0},
    4: {"min": 0.0, "max": 100.0},
    5: {"min": 0.0, "max": 1.0},
    6: {"min": 0.0, "max": 1.0},
    7: {"min": 0.0, "max": 1.0},
    8: {"min": 0.0, "max": 100.0},
    9: {"min": 0.0, "max": 100.0},
    10: {"min": 0.0, "max": 1.0},
    11: {"min": 0.0, "max": 1.0},
    12: {"min": 0.0, "max": 1.0},
}

dups = True

if dups:
    prefix = "logbins_withdups_"
else:
    prefix = "logbins_nodups_"


def parse_dataset():
    df = pd.DataFrame()

    for ds_year in ds_years:
        df_yr = pd.DataFrame()

        print("Parsing: " + ds_year)
        ds_year_months = sorted(os.listdir(os.path.join(ds_path, ds_year)))
        for idx_month, ds_year_month in enumerate(ds_year_months):
            ds_entries = []
            print(idx_month, ds_year_month)
            filenames = os.listdir(os.path.join(
                ds_path, ds_year, ds_year_month))
            for filename in sorted(filenames):
                filepath = os.path.join(
                    ds_path, ds_year, ds_year_month, filename)
                with open(filepath, "r") as f:
                    lines = f.read().splitlines()
                    line_items = []
                    for l in lines:
                        lsplit = l.split("\t")
                        timestamp = ds_year + "_" + \
                            ds_year_month + "_" + lsplit[-2]
                        protocol = lsplit[-1]
                        line_items.append(lsplit[:18] + [timestamp, protocol])

                    ds_entries += line_items
            df_yr_mt = pd.DataFrame(ds_entries)

            for col in cols:
                if col in [0, 2, 3]:
                    num_bins = len(bins)
                    bin_names = [
                        "c" + str(col).replace(" ", "") + str(idx)
                        for idx in range(num_bins - 1)
                    ]
                    df_yr_mt[col] = pd.cut(
                        df_yr_mt[col].astype(float),
                        bins=bins,
                        labels=bin_names,
                    ).astype(str)

            df_yr = pd.concat([df_yr, df_yr_mt])
            del df_yr_mt

            if not dups:
                df_yr = df_yr.drop_duplicates(keep=False)

            if (idx_month + 1) == 6:
                df_yr.to_csv(
                    f"./datasets/preprocessed/kyoto-2016/labeled/{prefix}kyoto-2016_{ds_year}_p1.csv")
                del df_yr
                df_yr = pd.DataFrame()
            if (idx_month + 1) == 12:
                df_yr.to_csv(
                    f"./datasets/preprocessed/kyoto-2016/labeled/{prefix}kyoto-2016_{ds_year}_p2.csv")

        if ds_year == "2006":
            df_yr.to_csv(
                f"./datasets/preprocessed/kyoto-2016/labeled/{prefix}kyoto-2016_{ds_year}.csv")
        del df_yr

    return df


def split_iid(p_test=0.1, random_state=10):
    for year in ds_years:
        df_year = pd.read_pickle(
            f"./datasets/preprocessed/kyoto-2016/{prefix}kyoto-2016_{year}.pkl")
        df_year_inlier = df_year[df_year[14] == "1"]
        df_year_outlier_known = df_year[df_year[14] == "-1"]
        df_year_outlier_unknown = df_year[df_year[14] == "-2"]

        df_year_inlier_test = df_year_inlier.sample(
            frac=p_test, random_state=random_state
        )
        df_year_outlier_known_test = df_year_outlier_known.sample(
            frac=p_test, random_state=random_state
        )
        df_year_outlier_unknown_test = df_year_outlier_unknown.sample(
            frac=p_test, random_state=random_state
        )

        df_year_test = pd.concat(
            [
                df_year_inlier_test,
                df_year_outlier_known_test,
                df_year_outlier_unknown_test,
            ]
        )

        df_year_train = pd.concat(
            [df_year, df_year_test]).drop_duplicates(keep=False)

        train_path = f"./datasets/preprocessed/kyoto-2016/{prefix}iid_kyoto-2016_{year}_train.pkl"
        test_path = f"./datasets/preprocessed/kyoto-2016/{prefix}iid_kyoto-2016_{year}_test.pkl"

        print("Saving train subset to ", train_path, df_year_train.shape)
        df_year_train.to_pickle(train_path)
        print("Saving test subset to ", test_path, df_year_test.shape)
        df_year_test.to_pickle(test_path)


parse_dataset()

if __name__ == '__main__':
    parse_dataset()
