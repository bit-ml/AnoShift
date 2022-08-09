import pandas as pd
import numpy as np

PATHS = [
    "/data/logs-datasets/cicids2017/MachineLearningCSV/MachineLearningCVE/Monday-WorkingHours.pcap_ISCX.csv",
    "/data/logs-datasets/cicids2017/MachineLearningCSV/MachineLearningCVE/Tuesday-WorkingHours.pcap_ISCX.csv",
    "/data/logs-datasets/cicids2017/MachineLearningCSV/MachineLearningCVE/Wednesday-workingHours.pcap_ISCX.csv",
    "/data/logs-datasets/cicids2017/MachineLearningCSV/MachineLearningCVE/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "/data/logs-datasets/cicids2017/MachineLearningCSV/MachineLearningCVE/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "/data/logs-datasets/cicids2017/MachineLearningCSV/MachineLearningCVE/Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "/data/logs-datasets/cicids2017/MachineLearningCSV/MachineLearningCVE/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "/data/logs-datasets/cicids2017/MachineLearningCSV/MachineLearningCVE/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
]


def concat_column_with_value(df):
    df = df.rename(columns=lambda s: s.replace(" ", "_"))
    return df.astype(str).radd("_").radd([*df])


df = pd.read_csv(PATHS[0])

for i in range(1, len(PATHS)):
    temp = pd.read_csv(PATHS[i])
    df = pd.concat([df, temp])
    break


m = df.loc[df[" Flow Packets/s"] != np.inf, " Flow Packets/s"].max()
df[" Flow Packets/s"].replace(np.inf, m, inplace=True)
m = df.loc[df["Flow Bytes/s"] != np.inf, "Flow Bytes/s"].max()
df["Flow Bytes/s"].replace(np.inf, m, inplace=True)
all_cols = df.columns

for col in all_cols:
    if df[col].nunique() <= 2:  # col will be removed
        continue

    if col == " Label":
        continue

    if df[col].nunique() > 200:
        num_bins = 100
    else:
        num_bins = 10

    df["binned_" + str(col).replace(" ", "")] = pd.cut(
        df[col],
        bins=num_bins,
        labels=[col.replace(" ", "") + str(idx) for idx in range(num_bins)],
    ).astype(str)


# df = df.replace(" ", "", regex=True)

print(df)

dtypes = df.dtypes
print(f"Number of columns with Int {len(dtypes[dtypes == int])}")
print(f"Number of columns with float {len(dtypes[dtypes == float])}")
print(f"Number of columns with object {len(dtypes[dtypes == object])}")

null_values = df.isna().sum()
null_values[null_values > 0]

null_index = np.where(df["Flow Bytes/s"].isnull())[0]
df.dropna(inplace=True)

inlier = df[df[" Label"] == "BENIGN"]
outlier = df[df[" Label"] != "BENIGN"]

inlier = inlier.drop(
    all_cols,
    axis=1,
)

outlier = outlier.drop(
    all_cols,
    axis=1,
)

print("Dropped cols")
print("Inlier", inlier.shape)
print("Outlier", outlier.shape)


inlier = inlier.drop_duplicates(keep=False)
outlier = outlier.drop_duplicates(keep=False)

inlier_size = inlier.shape[0]
train_size = 0.75
num_train = int(inlier_size * train_size)


inlier_train = inlier.sample(n=num_train)
inlier_test = pd.concat([inlier, inlier_train]).drop_duplicates(keep=False)
# outlier = outlier.sample(n=inlier_size - num_train)

print("Inlier train", inlier_train.shape)
print("Inlier test", inlier_test.shape)
print("Outlier", outlier.shape)

# inlier_train = concat_column_with_value(inlier_train)
# inlier_test = concat_column_with_value(inlier_test)
# outlier = concat_column_with_value(outlier)

inlier_train_txt = pd.Series(inlier_train.values.astype(str).tolist()).str.join(" ")
inlier_test_txt = pd.Series(inlier_test.values.astype(str).tolist()).str.join(" ")
outlier_txt = pd.Series(outlier.values.astype(str).tolist()).str.join(" ")

with open("/data/logs-datasets/cicids2017/train/inlier.txt", "w") as f:
    for line in inlier_train_txt:
        f.write(line + "\n")

with open("/data/logs-datasets/cicids2017/test/inlier.txt", "w") as f:
    for line in inlier_test_txt:
        f.write(line + "\n")

with open("/data/logs-datasets/cicids2017/test/outlier.txt", "w") as f:
    for line in outlier_txt:
        f.write(line + "\n")
