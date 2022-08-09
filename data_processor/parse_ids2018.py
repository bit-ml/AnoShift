import pandas as pd
import numpy as np
import math

ids_datatypes = {
    "Dst Port": np.int32,
    "Protocol": np.int8,
    "Flow Duration": np.int64,
    "Tot Fwd Pkts": np.int16,
    "Tot Bwd Pkts": np.int16,
    "TotLen Fwd Pkts": np.int32,
    "TotLen Bwd Pkts": np.int32,
    "Fwd Pkt Len Max": np.int32,
    "Fwd Pkt Len Min": np.int32,
    "Fwd Pkt Len Mean": np.float64,
    "Fwd Pkt Len Std": np.float64,
    "Bwd Pkt Len Max": np.float32,
    "Bwd Pkt Len Min": np.float32,
    "Bwd Pkt Len Mean": np.float64,
    "Bwd Pkt Len Std": np.float64,
    "Flow Byts/s": np.float64,
    "Flow Pkts/s": np.float64,
    "Flow IAT Mean": np.float64,
    "Flow IAT Std": np.float64,
    "Flow IAT Max": np.int64,
    "Flow IAT Min": np.int32,
    "Fwd IAT Tot": np.int32,
    "Fwd IAT Mean": np.float32,
    "Fwd IAT Std": np.float64,
    "Fwd IAT Max": np.int32,
    "Fwd IAT Min": np.int32,
    "Bwd IAT Tot": np.int32,
    "Bwd IAT Mean": np.float64,
    "Bwd IAT Std": np.float64,
    "Bwd IAT Max": np.int64,
    "Bwd IAT Min": np.int64,
    "Fwd PSH Flags": np.int8,
    "Bwd PSH Flags": np.int8,
    "Fwd URG Flags": np.int8,
    "Bwd URG Flags": np.int8,
    "Fwd Header Len": np.int32,
    "Bwd Header Len": np.int32,
    "Fwd Pkts/s": np.float64,
    "Bwd Pkts/s": np.float64,
    "Pkt Len Min": np.int16,
    "Pkt Len Max": np.int32,
    "Pkt Len Mean": np.float64,
    "Pkt Len Std": np.float64,
    "Pkt Len Var": np.float64,
    "FIN Flag Cnt": np.int8,
    "SYN Flag Cnt": np.int8,
    "RST Flag Cnt": np.int8,
    "PSH Flag Cnt": np.int8,
    "ACK Flag Cnt": np.int8,
    "URG Flag Cnt": np.int8,
    "CWE Flag Count": np.int8,
    "ECE Flag Cnt": np.int8,
    "Pkt Size Avg": np.float32,
    "Fwd Seg Size Avg": np.float32,
    "Bwd Seg Size Avg": np.float32,
    "Fwd Byts/b Avg": np.int8,
    "Fwd Pkts/b Avg": np.int8,
    "Fwd Blk Rate Avg": np.int8,
    "Bwd Byts/b Avg": np.int8,
    "Bwd Pkts/b Avg": np.int8,
    "Bwd Blk Rate Avg": np.int8,
    "Subflow Fwd Pkts": np.int16,
    "Subflow Fwd Byts": np.int32,
    "Subflow Bwd Pkts": np.int16,
    "Subflow Bwd Byts": np.int32,
    "Init Fwd Win Byts": np.int32,
    "Init Bwd Win Byts": np.int32,
    "Fwd Act Data Pkts": np.int16,
    "Fwd Seg Size Min": np.int8,
    "Active Mean": np.float64,
    "Active Std": np.float64,
    "Active Max": np.int32,
    "Active Min": np.int32,
    "Idle Mean": np.float64,
    "Idle Std": np.float64,
    "Idle Max": np.int64,
    "Idle Min": np.int64,
    "Label": object,
}
used_cols = ids_datatypes.keys()


df1 = pd.read_csv(
    "/data/logs-datasets/cicids2018/02-14-2018.csv", dtype=ids_datatypes, usecols=used_cols,
)
df2 = pd.read_csv(
    "/data/logs-datasets/cicids2018/02-15-2018.csv", dtype=ids_datatypes, usecols=used_cols,
)
df3 = pd.read_csv(
    "/data/logs-datasets/cicids2018/02-16-2018.csv", dtype=ids_datatypes, usecols=used_cols,
)
df4 = pd.read_csv(
    "/data/logs-datasets/cicids2018/02-20-2018.csv", dtype=ids_datatypes, usecols=used_cols
)
df5 = pd.read_csv(
    "/data/logs-datasets/cicids2018/02-21-2018.csv", dtype=ids_datatypes, usecols=used_cols,
)
df6 = pd.read_csv(
    "/data/logs-datasets/cicids2018/02-22-2018.csv", dtype=ids_datatypes, usecols=used_cols,
)
df7 = pd.read_csv(
    "/data/logs-datasets/cicids2018/02-23-2018.csv", dtype=ids_datatypes, usecols=used_cols,
)
df8 = pd.read_csv(
    "/data/logs-datasets/cicids2018/02-28-2018.csv", dtype=ids_datatypes, usecols=used_cols,
)
df9 = pd.read_csv(
    "/data/logs-datasets/cicids2018/03-01-2018.csv", dtype=ids_datatypes, usecols=used_cols,
)
df10 = pd.read_csv(
    "/data/logs-datasets/cicids2018/03-02-2018.csv", dtype=ids_datatypes, usecols=used_cols,
)

dfs = [df1, df2, df3, df4, df5, df6, df7, df8, df9, df10]
df_combine = pd.concat(dfs)

nbins = 200

for idx, df in enumerate(dfs):
    print(idx)
    for c_idx, col in enumerate(used_cols):
        if col == "Label":
            continue

        #df[col] = pd.to_numeric(df[col])
        print(col, len(df_combine[col].unique()))

        maxval = df_combine.loc[df_combine[col] != np.inf, col].max()
        df[col].replace(np.inf, maxval, inplace=True)
        df_combine[col].replace(np.inf, maxval, inplace=True)

        if df_combine[col].nunique() > nbins:
            base = math.e ** (math.log(maxval) / nbins)
            bins = [base** i for i in range(nbins + 1)]
            bins[0] -= 1.001
            bins[-1] += 1.001

            df[col] = pd.cut(
                df[col],
                bins=bins,
                labels=[str(c_idx) + "|" + str(bin_idx) for bin_idx in range(len(bins)-1)],
            ).astype(str)
        else:
            df[col] = df[col].astype(str)

    df.to_pickle(f"/data/logs-datasets/cicids2018/cicids2018_{idx}.pkl")
