import pandas as pd
import numpy as np


def listToInt(mylist, codetoi):
    return [codetoi[item] for item in mylist]


# Multihot, for single list - one row
def multihot(tags, classcodes):
    taglist = list(classcodes.index)
    return [1 if tag in tags else 0 for tag in taglist]


def main():
    reuters = pd.read_pickle("input/reuters_small.pkl")
    # read classcodes
    classcodes = pd.read_csv("input/classcodes.csv")
    # add index field to DataFrame
    classcodes = classcodes.reset_index()

    # Create dictionary index/int to classcode and classcode to int
    itocode = dict(zip(classcodes.index, classcodes.Code))
    codetoi = dict(zip(classcodes.Code, classcodes.index))

    reuters["codes"] = [listToInt(codelist, codetoi) for codelist in reuters.codes]

    # list of classes, 126 int: [0...125]
    Y_hot = [multihot(claslist, classcodes) for claslist in reuters.codes]
    reuters["codes"] = Y_hot

    # Put data in random order

    idx = np.random.permutation(len(reuters))
    reuters = reuters.iloc[idx]

    # split it
    size = len(reuters)
    train_size = int(0.7 * size)
    test_size = int(0.85 * size)

    train = reuters[0:train_size]
    val = reuters[train_size:test_size]
    test = reuters[test_size:size]

    train.to_json("input/train.json", orient="records", lines=True)
    test.to_json("input/test.json", orient="records", lines=True)
    val.to_json("input/val.json", orient="records", lines=True)


if __name__ == "__main__":
    main()
