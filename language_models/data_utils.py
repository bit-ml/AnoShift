from datasets import Dataset
import gc


ds_cols = ["input_ids", "idx", "service", "token_type_ids",
           "attention_mask", "labels", "count", 'protocol', 'connection', ]


def tokenize_function(examples, tokenizer, block_size):
    ex_cpy = {key: value for key, value in examples.items() if key not in ['idx', ]}

    tokenized = {"text": [" ".join(v)
                          for v in list(zip(*ex_cpy.values()))]}
    tokenized = tokenizer(
        tokenized["text"], padding="max_length", truncation=True, max_length=block_size
    )
    return tokenized


def prepare_test_ds_split(df_test_inlier, df_test_outlier, tokenizer, block_size):
    lm_ds_test_inlier = Dataset.from_pandas(df_test_inlier)
    lm_ds_test_outlier = Dataset.from_pandas(df_test_outlier)
    try:
        lm_ds_test_inlier = lm_ds_test_inlier.remove_columns(
            [
                "__index_level_0__",
            ]
        )
        lm_ds_test_outlier = lm_ds_test_outlier.remove_columns(
            [
                "__index_level_0__",
            ]
        )
    except Exception as ex:
        pass

    del df_test_inlier
    del df_test_outlier

    lm_ds_test_inlier = lm_ds_test_inlier.map(
        lambda ex: tokenize_function(ex, tokenizer, block_size),
        num_proc=4,
        batched=True,
        batch_size=256,
    ).remove_columns(
        [
            l
            for l in lm_ds_test_inlier.features.keys()
            if l not in ds_cols
        ]
    )

    lm_ds_test_outlier = lm_ds_test_outlier.map(
        lambda ex: tokenize_function(ex, tokenizer, block_size),
        num_proc=4,
        batched=True,
        batch_size=256,
    ).remove_columns(
        [
            l
            for l in lm_ds_test_outlier.features.keys()
            if l not in ds_cols
        ]
    )

    ds_test = {"inlier": lm_ds_test_inlier, "outlier": lm_ds_test_outlier}
    return ds_test


def train_df_to_ds(df_train):
    ds_train = Dataset.from_pandas(df_train)

    try:
        ds_train = ds_train.remove_columns(
            [
                "__index_level_0__",
            ]
        )
    except Exception as ex:
        pass

    del df_train
    return ds_train


def prepare_train_ds(df_train, tokenizer, block_size):
    ds_train = train_df_to_ds(df_train)
    gc.collect()

    print("Mapping tokenizer on train")
    ds_train = ds_train.map(
        lambda ex: tokenize_function(ex, tokenizer, block_size),
        num_proc=4,
        batched=True,
        batch_size=256,
    ).remove_columns(
        [
            l
            for l in ds_train.features.keys()
            if l not in ds_cols
        ]
    )
    print("Mapped tokenizer on train")

    gc.collect()
    return ds_train


def prepare_test_ds(dfs_test, tokenizer, block_size):
    gc.collect()
    ds_test = []

    for (test_part_name, df_test_split_inlier, df_test_split_outlier) in dfs_test:
        ds_test_part = prepare_test_ds_split(
            df_test_inlier=df_test_split_inlier,
            df_test_outlier=df_test_split_outlier,
            tokenizer=tokenizer,
            block_size=block_size,
        )
        ds_test.append((test_part_name, ds_test_part))
        del df_test_split_inlier
        del df_test_split_outlier
        gc.collect()

    return ds_test
