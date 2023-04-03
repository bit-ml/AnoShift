import sys
import os
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)) + '/..')

import pandas as pd
pd.options.mode.chained_assignment = None

from data_processor.data_loader import split_set
from transformers import PreTrainedTokenizerFast
from language_models.data_utils import prepare_train_ds, prepare_test_ds
from language_models.model_utils import configure_model, train_model


train_set_years = range(2006, 2011)
test_set_years = range(2006, 2016)

ds_size = "subset"
label_col_name = '18'
label_col_pos_val = '1'

# We only keep features 0 to 13
cols = [str(i) for i in range(14)]

def load_train_set(anoshift_db_path, years, ds_size):
    df = pd.DataFrame()
    keep_cols = cols

    for year in years:
        df_year_path = os.path.join(anoshift_db_path, f"subset/{year}_{ds_size}.parquet")

        print("Loading train set part:", df_year_path)
        df_year = pd.read_parquet(df_year_path)
        df_year = df_year[df_year[label_col_name] == label_col_pos_val]
        df_year = df_year.drop(columns=list(set(df_year.columns) - set(keep_cols)))
        
        df = pd.concat((df, df_year))
    return df_year


def load_test_set(anoshift_db_path, year, ds_size):
    keep_cols = cols + [label_col_name, ]
    if year < 2011:
        df_year_path = os.path.join(anoshift_db_path, f"subset/{year}_{ds_size}_valid.parquet")
    else:
        df_year_path = os.path.join(anoshift_db_path, f"subset/{year}_{ds_size}.parquet")

    print("Loading test set:", df_year_path)
    df_year = pd.read_parquet(df_year_path)
    df_year = df_year.drop(columns=list(set(df_year.columns) - set(keep_cols)))

    return df_year


if __name__ == "__main__":
    anoshift_db_path = sys.argv[1]
    df_train = load_train_set(anoshift_db_path, train_set_years, ds_size)

    tokenizer_path = 'saved_tokenizers/kyoto-2016.json'
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    tokenizer.add_special_tokens(
        {"pad_token": "[PAD]", "unk_token": "[UNK]", "mask_token": "[MASK]"}
    )


    lm_ds_train = prepare_train_ds(
        df_train=df_train, tokenizer=tokenizer, block_size=len(cols)
    )

    ds_test = []

    for year in test_set_years:
        df_test = load_test_set(anoshift_db_path, year, ds_size)

        # Split test set in inliers and outliers
        df_test_inlier, df_test_outlier = split_set(
            df_test, label_col_name=label_col_name, label_col_pos_val=label_col_pos_val
        )

        df_test_subset = [(year, df_test_inlier, df_test_outlier)]

        ds_test_subset = prepare_test_ds(
            dfs_test=df_test_subset, tokenizer=tokenizer, block_size=len(cols)
        )

        ds_test += ds_test_subset


    architecture = 'bert'
    pretrained = False
    vocab_size = len(tokenizer.get_vocab())
    bs_train = 256
    bs_eval = 256
    num_epochs = 5

    model_iid = configure_model(
            architecture=architecture,
            pretrained=pretrained,
            small=True,
            vocab_size=vocab_size,
            tokenizer=tokenizer,
            embed_size=len(cols)
        )


    print("Training iid model on set1")
    train_model(
        model=model_iid,
        tokenizer=tokenizer,
        ds_name='kyoto-2016',
        train_set_name=f'OOD_train_2006-2011',
        run_name='iid',
        lm_ds_train=lm_ds_train,
        lm_ds_eval=ds_test[0][1]['inlier'],
        dss_test=ds_test,
        save_model_path='/tmp/',
        batch_size_train=bs_train,
        batch_size_eval=bs_eval,
        num_epochs=num_epochs,
        tb_writer=None
    )

