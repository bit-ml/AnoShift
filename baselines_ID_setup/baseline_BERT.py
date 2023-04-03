import sys
import os
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)) + '/..')

import pandas as pd
pd.options.mode.chained_assignment = None

from data_processor.data_loader import split_set
from language_models.data_utils import prepare_train_ds, prepare_test_ds
from transformers import PreTrainedTokenizerFast

from language_models.model_utils import configure_model, train_model


years = range(2006, 2016)

ds_size = "subset"
label_col_name = '18'
label_col_pos_val = '1'


def prepare_set(anoshift_db_path, year, ds_size, is_test_set=False):
    keep_cols = [str(i) for i in range(14)]
    
    if is_test_set:
        keep_cols += [label_col_name, ]
        df_year_path = os.path.join(anoshift_db_path, f"{ds_size}/{year}_{ds_size}_valid.parquet")
        df_year = pd.read_parquet(df_year_path)
    else:
        df_year_path = os.path.join(anoshift_db_path, f"{ds_size}/{year}_{ds_size}.parquet")
        df_year = pd.read_parquet(df_year_path)
        df_year = df_year[df_year[label_col_name] == label_col_pos_val]
        

    print("Loading set:", df_year_path)
    df_year = df_year.drop(columns=list(set(df_year.columns) - set(keep_cols)))
    return df_year


if __name__ == "__main__":
    anoshift_db_path = sys.argv[1]

    df_train = pd.DataFrame([])

    for year in years:
        df_train = pd.concat([df_train, prepare_set(anoshift_db_path, year, ds_size)])

    print(df_train.head())

    tokenizer_path = 'saved_tokenizers/kyoto-2016.json'
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    tokenizer.add_special_tokens(
        {"pad_token": "[PAD]", "unk_token": "[UNK]", "mask_token": "[MASK]"}
    )


    ncols = df_train.shape[1]

    lm_ds_train = prepare_train_ds(
        df_train=df_train, tokenizer=tokenizer, block_size=ncols
    )

    ds_test = []


    for year in years:
        df_test = prepare_set(anoshift_db_path, year, ds_size, is_test_set=True)

        # Split test set in inliers and outliers
        df_test_inlier, df_test_outlier = split_set(
            df_test, label_col_name=label_col_name, label_col_pos_val=label_col_pos_val
        )

        df_test = [(year, df_test_inlier, df_test_outlier),]

        ds_test += prepare_test_ds(
            dfs_test=df_test, tokenizer=tokenizer, block_size=ncols
        )

    architecture = 'bert'
    pretrained = False
    vocab_size = len(tokenizer.get_vocab())
    bs_train = 256
    bs_eval = 256
    num_epochs = 10

    model_iid = configure_model(
            architecture=architecture,
            pretrained=pretrained,
            small=True,
            vocab_size=vocab_size,
            tokenizer=tokenizer,
            embed_size=ncols
        )

    print("Training ID model")
    train_model(
        model=model_iid,
        tokenizer=tokenizer,
        ds_name='kyoto-2016',
        train_set_name='ID',
        run_name='id',
        lm_ds_train=lm_ds_train,
        lm_ds_eval=ds_test[0][1]['inlier'],
        dss_test=ds_test,
        save_model_path='/tmp/',
        batch_size_train=bs_train,
        batch_size_eval=bs_eval,
        num_epochs=num_epochs,
        tb_writer=None
    )
