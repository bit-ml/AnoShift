from language_models.evaluation_utils import eval_rocauc
from data_processor.data_loader import load_local_dataset
from language_models.data_utils import prepare_train_ds, prepare_test_ds, train_df_to_ds
from language_models.tokenizer_utils import configure_tokenizer
from language_models.model_utils import configure_model, train_model, distil_model
import gc
import argparse
import logging
import random
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForMaskedLM
import torch

torch.cuda.empty_cache()
torch.cuda.memory_summary(device=None, abbreviated=False)

# torch.use_deterministic_algorithms(True)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__file__)

parser = argparse.ArgumentParser(
    description="Language Modelling with BERT parameters.")

parser.add_argument(
    "--ds",
    type=str,
    help="dataset name",
    default="kyoto-2016",
    choices=[
        "kyoto-2016",
    ],
)

parser.add_argument(
    "--architecture",
    type=str,
    help="arhictecture type",
    default="bert_small",
    choices=["bert", "bert_small", "electra", "lstm"],
)

parser.add_argument(
    "--experiment_type",
    type=str,
    help="experiment type",
    default="iid",
    choices=["iid", "finetune", "distil", ],
)


parser.add_argument(
    "--ds_size",
    type=str,
    help="dataset size: large or small",
    default="subset",
    choices=["full", "subset"],
)


parser.add_argument(
    "--contamination",
    type=float,
    help="percent of outliers in train data",
    default=0.0,
)

parser.add_argument(
    "--random_seed",
    type=int,
    help="percent of outliers in train data",
    default=42,
)


parser.add_argument(
    "--experiment_year",
    help="Year to train on. Only available for Kyoto dataset",
    type=str,
    default="",
    choices=[
        "",
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
    ],
)


parser.add_argument(
    "--epochs", help="Number of epochs to train", type=int, default=5)

parser.add_argument("--byte_level", action="store_true")
parser.add_argument("--pretrained", action="store_true")
parser.add_argument("--nolog", action="store_true")

args = parser.parse_args()

bs = 256
bs_eval = 512
num_epochs = args.epochs
ds_size = args.ds_size
random_seed = args.random_seed

torch.manual_seed(42)
random.seed(42)

byte_level_tokenization = args.byte_level
small = False
architecture = args.architecture
if architecture == "bert_small":
    architecture = "bert"
    small = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pretrained = args.pretrained
experiment_year = args.experiment_year
experiment_type = args.experiment_type
experiment_set = "principal"
contamination = args.contamination
nolog = args.nolog
print("Running", experiment_set, experiment_type)


ds = args.ds

if ds == "kyoto-2016":
    preload_tokenizer = True
else:
    preload_tokenizer = False

print("Loading datasets")
dfs_train, dfs_test = load_local_dataset(
    ds,
    experiment_year=experiment_year,
    experiment_set=experiment_set,
    experiment_type=experiment_type,
    contamination=contamination,
    ds_size=ds_size,
    random_seed=random_seed,
)
print("Loaded datasets")
block_size = dfs_test[0][1].shape[1]

if "HDFS" in ds:
    # HDFS has variable-length sequences
    block_size = 512  # max len train 273
    bs = 64
    bs_eval = 128

if "BGL" in ds:
    # BGL has variable-length sequences
    block_size = 1024
    bs = 16
    bs_eval = 16

if "Thunderbird" in ds:
    # BGL has variable-length sequences
    block_size = 1024
    bs = 16
    bs_eval = 16

if "spirit2" in ds:
    # BGL has variable-length sequences
    block_size = 1024
    bs = 16
    bs_eval = 16

if "kyoto" in ds:
    block_size = 14


if byte_level_tokenization is True:
    # increase block size for byte level tokenization
    block_size = 512

model_name = f"{architecture}"
if small:
    model_name += "_small"
if pretrained:
    model_name += "_pretrained"
if byte_level_tokenization:
    model_name += "_byte_level"
else:
    model_name += "_word_level"

train_part_name = ""
if len(dfs_train) > 0:
    train_part_name = dfs_train[0][0]

run_name = "[{}] Small_BERT_{}_{}_{}-{}-{}_contamination={}_nodups_nopadeval".format(
    train_part_name,
    ds,
    architecture,
    experiment_set,
    experiment_type,
    experiment_year,
    contamination,
)


writer = SummaryWriter(f"runs/{run_name}")

save_model_path = f"saved_models/{model_name}/{experiment_set}/{ds}_{ds_size}"

for df_train in dfs_train:
    print(df_train)
    save_model_path += f"_{df_train[0]}"

print("Model will be saved to: ", save_model_path)


def iid_experiment():
    print("Configuring tokenizer")
    tokenizer, vocab_size = configure_tokenizer(
        byte_level_tokenization=byte_level_tokenization,
        dfs_train=dfs_train,
        ds_name=ds,
    )
    print("Configured tokenizer")

    print("Preparing train ds")
    (df_name_train, df_train) = dfs_train[0]

    print("Train shape", df_train.shape)
    print(df_train)
    lm_ds_train = prepare_train_ds(
        df_train=df_train, tokenizer=tokenizer, block_size=block_size
    )

    print("Prepared train ds")
    gc.collect()

    print("Preparing test ds")
    dss_test = prepare_test_ds(
        dfs_test=dfs_test, tokenizer=tokenizer, block_size=block_size
    )

    print("Prepared test ds")
    gc.collect()

    print("Configuring model")
    model = configure_model(
        architecture=architecture,
        pretrained=pretrained,
        small=small,
        vocab_size=vocab_size,
        tokenizer=tokenizer,
        embed_size=block_size,
    )

    print("Training model")
    train_model(
        model=model,
        tokenizer=tokenizer,
        ds_name=ds,
        train_set_name=df_name_train,
        run_name=run_name,
        lm_ds_train=lm_ds_train,
        lm_ds_eval=dss_test[0][1]["inlier"],
        dss_test=dss_test,
        save_model_path=save_model_path,
        batch_size_train=bs,
        batch_size_eval=bs_eval,
        num_epochs=num_epochs,
        tb_writer=writer,
    )


def finetune_experiment():
    """
    Finetune experiment for one year requires the checkpoint of a finetune model on the previous year.
    Run as:
    python main.py --experiment_type=finetune --experiment_year=2006
    python main.py --experiment_type=finetune --experiment_year=2007
    and so on.
    """
    if experiment_year == '2006':
        return iid_experiment()

    print("Configuring tokenizer")
    tokenizer, vocab_size = configure_tokenizer(
        byte_level_tokenization=byte_level_tokenization,
        dfs_train=dfs_train,
        ds_name=ds,
        preload=True,
    )
    print("Configured tokenizer")

    dss_test = prepare_test_ds(
        dfs_test=dfs_test, tokenizer=tokenizer, block_size=block_size
    )
    print("Prepared test ds")

    prev_model_path = f"saved_models/{model_name}/{experiment_set}/{ds}_{ds_size}_{experiment_set}_{experiment_type}_{int(experiment_year)-1}_final"

    print("Loading model from ", prev_model_path)
    model = AutoModelForMaskedLM.from_pretrained(prev_model_path).cuda()
    print("Loaded model")

    num_experiment_years = len(dfs_train)
    for idx, (df_name_train, df_train) in enumerate(dfs_train):
        print("Training on: ", df_name_train, df_train.shape)
        lm_ds_train = prepare_train_ds(
            df_train=df_train, tokenizer=tokenizer, block_size=block_size
        )

        ds_test_step = [
            dss_test[idx],
        ] + dss_test[num_experiment_years:]
        print([d[0] for d in ds_test_step])

        model = train_model(
            model=model,
            tokenizer=tokenizer,
            ds_name=ds,
            train_set_name=df_name_train,
            run_name=run_name,
            lm_ds_train=lm_ds_train,
            lm_ds_eval=dss_test[0][1]["inlier"],
            dss_test=dss_test,
            save_model_path=save_model_path,
            batch_size_train=bs,
            batch_size_eval=bs_eval,
            num_epochs=num_epochs,
            tb_writer=writer,
        )


def distil_experiment():
    """
    Distil experiment for one year requires the checkpoint of a distil model on the previous year.
    Run as:
    python main.py --experiment_type=distil --experiment_year=2006
    python main.py --experiment_type=distil --experiment_year=2007
    and so on.
    """
    if experiment_year == '2006':
        return iid_experiment()

    student_dfs_train = dfs_train
    _, df_train = student_dfs_train[0]
    ds_train = train_df_to_ds(df_train)

    tokenizer, vocab_size = configure_tokenizer(
        byte_level_tokenization=byte_level_tokenization,
        dfs_train=student_dfs_train,
        ds_name=ds,
        preload=True,
    )
    print("Configured tokenizer")

    prev_model_path = f"saved_models/{model_name}/{experiment_set}/{ds}_{ds_size}_{experiment_set}_{experiment_type}_{int(experiment_year)-1}_final"

    print("Loading model from ", prev_model_path)

    teacher_model = AutoModelForMaskedLM.from_pretrained(
        prev_model_path).cuda()
    teacher_model.eval()

    dss_test = prepare_test_ds(
        dfs_test=dfs_test, tokenizer=tokenizer, block_size=block_size
    )

    print("Prepared ds test")

    student_model = configure_model(
        architecture=architecture,
        pretrained=pretrained,
        small=small,
        vocab_size=vocab_size,
        tokenizer=tokenizer,
        embed_size=block_size,
    )

    student_model = distil_model(
        teacher=teacher_model,
        student=student_model,
        tokenizer=tokenizer,
        ds_train=ds_train,
        dss_test=dss_test,
        save_model_path=save_model_path,
        batch_size_train=bs,
        batch_size_eval=bs_eval,
        num_epochs=num_epochs,
        tb_writer=writer,
    )


if __name__ == "__main__":
    if experiment_type == "iid":
        iid_experiment()
    elif experiment_type == "finetune":
        finetune_experiment()
    elif experiment_type == "distil":
        distil_experiment()
