from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers import Tokenizer, trainers
from tokenizers import ByteLevelBPETokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast
import os


unk_token = "[UNK]"
pad_token = "[PAD]"
mask_token = "[MASK]"
tokenizers_dir = "./saved_tokenizers/"


def configure_tokenizer(
    byte_level_tokenization, dfs_train, ds_name, tokenizer_name="customtokenizer", preload=True
):
    if preload and ds_name == "kyoto-2016":
        print("Loading presaved tokenizer")
        tokenizer_path = tokenizers_dir + ds_name + ".json"
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
        tokenizer.add_special_tokens(
            {"pad_token": "[PAD]", "unk_token": "[UNK]",
             "mask_token": "[MASK]"}
        )
        return tokenizer, tokenizer._tokenizer.get_vocab_size()

    if byte_level_tokenization is True:
        print("Training byte level tokenizer")
        custom_tokenizer = ByteLevelBPETokenizer()
        custom_tokenizer.train(
            files=["./datasets/{}/train/inlier.txt".format(ds_name)],
            special_tokens=[pad_token, unk_token],
        )
    else:
        print("Training word level tokenizer with whitespace delimiter")
        custom_tokenizer = Tokenizer(WordLevel(unk_token=unk_token))
        custom_tokenizer.add_tokens([unk_token, pad_token])

        trainer = trainers.WordLevelTrainer(
            vocab_size=30000, special_tokens=[pad_token, unk_token, mask_token]
        )

        def batch_iterator():
            for _, df_train in dfs_train:
                for i in range(0, len(df_train)):
                    yield " ".join([tok for tok in df_train.iloc[i]])

        custom_tokenizer.pre_tokenizer = WhitespaceSplit()
        custom_tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)

    if not os.path.isdir(tokenizers_dir):
        os.mkdir(tokenizers_dir)

    tokenizer_path = tokenizers_dir + ds_name + "_" + tokenizer_name + ".json"
    vocab_size = custom_tokenizer.get_vocab_size()
    print("Saving tokenizer to " + tokenizer_path)
    custom_tokenizer.save(tokenizer_path)
    print("Vocabulary size: " + str(vocab_size))

    # Load it using transformers
    custom_tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    custom_tokenizer.add_special_tokens(
        {"pad_token": "[PAD]", "unk_token": "[UNK]",
         "mask_token": "[MASK]"}
    )

    return custom_tokenizer, vocab_size
