import argparse
import os
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit


parser = argparse.ArgumentParser(description="Vocabulary generator")
tokenizers_dir = "saved_tokenizers/"

unk_token = "[UNK]"
pad_token = "[PAD]"
mask_token = "[MASK]"

parser.add_argument(
    "--ds",
    type=str,
    help="dataset name",
    default="kyoto-2016",
    choices=[
        "kyoto-2016",
    ],
)


def generate_kyoto_tokenizer(num_bins=233):
    all_tokens = [unk_token, pad_token, mask_token]

    for col_idx in [0, 2, 3]:
        all_tokens += ["c" + str(col_idx) + str(i) for i in range(num_bins)]

    # Add protocol tokens
    all_tokens += [
        "dns,sip",
        "dhcp",
        "other",
        "sip",
        "http,socks",
        "smtp,ssl",
        "http,irc",
        "socks",
        "ssh",
        "http,ssh",
        "http",
        "krb_tcp",
        "irc",
        "radius",
        "ftp-data",
        "snmp",
        "dns",
        "pop3",
        "ftp",
        "krb",
        "smtp",
        "ssl",
        "rdp",
    ]

    all_tokens += [
        "0.0",
        "0.00",
        "0.01",
        "0.02",
        "0.03",
        "0.04",
        "0.05",
        "0.06",
        "0.07",
        "0.08",
        "0.09",
        "0.10",
        "0.1",
        "0.11",
        "0.12",
        "0.13",
        "0.14",
        "0.15",
        "0.16",
        "0.17",
        "0.18",
        "0.19",
        "0.20",
        "0.2",
        "0.21",
        "0.22",
        "0.23",
        "0.24",
        "0.25",
        "0.26",
        "0.27",
        "0.28",
        "0.29",
        "0.30",
        "0.3",
        "0.31",
        "0.32",
        "0.33",
        "0.34",
        "0.35",
        "0.36",
        "0.37",
        "0.38",
        "0.39",
        "0.40",
        "0.4",
        "0.41",
        "0.42",
        "0.43",
        "0.44",
        "0.45",
        "0.46",
        "0.47",
        "0.48",
        "0.49",
        "0.50",
        "0.5",
        "0.51",
        "0.52",
        "0.53",
        "0.54",
        "0.55",
        "0.56",
        "0.57",
        "0.58",
        "0.59",
        "0.60",
        "0.6",
        "0.61",
        "0.62",
        "0.63",
        "0.64",
        "0.65",
        "0.66",
        "0.67",
        "0.68",
        "0.69",
        "0.70",
        "0.7",
        "0.71",
        "0.72",
        "0.73",
        "0.74",
        "0.75",
        "0.76",
        "0.77",
        "0.78",
        "0.79",
        "0.80",
        "0.8",
        "0.81",
        "0.82",
        "0.83",
        "0.84",
        "0.85",
        "0.86",
        "0.87",
        "0.88",
        "0.89",
        "0.90",
        "0.9",
        "0.91",
        "0.92",
        "0.93",
        "0.94",
        "0.95",
        "0.96",
        "0.97",
        "0.98",
        "0.99",
        "1.00",
        "1.0"
    ]

    all_tokens += [str(i) for i in range(0, 101)]
    all_tokens += [
        "OTH",
        "REJ",
        "RSTO",
        "RSTOS0",
        "RSTR",
        "RSTRH",
        "S0",
        "S1",
        "S2",
        "S3",
        "SF",
        "SH",
        "SHR",
    ]

    wl = WordLevel({tok: idx for idx, tok in enumerate(
        all_tokens)}, unk_token=unk_token)

    kyoto_tokenizer = Tokenizer(wl)
    kyoto_tokenizer.pre_tokenizer = WhitespaceSplit()

    print(kyoto_tokenizer.get_vocab_size())
    return kyoto_tokenizer


def generate_tokenizer(ds):
    if ds == "kyoto-2016":
        return generate_kyoto_tokenizer()


if __name__ == "__main__":
    if not os.path.isdir(tokenizers_dir):
        os.mkdir(tokenizers_dir)

    args = parser.parse_args()
    ds = args.ds

    tokenizer = generate_tokenizer(ds)
    tokenizer_path = tokenizers_dir + ds + ".json"
    tokenizer.save(tokenizer_path)
