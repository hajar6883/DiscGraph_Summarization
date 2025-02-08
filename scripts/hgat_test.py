import torch
import pandas as pd
import numpy as np
from models.hgat import HGAT
import argparse
from tools.load_hgat_data import load_data
import yaml
from yaml.loader import SafeLoader
from tools.make_submission import make_submission_from_df

import torch
import random


def _seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args):
    _seed()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    hgat_data = load_data(args.test_dir, device)

    test_dataset = pd.read_csv(args.test_csv)

    model_config = yaml.load(open(args.model_config), Loader=SafeLoader)

    in_channels = model_config["model"]["in_channels"]
    hidden_channels = model_config["model"]["hidden_channels"]
    out_channels = model_config["model"]["out_channels"]
    n_heads = model_config["model"]["n_heads"]
    dropout = model_config["model"]["dropout"]

    model = HGAT(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        heads=n_heads,
        dropout=dropout,
    ).to(device)

    checkpoint = torch.load(args.ckpt_path)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    with torch.no_grad():
        y_out = model(*hgat_data).detach().cpu()

    y_out = y_out.argmax(dim=-1)
    test_dataset["y_pred"] = y_out

    make_submission_from_df(test_dataset, args.out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a Hierarchical GAT Network"
    )
    parser.add_argument(
        "--model_config", type=str, metavar="DIR", help="path to yaml"
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        metavar="DIR",
        help="directory containing test data",
    )
    parser.add_argument(
        "--test_csv", type=str, metavar="DIR", help="test csv filename"
    )
    parser.add_argument(
        "--device",
        type=str,
        metavar="DIR",
        help="GPU or CPU to run the model on",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        metavar="DIR",
        help="path where the model checkpoint will be saved",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        metavar="DIR",
        help="path where the model checkpoint will be saved",
    )
    _args = parser.parse_args()
    main(_args)
