import torch
import pandas as pd
import numpy as np
from models.hgat import HGAT
from sklearn.metrics import f1_score
import argparse
from tools.load_hgat_data import load_data
import yaml
from yaml.loader import SafeLoader
from balanced_loss import Loss

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

    hgat_data = load_data(args.train_dir, device)

    train_dataset = pd.read_csv(args.train_csv)
    labels = train_dataset[args.label_column].values
    y_train = torch.tensor(labels, dtype=torch.int64).to(device)

    model_config = yaml.load(open(args.model_config), Loader=SafeLoader)

    in_channels = model_config["model"]["in_channels"]
    hidden_channels = model_config["model"]["hidden_channels"]
    out_channels = model_config["model"]["out_channels"]
    n_heads = model_config["model"]["n_heads"]
    dropout = model_config["model"]["dropout"]

    learning_rate = model_config["opt"]["learning_rate"]
    weight_decay = model_config["opt"]["weight_decay"]
    epochs = model_config["opt"]["epochs"]

    model = HGAT(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        heads=n_heads,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # class-balanced cross-entropy loss
    ce_loss = Loss(
        loss_type="cross_entropy",
        samples_per_class=[
            len(train_dataset) - sum(train_dataset["LABEL"]),
            sum(train_dataset["LABEL"]),
        ],
        class_balanced=True,
    )

    def train(loss_fn):
        model.train()
        out = model(*hgat_data)
        loss = loss_fn(out, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return float(loss)

    @torch.no_grad()
    def test():
        model.eval()
        pred = model(*hgat_data).cpu()
        pred = pred.argmax(dim=-1)

        acc = f1_score(pred, y_train.cpu())
        return acc

    for i in range(epochs):
        train_loss = train(ce_loss)
        print("Train Loss " + "e_" + str(i) + ": " + str(train_loss))

        if i % 10 == 0:
            tmp_test_acc = test()
            print("Test f1 score: " + str(tmp_test_acc))

    torch.save(
        {"model_state_dict": model.state_dict()},
        args.ckpt_path,
    )

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a Hierarchical GAT Network"
    )
    parser.add_argument(
        "--model_config", type=str, metavar="DIR", help="path to yaml"
    )
    parser.add_argument(
        "--train_dir",
        type=str,
        metavar="DIR",
        help="directory containing training data",
    )
    parser.add_argument(
        "--train_csv", type=str, metavar="DIR", help="training csv filename"
    )
    parser.add_argument(
        "--label_column", type=str, metavar="DIR", help="label column in csv"
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
    _args = parser.parse_args()
    main(_args)
