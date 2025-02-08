import argparse
import random

import lightning as L
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from yaml.loader import SafeLoader
from lightning.pytorch.loggers import WandbLogger

from models.nn import NN
from models.nn_ext import NN_EXT
from models.gat import GAT
from tools.dataset import SimpleDataset
from tools.make_submission import make_submission_from_df
import torch.nn.functional as F
from sklearn.metrics import f1_score


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

    # train_features = np.load(args.train_data)
    x_train = torch.load(args.train_data)
    train_mask = np.load(
        "out/features/train_pp.npy".split(".")[0] + "_mask.npy"
    )

    # test_features = np.load(args.test_data)
    x_test = torch.load(args.test_data)
    # test_mask = np.load(args.test_data.split(".")[0] + "_mask.npy")

    total_x = F.normalize(torch.cat((x_train, x_test), dim=0), dim=1)
    x_train = total_x[: len(x_train)].to(device=device)
    x_test = total_x[len(x_train) :]

    labels = pd.read_csv(args.labels)[args.label_column].values[train_mask]
    y_train = (
        torch.tensor(labels, dtype=torch.float32)
        .reshape((-1, 1))
        .to(device=device)
    )

    model_config = yaml.load(open(args.model_config), Loader=SafeLoader)

    input_dim = model_config["model"]["input_dim"]
    hidden_dim = model_config["model"]["hidden_dim"]
    output_dim = model_config["model"]["output_dim"]
    hidden_layers = model_config["model"]["hidden_layers"]
    dropout = model_config["model"]["dropout"]

    learning_rate = model_config["opt"]["learning_rate"]
    weight_decay = model_config["opt"]["weight_decay"]
    epochs = model_config["opt"]["epochs"]

    batch_size = model_config["loader"]["batch_size"]

    model = NN_EXT(
        in_channels=input_dim,
        hidden_layers=hidden_layers,
        hidden_channels=hidden_dim,
        out_channels=output_dim,
        dropout=dropout,
    ).to(device=device)

    dataset = SimpleDataset(x_train, y_train)

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [int(len(x_train) * 0.8), len(x_train) - int(len(x_train) * 0.8)],
    )

    training_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        dataset=val_dataset, batch_size=batch_size, shuffle=False
    )

    wandb_logger = WandbLogger(project="iml-challenge")
    trainer = L.Trainer(
        max_epochs=epochs, logger=wandb_logger, enable_checkpointing=False
    )
    trainer.fit(
        model=model,
        train_dataloaders=training_loader,
        val_dataloaders=val_loader,
    )
    trainer.save_checkpoint(f"ckpt/{args.model}.ckpt")

    # with torch.no_grad():
    #     y_pred = model.forward(x_test).cpu().numpy()
    #     y_pred = [0 if x <= 0 else 1 for x in y_pred]
    #     test_csv = pd.read_csv("test.csv")

    #     # y_pred_full = []
    #     # index = 0
    #     # for val in test_mask:
    #     #     if val:
    #     #         y_pred_full.append(y_pred[index])
    #     #         index += 1
    #     #     else:
    #     #         y_pred_full.append(0)

    #     test_csv["y_pred"] = y_pred # y_pred_full

    #     make_submission_from_df(test_csv, args.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and predict")
    parser.add_argument(
        "--model", type=str, help="name of the underlying model"
    )
    parser.add_argument(
        "--model_config", type=str, metavar="DIR", help="path to yaml"
    )
    parser.add_argument(
        "--train_data", type=str, metavar="DIR", help="npy filename"
    )
    parser.add_argument(
        "--labels", type=str, metavar="DIR", help="csv filename"
    )
    parser.add_argument(
        "--label_column", type=str, metavar="DIR", help="label column"
    )
    parser.add_argument(
        "--test_data", type=str, metavar="DIR", help="npy filename"
    )
    parser.add_argument(
        "--device",
        type=str,
        metavar="DIR",
        help="GPU or CPU to run the model on",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        metavar="DIR",
        help="path where the predictions will be saved",
    )
    _args = parser.parse_args()
    main(_args)
