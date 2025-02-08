from torch.utils.data import Dataset
import pandas as pd


class SimpleDataset(Dataset):
    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx]


class CSVDataset(Dataset):
    def __init__(self, csv_file, feature_columns, label_column):
        df = pd.DataFrame(csv_file)
        self.x_train = df[feature_columns]
        self.y_train = df[label_column]

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx]
