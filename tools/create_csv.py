import pandas as pd
import json
from pathlib import Path
import os

path_to_training = os.path.abspath("data/training")
path_to_test = os.path.abspath("data/test")


def flatten(list_of_list):
    return [item for sublist in list_of_list for item in sublist]


training_set = [
    "ES2002",
    "ES2005",
    "ES2006",
    "ES2007",
    "ES2008",
    "ES2009",
    "ES2010",
    "ES2012",
    "ES2013",
    "ES2015",
    "ES2016",
    "IS1000",
    "IS1001",
    "IS1002",
    "IS1003",
    "IS1004",
    "IS1005",
    "IS1006",
    "IS1007",
    "TS3005",
    "TS3008",
    "TS3009",
    "TS3010",
    "TS3011",
    "TS3012",
]
training_set = flatten(
    [[m_id + s_id for s_id in "abcd"] for m_id in training_set]
)
training_set.remove("IS1002a")
training_set.remove("IS1005d")
training_set.remove("TS3012c")

test_set = [
    "ES2003",
    "ES2004",
    "ES2011",
    "ES2014",
    "IS1008",
    "IS1009",
    "TS3003",
    "TS3004",
    "TS3006",
    "TS3007",
]
test_set = flatten([[m_id + s_id for s_id in "abcd"] for m_id in test_set])

test_series = []

for transcription_id in test_set:
    file_path = os.path.join(path_to_test, f'{transcription_id}.json')
    with open(file_path, 'r') as file:
        transcription = json.load(file)
        for row in transcription:
            test_series.append(
                {
                    "ID": transcription_id,
                    "SPEAKER": row["speaker"],
                    "TEXT": row["text"],
                }
            )

test_df = pd.DataFrame(test_series)

training_series = []
file_path = os.path.abspath("data/training/training_labels.json")
with open(file_path, 'r') as file:
    training_labels = json.load(file)
for transcription_id in training_set:
    file_path = os.path.join(path_to_training, f'{transcription_id}.json')
    with open(file_path, 'r') as file:
        transcription = json.load(file)
        for i in range(len(transcription)):
            row = transcription[i]
            training_series.append(
                {
                    "ID": transcription_id,
                    "SPEAKER": row["speaker"],
                    "TEXT": row["text"],
                    "LABEL": training_labels[transcription_id][i],
                }
            )
    file_path = os.path.join(path_to_training, f'{transcription_id}.txt')
    f =open(file_path, 'r')
    lines = f.readlines()

train_df = pd.DataFrame(training_series)
print("creation csv")
test_df.to_csv('data/test/test.csv', index=False)
train_df.to_csv('data/training/train.csv', index=False)
