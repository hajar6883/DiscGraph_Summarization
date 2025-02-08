import pandas as pd
import matplotlib.pyplot as plt
import re
import json
import csv
import numpy as np

import nltk

words = set(nltk.corpus.words.words())

train_csv = pd.read_csv("train.csv")
text = train_csv["TEXT"]
text = text.apply(lambda x: re.sub("<[^>]+>", "", x.lower()))
text = text.apply(lambda x: re.sub(r"[^\w\s]", "", x))
text = text.apply(lambda x: re.sub(r"um", "", x))
text = text.apply(lambda x: re.sub(r"uh", "", x))
text = text.apply(lambda x: re.sub(r"rry", "", x))
text = text.apply(lambda x: re.sub(r"mm", "", x))
text = text.apply(lambda x: re.sub(r"mmhmm", "", x))
text = text.apply(lambda x: re.sub(r"well", "", x))
text = text.apply(lambda x: re.sub(r"okay", "", x))
text = text.apply(lambda x: re.sub(r"by the way", "", x))
text = text.apply(lambda x: re.sub(r"yeah", "", x))
text = text.apply(lambda x: re.sub(r"no", "", x))
text = text.apply(lambda x: re.sub(r"so", "", x))
text = text.apply(lambda x: re.sub(" +", " ", x.strip()))
text = text.apply(lambda x: re.sub(r"(\w+) \1", r"\1", x))
train_csv["TEXT"] = text
train_csv["LEN"] = train_csv["TEXT"].apply(lambda x: len(x.split()))

print(
    train_csv.groupby(["TEXT"])["TEXT"]
    .count()
    .sort_values(ascending=False)[:20]
)

subset = train_csv[train_csv["LEN"] > 2]
s = sum(subset["LABEL"])

print(
    subset.groupby(["TEXT"])["TEXT"].count().sort_values(ascending=False)[:20]
)

tot_sum = sum(train_csv["LABEL"])

print(len(subset))
print(s)

print(s / len(subset))
print(tot_sum / len(train_csv))

print(len(subset) / len(train_csv))
print(s / tot_sum)


# def load_edges(edge_path):
#     file = open(edge_path, "r")

#     edge_indices = list(csv.reader(file, delimiter=","))
#     edge_indices = [[int(x) for x in row] for row in edge_indices]

#     return edge_indices

# from os import listdir
# from os.path import isfile, join
# onlyfiles = [f for f in listdir("data/training/") if isfile(join("data/training/", f))]

# for file in onlyfiles:
#     if file.endswith(".csv"):
#         edges = np.array(load_edges("data/training/" + file))
#         lower = 0
#         higher = 0
#         for edge in edges:
#             if np.sum(edge) < 3:
#                 lower += 1
#             else:
#                 higher += 1
#         print(lower)
#         print(higher)
# tt = 0
# ts = 0

# for t in train_csv.groupby(['TEXT'])['TEXT'].count().reset_index(name='count').sort_values(['count'], ascending=False)['TEXT'][:20]:
#     subset = train_csv[train_csv['TEXT'] == t]
#     s = sum(subset['LABEL'])

#     tt += s
#     ts += len(subset)
#     print(s / tot_sum)

# print('----------')

# print(ts / len(train_csv))
# print(tt / tot_sum)
