from pathlib import Path
import csv

path_to_training = Path("data/training")
path_to_test = Path("data/test")


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

edge_list = [
    "Alternation",
    "Narration",
    "Question-answer_pair",
    "Acknowledgement",
    "Elaboration",
    "Result",
    "Correction",
    "Comment",
    "Contrast",
    "Explanation",
    "Clarification_question",
    "Q-Elab",
    "Background",
    "Continuation",
    "Conditional",
    "Parallel",
]

edge_from = []
edge_to = []
edge_types = []

i = 0

for transcription_id in training_set:
    f = open(path_to_training / f"{transcription_id}.txt", "r")
    lines = [line.split() for line in f.readlines()]
    for line in lines:
        edge_from.append(int(line[0]) + i)
        edge_to.append(int(line[2]) + i)
        edge_attr = [0 for _ in range(len(edge_list) + 1)]
        edge_attr[edge_list.index(line[1])] = 1
        edge_types.append(edge_attr)
    for j in range(len(lines) - 1):
        edge_from.append(j + i)
        edge_to.append(j + i + 1)
        edge_attr = [0 for _ in range(len(edge_list) + 1)]
        edge_attr[16] = 1
        edge_types.append(edge_attr)
    i += len(lines)

with open(f"out/features/train_edge_indices.csv", "w") as f:
    # using csv.writer method from CSV package
    write = csv.writer(f)

    write.writerows([edge_from, edge_to])


with open("out/features/train_edge_types.csv", "w") as f:
    # using csv.writer method from CSV package
    write = csv.writer(f)

    write.writerows(edge_types)

edge_from = []
edge_to = []
edge_types = []

i = 0

for transcription_id in test_set:
    f = open(path_to_test / f"{transcription_id}.txt", "r")
    lines = [line.split() for line in f.readlines()]
    for line in lines:
        edge_from.append(int(line[0]) + i)
        edge_to.append(int(line[2]) + i)
        edge_attr = [0 for _ in range(len(edge_list) + 1)]
        edge_attr[edge_list.index(line[1])] = 1
        edge_types.append(edge_attr)
    for j in range(len(lines) - 1):
        edge_from.append(j + i)
        edge_to.append(j + i + 1)
        edge_attr = [0 for _ in range(len(edge_list) + 1)]
        edge_attr[16] = 1
        edge_types.append(edge_attr)
    i += len(lines)

with open(f"out/features/test_edge_indices.csv", "w") as f:
    # using csv.writer method from CSV package
    write = csv.writer(f)

    write.writerows([edge_from, edge_to])


with open("out/features/test_edge_types.csv", "w") as f:
    # using csv.writer method from CSV package
    write = csv.writer(f)

    write.writerows(edge_types)