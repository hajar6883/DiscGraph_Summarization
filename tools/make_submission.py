"""
This script converts test_labels.json into submission.csv
python make_submission.py --json_path test_labels_naive_baseline.json
"""
import json
from pathlib import Path


def make_submission(json_path: Path = Path("test_labels_text_baseline.json")):
    with open(json_path, "r") as file:
        test_labels = json.load(file)

    file = open("submission.csv", "w")
    file.write("id,target_feature\n")
    for key, value in test_labels.items():
        u_id = [key + "_" + str(i) for i in range(len(value))]
        target = map(str, value)
        for row in zip(u_id, target):
            file.write(",".join(row))
            file.write("\n")
    file.close()


def make_submission_from_df(dataframe, path):
    file = open(path, "w")

    file.write("id,target_feature\n")
    index_table = dict()

    for _, row in dataframe.iterrows():
        key = row["ID"]
        value = row["y_pred"]

        if key in index_table:
            index = index_table[key] + 1
        else:
            index = 0

        index_table[key] = index

        u_id = key + "_" + str(index)

        file.write(f"{u_id},{value}")
        file.write("\n")

    file.close()
