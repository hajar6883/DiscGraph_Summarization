from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertModel
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
import torch
from autocorrect import Speller
import re
import nltk

words = set(nltk.corpus.words.words())

tqdm.pandas()


def main(args):
    bert_model = SentenceTransformer(args.model)

    df = pd.read_csv(args.filename)
    text = df[args.column]

    spell = Speller(fast=True)
    text = text.progress_apply(spell)

    text = text.progress_apply(
        lambda x: re.sub("<[^>]+>", "", x.lower()).strip()
    )
    text = text.progress_apply(lambda x: re.sub(r"[^\w\s]", "", x))
    text = text.progress_apply(lambda x: re.sub(r"um", "", x))
    text = text.progress_apply(lambda x: re.sub(r"uh", "", x))
    text = text.progress_apply(lambda x: re.sub(r"rry", "", x))
    text = text.progress_apply(lambda x: re.sub(r"mm", "", x))
    text = text.progress_apply(lambda x: re.sub(r"mmhmm", "", x))
    text = text.progress_apply(lambda x: re.sub(r"well", "", x))
    text = text.progress_apply(lambda x: re.sub(r"okay", "", x))
    text = text.progress_apply(lambda x: re.sub(r"by the way", "", x))
    text = text.progress_apply(lambda x: re.sub(r"yeah", "", x))
    text = text.progress_apply(lambda x: re.sub(r"no", "", x))
    text = text.progress_apply(lambda x: re.sub(r"so", "", x))
    text = text.apply(lambda x: re.sub(" +", " ", x.strip()))
    text = text.apply(lambda x: re.sub(r"(\w+) \1", r"\1", x))

    masked_indices = []
    for _, t in tqdm(enumerate(text)):
        masked_indices.append(len(t.split()) > 2)

    text = text[masked_indices].reset_index(drop=True)

    extracted_text = bert_model.encode(text, show_progress_bar=True)
    np.save(args.save_path, np.array(extracted_text))
    np.save(args.save_path.split(".")[0] + "_mask", np.array(masked_indices))

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # model = BertModel.from_pretrained("bert-base-uncased").to(device)

    # features = np.zeros((len(text), 768))

    # for i, t in enumerate(tqdm(text)):
    #     encoded_input = tokenizer(t, return_tensors='pt').to(device)
    #     output = model(**encoded_input)
    #     features[i] = output.pooler_output[0].detach().cpu().numpy()

    # np.save(args.save_path, features)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract text features using the bert model"
    )
    parser.add_argument(
        "--model", type=str, help="name of the underlying model"
    )
    parser.add_argument(
        "--filename", type=str, metavar="DIR", help="csv filename"
    )
    parser.add_argument(
        "--column", type=str, metavar="DIR", help="csv text column"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        metavar="DIR",
        help="path where the features will be saved",
    )
    _args = parser.parse_args()
    main(_args)
