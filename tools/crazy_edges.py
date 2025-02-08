import numpy as np
import pandas as pd
import csv
import re
from autocorrect import Speller
from tqdm import tqdm
import argparse

from nltk.stem.snowball import SnowballStemmer
from sentence_transformers import SentenceTransformer

from nltk import word_tokenize
from nltk.util import ngrams

def main(args):
    stemmer = SnowballStemmer("english")

    file = open(f"out/features/{args.data}_edge_indices.csv", "r")
    s2s_indices = list(csv.reader(file, delimiter=","))
    s2s_indices = [[int(x) for x in row] for row in s2s_indices]

    file = open(f"out/features/{args.data}_edge_types.csv", "r")
    s2s_types = list(csv.reader(file, delimiter=","))
    s2s_types = [[int(x) for x in row] for row in s2s_types]

    if args.data == "train":
        df = pd.read_csv("data/training/train.csv")
    if args.data == "test":
        df = pd.read_csv("data/test/test.csv")
    texts = df["TEXT"]

    spell = Speller(fast=True)
    texts = texts.apply(spell)

    texts = texts.apply(lambda x: re.sub("<[^>]+>", "", x.lower()).strip())
    texts = texts.apply(lambda x: re.sub(r"[^\w\s]", "", x))
    texts = texts.apply(lambda x: re.sub(r"um", "", x))
    texts = texts.apply(lambda x: re.sub(r"uh", "", x))
    texts = texts.apply(lambda x: re.sub(r"rry", "", x))
    texts = texts.apply(lambda x: re.sub(r"mm", "", x))
    texts = texts.apply(lambda x: re.sub(r"mmhmm", "", x))
    texts = texts.apply(lambda x: re.sub(r"well", "", x))
    texts = texts.apply(lambda x: re.sub(r"okay", "", x))
    texts = texts.apply(lambda x: re.sub(r"by the way", "", x))
    texts = texts.apply(lambda x: re.sub(r"yeah", "", x))
    texts = texts.apply(lambda x: re.sub(r"no", "", x))
    texts = texts.apply(lambda x: re.sub(r"so", "", x))
    texts = texts.apply(lambda x: re.sub(" +", " ", x.strip()))
    texts = texts.apply(lambda x: re.sub(r"\b(\w+\s*)\1{1,}", "\\1", x))

    spell = Speller(fast=True)
    texts = texts.apply(spell)

    texts_stemmed = texts.apply(
        lambda x: " ".join([stemmer.stem(y) for y in x.split()])
    )

    masked_indices = []
    for _, t in tqdm(enumerate(texts)):
        masked_indices.append(len(t.split()) >= 1)

    texts = texts[masked_indices].reset_index(drop=True)

    word_embedder = dict()
    word_to_word = dict()
    word_to_sent = dict()

    for i, sent in enumerate(texts_stemmed):
        words = sent.split()
        last_word = None
        for word in words:
            if word not in word_embedder:
                word_embedder[word] = len(word_embedder)

            if word_embedder[word] not in word_to_word:
                word_to_word[word_embedder[word]] = set()
            if last_word is not None:
                word_to_word[word_embedder[last_word]].add(word_embedder[word])

            if word_embedder[word] not in word_to_sent:
                word_to_sent[word_embedder[word]] = set()

            word_to_sent[word_embedder[word]].add(i)

            last_word = word

    w2w_indices = [[], []]
    w2s_indices = [[], []]

    for edge_from in word_to_word:
        for edge_to in word_to_word[edge_from]:
            w2w_indices[0].append(edge_from)
            w2w_indices[1].append(edge_to)

    for edge_from in word_to_sent:
        for edge_to in word_to_sent[edge_from]:
            w2s_indices[0].append(edge_from)
            w2s_indices[1].append(edge_to + len(word_embedder))

    w2w_indices = np.array(w2w_indices)
    w2s_indices = np.array(w2s_indices)

    np.save(arr=w2w_indices, file=f"out/features/hgat/{args.data}_w2w_indices.npy")
    np.save(arr=w2s_indices, file=f"out/features/hgat/{args.data}_w2s_indices.npy")
    np.save(arr=masked_indices, file=f"out/features/hgat/hgat_{args.data}_mask.npy")

    bert_model = SentenceTransformer("all-mpnet-base-v2")
    word_embeddings = bert_model.encode(
        list(word_embedder.keys()), show_progress_bar=True
    )

    sent_embeddings = bert_model.encode(texts, show_progress_bar=True)

    np.save(
        arr=np.array(word_embeddings),
        file="out/features/hgat/{args.data}_word_embeddings.npy",
    )
    np.save(
        arr=np.array(sent_embeddings),
        file=f"out/features/hgat/{args.data}_sent_embeddings.npy",
    )

    np.save(
        arr=np.array(s2s_indices), file=f"out/features/hgat/{args.data}_s2s_indices.npy"
    )
    np.save(arr=np.array(s2s_types), file=f"out/features/hgat/{args.data}_s2s_types.npy")

    df["TEXT"] = texts
    df = df.groupby(["ID"])["TEXT"].apply(list)

    sent_index = 0
    similar_sent = [[], []]

    for document in df:
        trigrams_set = dict()
        for i, sentence in enumerate(document):
            token = word_tokenize(sentence)
            token
            trigrams = list(ngrams(token, 3))
            for trigram in trigrams:
                if trigram not in trigrams_set:
                    trigrams_set[trigram] = set()
                trigrams_set[trigram].add(i)
        for key in trigrams_set:
            sim_sents = list(trigrams_set[key])
            if len(sim_sents) > 1:
                for i in range(len(sim_sents)):
                    for j in range(i + 1, len(sim_sents)):
                        similar_sent[0].append(sent_index + sim_sents[i])
                        similar_sent[1].append(sent_index + sim_sents[j])
                        similar_sent[0].append(sent_index + sim_sents[j])
                        similar_sent[1].append(sent_index + sim_sents[i])
        sent_index += len(document)

    np.save(
        arr=np.array(similar_sent), file=f"out/features/hgat/{args.data}_s2s_similar.npy"
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="extract hgat edges")
    parser.add_argument(
        "--data", type=str, help="train or test"
    )
    _args = parser.parse_args()
    main(_args)