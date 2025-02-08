import torch
import numpy as np
from os import listdir
from os.path import isfile, join


def load_data(features_path, device):
    all_files = [
        f for f in listdir(features_path) if isfile(join(features_path, f))
    ]

    word_embed = None
    sent_embed = None
    w2w_indices = None
    w2s_indices = None
    s2s_indices = None
    s2s_types = None
    s2s_sim = None

    for file in all_files:
        data = torch.tensor(np.load(join(features_path, file))).to(device)
        if file.endswith("word_embeddings.npy"):
            word_embed = data
        if file.endswith("sent_embeddings.npy"):
            sent_embed = data
        if file.endswith("w2w_indices.npy"):
            w2w_indices = data
        if file.endswith("w2s_indices.npy"):
            w2s_indices = data
        if file.endswith("s2s_indices.npy"):
            s2s_indices = data
        if file.endswith("s2s_types.npy"):
            s2s_types = data
        if file.endswith("s2s_similar.npy"):
            s2s_sim = data

    return (
        word_embed,
        sent_embed,
        w2w_indices,
        w2s_indices,
        s2s_indices,
        s2s_types,
        s2s_sim,
    )
