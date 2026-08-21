# Extractive Summarisation of Meeting Transcripts

Labelling each utterance in a meeting transcript as keep or drop, i.e. extractive
summarisation as binary sentence classification. Course project for INF554, Institut
Polytechnique de Paris.

## Problem

Given an automatically transcribed meeting, predict for every utterance whether it belongs
in the summary. The output is one binary label per utterance, so this is sentence
classification, not text generation.

## Data

Each meeting has two files: a JSON transcript of ASR utterances with speaker tags, and a
`.txt` discourse graph giving each utterance a typed relation to nearby utterances (16
relation types, listed in `tools/parse_adj.py`). The text is noisy: misspellings, incomplete
sentences, and artefact tokens like `<vocalsound>` and `<disfmarker>`.

The labels are imbalanced: 59,331 negative vs 13,292 positive over 72,623 training
utterances. We report F1 because accuracy is not useful at that ratio (always predicting
drop already gives 82%). We handle the imbalance by downsampling negatives to 25,000 for the
LSTM, and with a class-balanced cross-entropy loss for the HGAT.

## Approach

### Preprocessing (dropped)

We wrote a cleaning pipeline: lowercasing, contraction expansion, removing ASR artefact
tokens, stopword removal, onomatopoeia removal, and a minimum word length filter. We then
measured each step instead of assuming it helped.

**Every step except contraction expansion made F1 worse, so we dropped the pipeline.** That
is why `tools/clean_csv.py` is committed with its body commented out. On the ASR text,
cleaning removed more signal than noise.

### Models

Word and sentence embeddings come from frozen pretrained encoders (`all-mpnet-base-v2` in
`tools/crazy_edges.py`, `bert-base-uncased` in `scripts/preprocessing.py`). Nothing is
fine-tuned.

- **Feed-forward** (`models/nn.py`, `models/nn_ext.py`) — Linear/LayerNorm/ReLU over
  sentence embeddings.
- **LSTM** (`models/lstm.py`, trained in `notebooks/LSTM.ipynb`) — own 3k-word vocabulary
  and embedding table, single-layer LSTM.
- **GAT** (`models/gat.py`) — `GATv2Conv` layers over the discourse graph.
- **HGAT** (`models/hgat.py`) — adapted from Jia et al., EMNLP 2020, using PyTorch and
  PyTorch Geometric, 2,708,642 parameters, LayerNorm and dropout on each layer. `GATv2Conv`
  layers run over a word co-occurrence graph, then word-to-sentence edges, then
  sentence-to-sentence edges from the discourse graph (17-dim edge attribute: 16 relations
  plus one for consecutive utterances), then redundancy edges between sentences sharing a
  trigram, then a feed-forward classifier.

## Results

| Model | Hidden channels | F1 |
|---|---|---|
| LSTM | 100 | **0.596** |
| HGAT | 128 | 0.533 |
| HGAT | 32 | 0.510 |

The LSTM beat the graph model, even though the HGAT is much bigger and uses the discourse
structure.

## Analysis

The main reason seems to be that the discourse graph gives very little structure. The mean
out-degree is 0.999 (72,526 edges for 72,623 utterances) and only 73% of utterances have any
outgoing edge, so the graph is close to a chain. With neighbourhoods that small there is not
much for graph attention to use that a sequence model does not already capture.

Two things we did not get to:

- No neighbourhood-aware sampler, so HGAT training is full-batch over the whole graph.
- Speaker identity was never used as a feature, even though speaker tags are in the data.

## How to run

Python 3.10, `pip install -r requirements.txt`.

```
bash prep_data.sh

python -m scripts.hgat_train --model_config configs/hgat.yaml \
    --train_dir out/features/hgat --train_csv data/training/train.csv \
    --label_column LABEL --device cuda:0 --ckpt_path ckpt/hgat.pt

python -m scripts.hgat_test --model_config configs/hgat.yaml \
    --test_dir out/features/hgat --test_csv data/test/test.csv \
    --device cuda:0 --ckpt_path ckpt/hgat.pt --out_path out/hgat.csv
```

The feed-forward model is `scripts/train_predict.py` with `configs/nn.yaml`. The LSTM is run
from `notebooks/LSTM.ipynb`.

Note: the scripts expect the data in `data/training/` and `data/test/` while it is currently
in `data/raw/`, and they write to `out/` and `ckpt/`, which you need to create first.

## Repo layout

```
configs/       YAML hyperparameters for the NN and HGAT
data/raw/      Transcripts (.json), discourse graphs (.txt), training labels
models/        Model definitions: nn, nn_ext, lstm, gat, hgat, and older GCN/RGCN
notebooks/     LSTM.ipynb (best result), draft.ipynb (exploration)
scripts/       Training and prediction entry points, plus the provided baselines
tools/         Feature extraction, graph construction, dataset and output helpers
prep_data.sh   Runs the three feature-building steps in order
```
