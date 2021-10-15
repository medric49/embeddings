import os
import random
import sys
import time

import numpy as np
from gensim import models


def train(corpus, id, dim=100, window=5, negative=5, workers=8, log_dir=None, log_parent_dir='logs'):

    if log_dir is None:
        log_dir = f'{log_parent_dir}/model-{id}-{dim}-{window}-{negative}'

    os.makedirs(log_dir, exist_ok=True)
    log_file = f'{log_dir}/log.txt'
    model_file = f'{log_dir}/model.pth'

    tic = time.time()
    model = models.Word2Vec(corpus_file=corpus, vector_size=dim, window=window, workers=workers,
                            negative=negative)
    tac = time.time() - tic

    with open(log_file, 'w') as log_file:
        log_file.writelines([
            f'time\t{tac}\n',
        ])
    model.save(model_file)

    return tac


def sigmoid(z):
    return 1/(1 + np.exp(-z, ))


def load_model(id, dim=100, window=5, negative=5, workers=8, log_dir=None, log_parent_dir='logs'):
    if log_dir is None:
        log_dir = f'{log_parent_dir}/model-{id}-{dim}-{window}-{negative}'

    model_file = f'{log_dir}/model.pth'
    return models.KeyedVectors.load(model_file)


def eval_seq(model, sequence, window):
    if len(sequence) < window:
        return None
    start = random.randint(0, len(sequence) - window)
    context = sequence[start: start + window]

    w1, w2 = random.sample(context, 2)

    if w1 not in model.wv or w2 not in model.wv:
        return None

    v1 = model.wv[w1]
    v2 = model.wv[w2]

    value = sigmoid(np.dot(v1, v2))

    return value


def eval_seqs(model, sequences, window):
    value = []
    for seq in sequences:
        v = eval_seq(model, seq, window)
        if v is not None:
            value.append(v)

    return np.mean(value)


def eval(corpus_file, id, dim=100, window=5, negative=5, workers=8, log_dir=None, log_parent_dir='logs'):
    if log_dir is None:
        log_dir = f'{log_parent_dir}/model-{id}-{dim}-{window}-{negative}'

    if not os.path.isdir(log_dir):
        return None

    model_file = f'{log_dir}/model.pth'
    model = models.KeyedVectors.load(model_file)

    with open(corpus_file, 'r') as corpus_file:
        corpus = corpus_file.read()

    corpus = corpus.split('\n')
    corpus.remove('')

    return eval_seqs(model, corpus, 5)

