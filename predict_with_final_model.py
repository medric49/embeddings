import os

import numpy as np

from train import train, eval, load_model
from gensim.models import Word2Vec


def proximity(v1, v2):
    return 1 / (1 + np.exp(-np.dot(v1, v2)))


if __name__ == '__main__':
    corpus_file = 'corpus/test_dataset.txt'
    model = load_model(id=9, dim=275, window=9, negative=17, log_parent_dir='logs')

    with open('liste_mots_devoir4.txt', 'r') as word_file:
        words = word_file.read()
    words = words.split('\n')
    words.remove('')

    for word in words:
        if word in model.wv:
            most_similar = model.wv.most_similar(word, topn=10)
            most_similar = [f'{w} [{p}]' for w, p in most_similar]
            most_similar = ' '.join(most_similar)

            print(f'{word}\t{most_similar}')


