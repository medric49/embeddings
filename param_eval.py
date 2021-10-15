import os

from train import train, eval

if __name__ == '__main__':
    corpus_file = 'corpus/test_dataset.txt'

    for k in range(1, 100):
        training_time = eval(corpus_file, id=k, dim=100, window=5, negative=5, log_parent_dir='logs')
        if training_time is None:
            continue
        print(k, training_time)

