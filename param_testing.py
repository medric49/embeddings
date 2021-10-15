import os

from train import train

if __name__ == '__main__':
    root_dir = '1g-word-1m-benchmark-r13output/training-monolingual.tokenized.shuffled'
    corpus_root = 'corpus'

    i = 1
    new_corpus_file = f'{corpus_root}/{i}.txt'
    new_corpus = open(new_corpus_file, 'a')

    for j in range(1, i+1):
        num = str(j).zfill(5)
        corpus_file = f'{root_dir}/news.en-{num}-of-00100'
        with open(corpus_file, 'r') as corpus_file:
            corpus = corpus_file.read()
        new_corpus.write(corpus)
    new_corpus.close()

    for k in range(5, 20):
        training_time = train(new_corpus_file, i, dim=200, window=5, negative=k, log_parent_dir='param_tests')
        print(k, training_time)

    os.system(f'rm {new_corpus_file}')
