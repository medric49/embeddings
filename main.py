from sys import stdin
import sys
import argparse

from train import train

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('corpus', type=str, help='Glossary file.')
    parser.add_argument('--log', type=str, help='Log directory of the training', default=None)
    parser.add_argument('--id', type=int, help='Group id', default=0)
    parser.add_argument('--dim', type=int, help='Dimension of embeddings', default=100)
    parser.add_argument('--window', type=int, help='Context window size', default=5)
    parser.add_argument('--negative', type=int, help='Number of negative inputs', default=5)
    parser.add_argument('--workers', type=int, help='Number of workers', default=8)
    args, _ = parser.parse_known_args(sys.argv[1:])

    train(args.corpus, args.id, args.dim, args.window, args.negative, args.workers)
