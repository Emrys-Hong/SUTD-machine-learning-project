#!/usr/bin/env python

import argparse
from crf import LinearChainCRF

if __name__ == '__main__':
    datafile = '../dataset/EN/train_small'
    modelfile = 'parameters'
    test_corpus_filename = '../dataset/EN/dev.out'
    output_filename = 'crf_output'
    regularization = 10

    crf = LinearChainCRF(datafile, modelfile, regularization)
    crf.load_model(modelfile)
    crf.test(test_corpus_filename, output_filename)

