#!/usr/bin/env python

import argparse
from crf import LinearChainCRF

if __name__ == '__main__':
    datafile = '../../dataset/EN/train_middle'
    modelfile = 'parameters'
    regularization = 10
    epoch = 30

    crf = LinearChainCRF(datafile, modelfile, regularization)
    crf.load_data()
    crf.train(epoch)
