#!/usr/bin/env python

import argparse
from crf import LinearChainCRF

if __name__ == '__main__':
    datafile = '../dataset/EN/train'
    modelfile = 'parameters'

    crf = LinearChainCRF()
    crf.train(datafile, modelfile)
