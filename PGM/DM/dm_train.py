import argparse
from dm import DM

if __name__ == '__main__':
    datafile = '../../dataset/EN/train_small'
    modelfile = 'parameters'
    squared_sigma = 10
    epoch = 30

    crf = DM.get_SSVM(datafile, modelfile, squared_sigma)
    crf.load_data()
    crf.train(prec=0.1)
