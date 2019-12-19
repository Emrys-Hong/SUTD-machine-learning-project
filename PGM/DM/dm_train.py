import argparse
from dm import DM

if __name__ == '__main__':
    datafile = '../../dataset/EN/train_small'
    modelfile = 'parameters'
    squared_sigma = 10
    epoch = 30

    crf = DM.get_CRF(datafile, modelfile, squared_sigma)
    crf.load_data()
    crf.train(epoch=100, lr=0.002, decay=0.501, start_epoch=0, train_set=1)
