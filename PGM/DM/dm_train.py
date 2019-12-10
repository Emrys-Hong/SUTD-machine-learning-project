import argparse
from dm import DM

if __name__ == '__main__':
    datafile = '../../dataset/EN/train_middle'
    modelfile = 'parameters'
    regularization = 10
    epoch = 15

    crf = DM.get_CRF(datafile, modelfile, regularization)
    crf.load_data()
    crf.train(epoch)
