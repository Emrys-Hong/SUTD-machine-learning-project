from crf import CRF
from read_data import read_data

file_path = 'data/small_train.data'
data = read_data(file_path)

learing_rate = 5e-2
num_iters = 100
square_sigma = 1000
save_modelname = 'test_model'

crf = CRF(data)
crf.train(num_iters, learing_rate, square_sigma)
crf.save_model(save_modelname)
