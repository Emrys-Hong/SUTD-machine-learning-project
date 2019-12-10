from hidden_markov_model import *

if __name__ == "__main__":
    DATA_FOLDER = Path('../../../dataset/')
    AL = DATA_FOLDER/'AL'
    AL_train = AL/'train'
    AL_dev_x = AL/'dev.in'
    AL_dev_y = AL/'dev.out'
    AL_out_4 = AL/'dev.p4.out'
    hmm = HMM(AL_train)
    hmm.train()
    hmm.predict_top_k(AL_dev_y, AL_out_4, k=7)
    print("success")
