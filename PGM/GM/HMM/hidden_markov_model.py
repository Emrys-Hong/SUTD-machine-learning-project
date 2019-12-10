import numpy as np
import os
from pathlib import Path
from collections import Counter, defaultdict

import warnings
warnings.filterwarnings('ignore')

def get_train_data(filename):
    with open(filename) as f:
        lines = f.readlines()
    x, y = [], []
    temp_x, temp_y = [], []
    for l in lines:
        if len(l) == 1:
            assert(len(temp_x) == len(temp_y))
            x.append(temp_x)
            y.append(temp_y)
            temp_x, temp_y = [], []
            continue
        xx, yy = l.split()
        temp_x.append(xx)
        temp_y.append(yy)
    if len(temp_x) != 0:
        x.append(temp_x)
        y.append(temp_y)
    assert(len(x) == len(y))
    
    return x, y

def get_index_index(argmax, index, k):
    count = 0
    for o in argmax:
        if o == index:
            return count
        if o//k == index//k:
            count += 1

def get_test_data(filename, word2index):
    """Return:
                x: nested list of string
                x_int: nested list of integer"""
    with open(filename) as f:
        lines = f.readlines()
    x = []
    temp_x = []
    for l in lines:
        if len(l.strip()) == 0:
            x.append(temp_x)
            temp_x = []
            continue
        xx = l.split()
        temp_x.append(xx[0])
    if len(temp_x) != 0:
        x.append(temp_x)
    x_int = [[word2index[oo] for oo in o] for o in x ]
    return x, x_int


class HMM:
    def __init__(self, train_file):
        # read data
        self.words, self.labels = get_train_data(train_file)
        # create vocab
        self.tags = list(set([oo for o in self.labels for oo in o])) + ['SOS', 'EOS']
        self.tag2index = {o:i for i,o in enumerate(self.tags)}
        vocab_count = Counter([oo for o in self.words for oo in o])
        self.vocab = [o for o, v in dict(vocab_count).items() if v>=3] + ['#UNK#']
        self.word2index = defaultdict(int)
        for i,o in enumerate(self.vocab): self.word2index[o] = i+1
        # text to int
        self.x = [[self.word2index[oo] for oo in o] for o in self.words]
        self.y = [[self.tag2index[oo] for oo in o] for o in self.labels]

    def train_emission(self):
        emission = np.zeros((len(self.vocab), len(self.tags)))
        flat_y = [oo for o in self.y for oo in o]
        flat_x = [oo for o in self.x for oo in o]
        for xx, yy in zip(flat_x,flat_y):
            emission[xx, yy] += 1
        
        y_count = np.zeros(len(self.tags))
        for yy in flat_y:
            y_count[yy] += 1
        emission = emission/ y_count[None, :]
        np.nan_to_num(emission, 0)
        # emission_matrix += 1e-5 # Adding this smoothing will increase performance
        self.emission = emission

    def train_transition(self):
        """
        tags: included 'SOS' and 'EOS'
        transition from: (v1, v2, ..., 'SOS')
        transition to:   (v1, v2, ..., 'EOS')
        rows are transition_from
        cols are transition_to
        """
        transition = np.zeros((len(self.tags)-1, len(self.tags)-1))
        for yy in self.y: 
            transition[-1, yy[0]] += 1 # START transition
            for i in range(len(yy)-1): # tags transition from position 0 to len(yy)-2
                transition[yy[i], yy[i+1]] += 1
            transition[yy[-1], -1] += 1 # STOP transition
        transition = transition/np.sum(transition, axis=1)
        self.transition = transition
    
    def train(self):
        self.train_emission()
        self.train_transition()

    def _viterbi_top_k(self, x, k=7):
        """transition_matrix: before log
        x: [1, 2, 4, 19, ...]
        transition: after log
        time complexity: O(knt^2)
        return: 
                path: (len(x), )
                log(max_score)
        """
        score = np.zeros( (len(x)+2, len(self.tags)-2, k) ) 
        argmax = np.zeros( (len(x)+2, len(self.tags)-2, k), dtype=np.int) 
        transition, emission = np.log(self.transition), np.log(self.emission)
        # initialization at j=1
        score[1, :, 0] = (transition[-1, :-1] + emission[x[0], :-2])
        score[1, :, 1:] = -np.inf
        # j=2, ..., n
        for j in range(2, len(x)+1): 
            for t in range(len(self.tags)-2):
                pi = score[j-1, :]  # (num_of_tags-2, 7)
                a = transition[:-1, t] # (num_of_tags-2,)
                b = emission[x[j-1], t] # (1,)
                previous_all_scores = ((pi + a[:, None])).flatten()
                topk = previous_all_scores.argsort()[-k:][::-1] # big to small
                argmax[j, t] = topk # big: 0, small: -1 # topk // k is the real argument
                score[j, t] = previous_all_scores[topk] + b

        # j=n+1 step
        pi = score[len(x)] # (num_of_tags-2, 7)
        a = transition[:-1, -1]
        argmax_stop = (pi + a[:,None]).flatten().argsort()[-k:][::-1] # big to small
        log_likelihood = np.min(pi+a[:,None])
        argmax = argmax[2:-1] # (len(x)-1, num_of_tags-2, 7)
        
        # decoding

        ## Initialize the last pointer backward
        temp_index = argmax_stop[-1] # range from 0 ~ k * ( len(tags) - 1 )
        temp_index_index = get_index_index(argmax_stop, temp_index, k) # range from 0 ~ k-1, next index in next tag
        temp_arg = temp_index // k # range from 0 ~ k-1, next tag index
        
        path = [argmax_stop[-1]//k] # initialize path as an array

        for i in range(len(argmax)-1, -1, -1):
            temp_index = argmax[i, temp_arg, temp_index_index]
            temp_index_index = get_index_index(argmax[i, temp_arg], temp_index, k)
            temp_arg = temp_index // k
            path.append(temp_arg)
        return path[::-1], log_likelihood 

    def predict_top_k(self, dev_x_filename, output_filename, k=7):
        assert hasattr(self, 'transition') and hasattr(self, 'emission'), "run self.train() first"
        with open(output_filename, 'w') as f:
            words, dev_x = get_test_data(dev_x_filename, self.word2index)
            score_list = []
            for i, (ws,o) in enumerate(zip(words, dev_x)):
                path, log_max_score = self._viterbi_top_k(o, k)
                score_list.append(log_max_score)
                for w, p in zip(ws, path):
                    f.write(w + ' ' + self.tags[p] + '\n')
                f.write('\n')
        return

    def _navie_decoding(self, x):
        """emission matrix: (vocab_size, tag_size)
        x: converted to integer arrays"""
        return self.emission[x].argmax(axis=1)

    def navie_predict(self, dev_x_filename, output_filename):
        assert hasattr(self, 'emission'), "run self.train_emission() First!"
        with open(output_filename, 'w') as f:
            words, dev_x = get_test_data(dev_x_filename, self.word2index)
            for i, (ws,o) in enumerate(zip(words, dev_x)):
                path = self._navie_decoding(o)
                for w, p in zip(ws, path):
                    f.write(w + ' ' + self.tags[p] + '\n')
                f.write('\n')
        return

