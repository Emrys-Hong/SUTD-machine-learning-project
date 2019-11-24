import numpy as np 
import json
from read_data import read_data
from collections import Counter
from feature import Feature, START_LABEL_IDX


class CRF():

    def __init__(self, data=None):
        self.Feature = Feature(data)
        self.Feature._scan()
        self.num_features = self.Feature.num_features
        self.params = np.zeros(self.num_features)
        self.num_labels = len(self.Feature.label_array)
        

    def potential_table(self, sentence, inference = True):
        tables = list()
        for pos in range(len(sentence)):
            table = np.zeros((self.num_labels, self.num_labels))
            if inference:
                pass
            else:
                feature_func_ids_at_pos = self.Feature.get_feature_func_id(sentence, pos)
                for (prev_lbl_idx, cur_lbl_idx), feature_ids in feature_func_ids_at_pos:
                    score = sum([self.params[fid] for fid in feature_ids])
                    if prev_lbl_idx == -1:
                        table[:, cur_lbl_idx] += score
                    else:
                        table[prev_lbl_idx, cur_lbl_idx] += score
            table = np.exp(table)
            if pos == 0:
                table[START_LABEL_IDX+1:] = 0
            else:
                table[START_LABEL_IDX,:] = 0
                table[:,START_LABEL_IDX] = 0
            tables.append(table)

        return tables
    
    def forward_backward(self, sentence, inference):
        sentence_length = len(sentence)
        potential_tables = self.potential_table(sentence, inference=inference)

        alpha = np.zeros((sentence_length, self.num_labels))
        pos = 0
        for label_id in range(self.num_labels):
            alpha[pos, label_id] = potential_tables[pos][START_LABEL_IDX, label_id]
        pos = 1
        while pos < sentence_length:
            label_id = 1
            while label_id < self.num_labels:
                alpha[pos, label_id] = alpha[pos-1,:].dot(potential_tables[pos][:,label_id])
                label_id += 1
            pos+=1
        
        beta = np.zeros((sentence_length, self.num_labels))
        pos = sentence_length - 1
        for label_id in range(self.num_labels):
            beta[pos,label_id] = 1.0
        
        for pos in range(sentence_length-2, -1, -1):
            for label_id in range(1, self.num_labels):
                beta[pos, label_id] = beta[pos+1,:].dot(potential_tables[pos+1][label_id,:])

        Z = sum(alpha[sentence_length-1])

        return alpha, beta, Z, potential_tables
    
    def log_likelihood(self, square_sigma, inference):
        empirical_counts = self.Feature.get_empirical_counts()
        total_logZ = 0
        expected_counts = np.zeros(self.num_features)

        for sentence in self.Feature.data:
            alpha, beta, Z, potential_tables = self.forward_backward(sentence, inference)
            total_logZ += np.log(Z)
            for pos in range(len(sentence)):
                potential = potential_tables[pos]
                for (prev_lbl_idx, cur_lbl_idx), feature_ids in self.Feature.get_feature_func_id(sentence, pos):
                    if prev_lbl_idx == -1:
                        prob = (alpha[pos, cur_lbl_idx] * beta[pos, cur_lbl_idx]) / Z
                    elif pos == 0:
                        if prev_lbl_idx is not START_LABEL_IDX:
                            continue
                        else:
                            prob = (potential[START_LABEL_IDX, cur_lbl_idx] * beta[pos][cur_lbl_idx]) / Z
                    else:
                        if prev_lbl_idx is START_LABEL_IDX or cur_lbl_idx is START_LABEL_IDX:
                            continue
                        else:
                            prob = (alpha[pos-1, prev_lbl_idx] * potential[prev_lbl_idx, cur_lbl_idx] * beta[pos, cur_lbl_idx]) / Z
                    for fid in feature_ids:
                        expected_counts[fid] += prob
        likelihood = np.dot(empirical_counts, self.params) - total_logZ - np.sum(np.dot(self.params, self.params))/(square_sigma*2)
        gradients = empirical_counts - expected_counts - self.params/square_sigma

        print('likelihood:', likelihood * (-1))
        return likelihood, gradients

    def train(self, num_iters, learning_rate, square_sigma):
        
        for n in range(num_iters):
            log_likelihood, gradients = self.log_likelihood(square_sigma, False)
            self.params += learning_rate * gradients
        
    
    def save_model(self, model_filename):
        model = {
            'params':list(self.params),
            'num_features':self.num_features,
            'num_labels':self.num_labels,
            'Feature':self.Feature.serialize()
        }
        with open(model_filename, 'w') as f:
            json.dump(model, f)

    def load_model(self, model_filename):
        with open(model_filename, 'r') as f:
            model = json.load(f)
            self.Feature = Feature()
            self.Feature.load(model['Feature'])
            self.params = np.array(model['params'])
            self.num_features = model['num_features']
            self.num_labels = model['num_labels']
            