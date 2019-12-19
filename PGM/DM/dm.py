from .read_corpus import read_conll_corpus
from .feature import FeatureSet, STARTING_LABEL_INDEX

import numpy as np
import time
import json
import random
from collections import Counter
import os

from scipy.optimize import fmin_l_bfgs_b

def logsumexp(a):
    """
    Compute the log of the sum of exponentials of an array ``a``, :math:`\log(\exp(a_0) + \exp(a_1) + ...)`
    """
    b = a.max()
    return b + np.log((np.exp(a-b)).sum())

class DM:
    def __init__(self, corpus_filename, model_filename, squared_sigma, model_type='CRF', structual_label=None, epsilon=0):
        self.squared_sigma = squared_sigma
        # Read the training corpus
        self.corpus_filename = corpus_filename
        self.model_filename = model_filename
        assert model_type in ['CRF', 'MEMM', 'SSVM', 'SP'], "Unsupported model type"
        self.model_type = model_type
        print(f' ******** {self.model_type} *********')
        
        if model_type == 'SP':
            # "hard" max
            self.max_func = np.max
        else:
            # softmax
            self.max_func = logsumexp

        if model_type == 'SSVM':
            self.structual_label = structual_label
            self.epsilon = epsilon
        pass
    
    @classmethod
    def get_CRF(cls, corpus_filename, model_filename, squared_sigma):
        return cls(corpus_filename, model_filename, squared_sigma, 'CRF')


    @classmethod
    def get_MEMM(cls, corpus_filename, model_filename, squared_sigma):
        return cls(corpus_filename, model_filename, squared_sigma, 'MEMM')

    @classmethod
    def get_SP(cls, corpus_filename, model_filename, squared_sigma):
        return cls(corpus_filename, model_filename, squared_sigma, 'SP')

    @classmethod
    def get_SSVM(cls, corpus_filename, model_filename, squared_sigma):
        # add more loss to this label
        label = ['B-SBAR']
        return cls(corpus_filename, model_filename, squared_sigma, 'SSVM', structual_label=label, epsilon=0.1)



    def _read_corpus(self, filename):
        return read_conll_corpus(filename)

    def _get_training_feature_data(self):
        return [[self.feature_set.get_feature_list(X, t) for t in range(len(X))] for X, _ in self.training_data]

    def _easy_get_training_feature_data(self, easy_train):
        return [[self.feature_set.get_feature_list(X, t) for t in range(len(X))] for X, _ in easy_train]

    def _log_potential_table(self, X, inference=True):
        """
        Generates a potential table using given observations.
        * potential_table[t][prev_y, y]
            := (inner_product(params, feature_vector(prev_y, y, X, t)))
            (where 0 <= t < len(X))
        """
        num_labels = len(self.label_dic)
        tables = list()

        for t in range(len(X)):
            table = np.zeros((num_labels, num_labels))
            if inference:
                for (prev_y, y), score in self.feature_set.calc_inner_products(self.params, X, t):
                    if prev_y == -1:
                        table[:, y] += score
                    else:
                        table[prev_y, y] += score
            else:
                for (prev_y, y), feature_ids in X[t]:
                    score = sum(self.params[fid] for fid in feature_ids)
                    if prev_y == -1:
                        table[:, y] += score
                    else:
                        table[prev_y, y] += score
            
            # Make everything except the starting label 0
            if t == 0:
                table[STARTING_LABEL_INDEX+1:] = -np.inf # TODO is this correct?

            # Make transition to starting label and transition from the starting label 0
            else:
                table[:,STARTING_LABEL_INDEX] = -np.inf # TODO is this correct?
                table[STARTING_LABEL_INDEX,:] = -np.inf # TODO is this correct?
            tables.append(table)

        return tables


    def _forward_backward(self, time_length, potential_table):
        """
        Everything have taken the log
        """
        alpha = self._forward(time_length, potential_table)
        beta = self._backward(time_length, potential_table)
        Z = self.max_func(alpha[time_length-1])

        return alpha, beta, Z


    def _forward(self, time_length, potential_table):
        num_labels = len(self.label_dic)
        alpha = np.zeros((time_length, num_labels))
        t = 0
        for label_id in range(num_labels):
            alpha[t, label_id] = potential_table[t][STARTING_LABEL_INDEX, label_id]

        t = 1
        while t < time_length:
            label_id = 1
            while label_id < num_labels:
                alpha[t, label_id] = self.max_func(alpha[t-1,:] + potential_table[t][:,label_id])
                label_id += 1
            t += 1
        return alpha


    def _backward(self, time_length, potential_table):
        num_labels = len(self.label_dic)
        beta = np.zeros((time_length, num_labels))
        t = time_length - 1
        for label_id in range(num_labels):
            beta[t, label_id] = 0 

        for t in range(time_length-2, -1, -1):
            for label_id in range(1, num_labels):
                beta[t, label_id] = self.max_func(beta[t+1,:] + potential_table[t+1][label_id,:])
        
        return beta


    def _log_likelihood(self, parameters):
        """
        Calculate likelihood and gradient
        train_set: a number < len(dataset), if None, will use the entire training set
        """
        self.params = parameters
        train_set = None # add this line first
        if train_set == None: 
            train_data = self._get_training_feature_data()
            empirical_counts = self.feature_set.get_empirical_counts()
        else:
            easy_data = random.sample(self.training_data, train_set)
            train_data = self._easy_get_training_feature_data(easy_data)
            self.feature_set.easy_scan(easy_data)
            empirical_counts = self.feature_set.easy_get_empirical_counts()
            
        expected_counts = np.zeros(len(self.feature_set))

        total_logZ = 0
        for X_features in train_data:
            potential_table = self._log_potential_table(X_features, inference=False)
            alpha, beta, Z = self._forward_backward(len(X_features), potential_table)
            total_logZ += Z
            for t in range(len(X_features)):
                potential = potential_table[t]
                for (prev_y, y), feature_ids in X_features[t]:
                    prob = 0
                    if self.model_type == 'SSVM' and y in self.structual_label:
                        # the loss added can be tuned
                        prob += self.epsilon


                    # Adds p(prev_y, y | X, t)
                    if prev_y == -1:
                        if self.model_type == "MEMM": # For MEMM use local normalization
                            prob +=  (alpha[t, y]) - self.max_func(alpha[t, :])
                        else: # For other models using global normalization
                            prob +=  (alpha[t, y] + beta[t, y]) - Z                                   
                        prob = np.exp(prob).clip(0., 1.)

                    elif t == 0:
                        if prev_y is not STARTING_LABEL_INDEX:
                            continue
                        else:
                            if self.model_type == "MEMM":
                                prob += (potential[STARTING_LABEL_INDEX, y]) # TODO should i minus localnormalization? which one?
                            else: 
                                prob += (potential[STARTING_LABEL_INDEX, y] + beta[t, y]) - Z 
                            prob = np.exp(prob).clip(0., 1.)

                    else:
                        if prev_y is STARTING_LABEL_INDEX or y is STARTING_LABEL_INDEX:
                            continue
                        else:
                            if self.model_type == "MEMM":
                                prob += (alpha[t-1, prev_y] + potential[prev_y, y]) - self.max_func(alpha[t-1, :])
                            else:
                                prob += (alpha[t-1, prev_y] + potential[prev_y, y] + beta[t, y]) - Z
                            prob = np.exp(prob).clip(0., 1.)

                    for fid in feature_ids:
                        expected_counts[fid] += prob

        # TODO: make sure this is log likelihood
        log_likelihood = np.dot(empirical_counts, self.params) - total_logZ - np.sum(np.dot(self.params, self.params))/(self.squared_sigma*2)        
        gradients = empirical_counts - expected_counts - self.params/self.squared_sigma

        print('neg_log_likelihood: ', -log_likelihood)
        return -log_likelihood, -gradients


    def train(self, prec=0.1):
        # Estimates parameters to maximize log-likelihood of the corpus.
        start_time = time.time()
        print(' ******** Start Training *********')
        print('* Squared sigma:', self.squared_sigma)
        print('* Start Gradient Descend')
        print('   ========================')
        print('   iter(sit): Negative log-likelihood')
        print('   ------------------------')
        
        # for i in range(start_epoch, start_epoch+epoch):
        #     neg_log_likelihood, gradient = self._log_likelihood(train_set)
        #     print(f'   Iteration: {i}, Negative Log-likelihood: {neg_log_likelihood}')
        #     # The key: gradient clipping for more stable answer
        #     self.params -= lr * np.clip(gradient, -3, 3) / (i+1) ** decay
        #     
        #     if neg_log_likelihood <= self.neg_log_likelihood:
        #         self.neg_log_likelihood = neg_log_likelihood
        #         self.save_model(self.model_filename)

        self.params, self.log_likelihood, information = fmin_l_bfgs_b(self._log_likelihood, self.params, pgtol=prec)
        print('   ========================')
        print(' ******** Finished Training *********')

        self.save_model(self.model_filename)
        elapsed_time = time.time() - start_time
        print(f'* Elapsed time: {elapsed_time//60} mins')


    def test(self, test_corpus_filename, output_filename):
        if self.params is None:
            raise BaseException("You should load a model first!")

        test_data = self._read_corpus(test_corpus_filename)

        with open(output_filename, 'w') as output_file:
            total_count = 0
            correct_count = 0
            for X, Y in test_data:
                Yprime = self.inference(X)
                for xx, yy in zip(X, Yprime):
                    output_file.write(xx[0] + ' ' + yy + '\n')
                output_file.write('\n')
                for t in range(len(Y)):
                    total_count += 1
                    if Y[t] == Yprime[t]:
                        correct_count += 1
        
        print('* Test output has been saved at "%s/%s"' % (os.getcwd(), output_filename))

    def inference(self, X):
        """
        Finds the best label sequence.
        """
        potential_table = self._log_potential_table(X, inference=True)
        Yprime = self.viterbi(X, potential_table)
        return Yprime

    def viterbi(self, X, potential_table):
        time_length = len(X)
        max_table = np.zeros((time_length, self.num_labels))
        argmax_table = np.zeros((time_length, self.num_labels), dtype='int64')

        t = 0
        for label_id in range(self.num_labels):
            max_table[t, label_id] = potential_table[t][STARTING_LABEL_INDEX, label_id]
        for t in range(1, time_length):
            for label_id in range(1, self.num_labels):
                max_value = -np.inf
                max_label_id = None
                for prev_label_id in range(1, self.num_labels):
                    value = max_table[t-1, prev_label_id] + potential_table[t][prev_label_id, label_id] # Not sure whether this is correct
                    if value > max_value:
                        max_value = value
                        max_label_id = prev_label_id
                max_table[t, label_id] = max_value
                argmax_table[t, label_id] = max_label_id

        sequence = list()
        next_label = max_table[time_length-1].argmax()
        sequence.append(next_label)
        for t in range(time_length-1, -1, -1):
            next_label = argmax_table[t, next_label]
            sequence.append(next_label)
        return [self.label_dic[label_id] for label_id in sequence[::-1][1:]]


    def load_data(self):
        self.training_data = self._read_corpus(self.corpus_filename)
        # Generate feature set from the corpus
        self.feature_set = FeatureSet()
        self.feature_set.scan(self.training_data)
        self.label_dic, self.label_array = self.feature_set.get_labels()
        if self.model_type == 'SSVM': self.structual_label = [self.label_dic[o] for o in self.structual_label]
        self.num_labels = len(self.label_array)
        print("* Number of labels: %d" % (self.num_labels-1))
        print("* Number of features: %d" % len(self.feature_set))
        print("Reinitialized model parameters")
        self._initialize_parameters()
        self.neg_log_likelihood = np.inf

    def _initialize_parameters(self):
        # zero initialization is better than random initialization
        self.params = np.zeros(len(self.feature_set))
        print("* Initialized weight of size: %d" % len(self.feature_set))
        
    def save_model(self, model_filename):
        model = {
                 "feature_dic": self.feature_set.serialize_feature_dic(),
                 "num_features": self.feature_set.num_features,
                 "labels": self.feature_set.label_array,
                 "params": list(self.params),
                 "neg_log_likelihood": self.neg_log_likelihood
                }
        with open(model_filename, 'w') as f: json.dump(model, f, ensure_ascii=False, indent=2, separators=(',', ':'))
        print('* Trained CRF Model has been saved at "%s/%s"' % (os.getcwd(), model_filename))

    def load_model(self, model_filename):
        f = open(model_filename)
        model = json.load(f)
        f.close()

        self.feature_set = FeatureSet()
        self.feature_set.load(model['feature_dic'], model['num_features'], model['labels'])
        self.label_dic, self.label_array = self.feature_set.get_labels()
        self.num_labels = len(self.label_array)
        self.params = np.asarray(model['params'])
        if 'neg_log_likelihood' in model:
            self.neg_log_likelihood = model['neg_log_likelihood']
        else:
            self.neg_log_likelihood = np.inf

        print('CRF model loaded')


# For testing
if __name__ == "__main__":
    ## for training: crf_train.py
    ## for testing: crf_testing.py
    pass
