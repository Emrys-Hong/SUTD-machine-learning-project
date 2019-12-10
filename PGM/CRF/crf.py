from read_corpus import read_conll_corpus
from feature import FeatureSet, STARTING_LABEL_INDEX

from numpy import exp, log
import numpy as np
import time
import json
from collections import Counter
from scipy.optimize import fmin_l_bfgs_b
SCALING_THRESHOLD = 1e250


class LinearChainCRF:
    """
    Linear-chain Conditional Random Field
    """
    def __init__(self, corpus_filename, model_filename, squared_sigma):
        self.squared_sigma = squared_sigma
        # Read the training corpus
        self.corpus_filename = corpus_filename
        self.model_filename = model_filename
        pass

    def _read_corpus(self, filename):
        return read_conll_corpus(filename)

    def _get_training_feature_data(self):
        return [[self.feature_set.get_feature_list(X, t) for t in range(len(X))] for X, _ in self.training_data]


    def _generate_potential_table(self, X, inference=True):
        """
        Generates a potential table using given observations.
        * potential_table[t][prev_y, y]
            := exp(inner_product(params, feature_vector(prev_y, y, X, t)))
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
            table = np.exp(table)
            
            # Make everything except the starting label 0
            if t == 0:
                table[STARTING_LABEL_INDEX+1:] = 0

            # Make transition to starting label and transition from the starting label 0
            else:
                table[:,STARTING_LABEL_INDEX] = 0
                table[STARTING_LABEL_INDEX,:] = 0
            tables.append(table)

        return tables


    def _forward_backward(self, time_length, potential_table):
        num_labels = len(self.label_dic)
        alpha = np.zeros((time_length, num_labels))
        scaling_dic = dict()
        t = 0
        for label_id in range(num_labels):
            alpha[t, label_id] = potential_table[t][STARTING_LABEL_INDEX, label_id]
        #alpha[0, :] = potential_table[0][STARTING_LABEL_INDEX, :]  # slow
        t = 1
        while t < time_length:
            scaling_time = None
            scaling_coefficient = None
            overflow_occured = False
            label_id = 1
            while label_id < num_labels:
                alpha[t, label_id] = np.dot(alpha[t-1,:], potential_table[t][:,label_id])
                if alpha[t, label_id] > SCALING_THRESHOLD:
                    if overflow_occured:
                        print('******** Consecutive overflow ********')
                        raise BaseException()
                    overflow_occured = True
                    scaling_time = t - 1
                    scaling_coefficient = SCALING_THRESHOLD
                    scaling_dic[scaling_time] = scaling_coefficient
                    break
                else:
                    label_id += 1
            if overflow_occured:
                alpha[t-1] /= scaling_coefficient
                alpha[t] = 0
            else:
                t += 1

        beta = np.zeros((time_length, num_labels))
        t = time_length - 1
        for label_id in range(num_labels):
            beta[t, label_id] = 1.0
        #beta[time_length - 1, :] = 1.0     # slow
        for t in range(time_length-2, -1, -1):
            for label_id in range(1, num_labels):
                beta[t, label_id] = np.dot(beta[t+1,:], potential_table[t+1][label_id,:])
            if t in scaling_dic.keys():
                beta[t] /= scaling_dic[t]

        Z = sum(alpha[time_length-1])

        return alpha, beta, Z, scaling_dic
    

    def _log_likelihood(self, params):
        """
        Calculate likelihood and gradient
        """
        # previous iteration
        self.params = params

        empirical_counts = self.feature_set.get_empirical_counts()
        expected_counts = np.zeros(len(self.feature_set))

        total_logZ = 0
        training_data = self._get_training_feature_data()
        for X_features in training_data:
            potential_table = self._generate_potential_table(X_features, inference=False)
            alpha, beta, Z, scaling_dic = self._forward_backward(len(X_features), potential_table)
            total_logZ += log(Z) + sum(log(scaling_coefficient) for _, scaling_coefficient in scaling_dic.items())
            for t in range(len(X_features)):
                potential = potential_table[t]
                for (prev_y, y), feature_ids in X_features[t]:
                    # Adds p(prev_y, y | X, t)
                    if prev_y == -1:
                        if t in scaling_dic.keys():
                            prob = (alpha[t, y] * beta[t, y] * scaling_dic[t])/Z
                        else:
                            prob = (alpha[t, y] * beta[t, y])/Z
                    elif t == 0:
                        if prev_y is not STARTING_LABEL_INDEX:
                            continue
                        else:
                            prob = (potential[STARTING_LABEL_INDEX, y] * beta[t, y])/Z
                    else:
                        if prev_y is STARTING_LABEL_INDEX or y is STARTING_LABEL_INDEX:
                            continue
                        else:
                            prob = (alpha[t-1, prev_y] * potential[prev_y, y] * beta[t, y]) / Z
                    for fid in feature_ids:
                        expected_counts[fid] += prob

        likelihood = np.dot(empirical_counts, self.params) - total_logZ - np.sum(np.dot(self.params, self.params))/(self.squared_sigma*2)
        gradients = empirical_counts - expected_counts - self.params/self.squared_sigma
        self.nll = likelihood * -1
        print(f'   Log Likelihood: {self.nll}')
        return self.nll, -gradients

    def train(self, epoch=50):
        self.params = np.random.randn(len(self.feature_set))
        self.params = np.zeros(len(self.feature_set))
        # Estimates parameters to maximize log-likelihood of the corpus.
        start_time = time.time()
        print(' ******** Start Training *********')
        print('* Squared sigma:', self.squared_sigma)
        print('* Start L-BGFS')
        print('   ========================')
        print('   iter(sit): likelihood')
        print('   ------------------------')
        
        #for _ in range(epoch):
        #    log_likelihood, gradient = self._log_likelihood()
        #    print(f'   Log Likelihood: {log_likelihood}')
        #    self.params -= gradient
        self.params, self.nll, information = fmin_l_bfgs_b(func=self._log_likelihood, x0=self.params, pgtol=0.01)
        print('   ========================')
        print('* Likelihood: %s' % str(self.nll))
        print(' ******** Finished Training *********')

        self.save_model(self.model_filename)
        elapsed_time = time.time() - start_time
        print('* Elapsed time: %f' % elapsed_time)

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
        
        import os
        print('* Trained CRF Model has been saved at "%s/%s"' % (os.getcwd(), output_filename))

    def inference(self, X):
        """
        Finds the best label sequence.
        """
        potential_table = self._generate_potential_table(X, inference=True)
        Yprime = self.viterbi(X, potential_table)
        return Yprime

    def viterbi(self, X, potential_table):
        """
        The Viterbi algorithm with backpointers
        """
        time_length = len(X)
        max_table = np.zeros((time_length, self.num_labels))
        argmax_table = np.zeros((time_length, self.num_labels), dtype='int64')

        t = 0
        for label_id in range(self.num_labels):
            max_table[t, label_id] = potential_table[t][STARTING_LABEL_INDEX, label_id]
        for t in range(1, time_length):
            for label_id in range(1, self.num_labels):
                max_value = -float('inf')
                max_label_id = None
                for prev_label_id in range(1, self.num_labels):
                    value = max_table[t-1, prev_label_id] * potential_table[t][prev_label_id, label_id]
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
        self.num_labels = len(self.label_array)
        print("* Number of labels: %d" % (self.num_labels-1))
        print("* Number of features: %d" % len(self.feature_set))
        
    def save_model(self, model_filename):
        model = {"feature_dic": self.feature_set.serialize_feature_dic(),
                 "num_features": self.feature_set.num_features,
                 "labels": self.feature_set.label_array,
                 "params": list(self.params)}
        with open(model_filename, 'w') as f: json.dump(model, f, ensure_ascii=False, indent=2, separators=(',', ':'))
        import os
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

        print('CRF model loaded')


# For testing
if __name__ == "__main__":
    ## for training: crf_train.py
    ## for testing: crf_testing.py
    pass