import numpy as np
from collections import Counter
from collections import defaultdict

from read_data import read_data

START_LABEL = '@@'
START_LABEL_IDX = 0

class Feature():
    
    def __init__(self, data=None):
        self.feature_dict = dict()                                  # feature function dictionary
        self.num_features = 0                                       # num of feature function
        self.empirical_counts = Counter()                           # counts of each feature function
        self.label_dict = {START_LABEL: START_LABEL_IDX}            # label dictionary
        self.label_array = [START_LABEL]                            # unique label array
        self.data = data

    def extract_feature_at_pos(self, sentence, pos):
        features = []
        features.append('U02:%s' % sentence[pos][0])
        features.append('U12:%s' % sentence[pos][1])
        return features

    def _add_feature_func(self, prev_label_idx, cur_label_idx, sentence, pos):
        for feature_str in self.extract_feature_at_pos(sentence, pos):
            if feature_str in self.feature_dict.keys():
                if (prev_label_idx, cur_label_idx) in self.feature_dict[feature_str].keys():
                    self.empirical_counts[self.feature_dict[feature_str][(prev_label_idx, cur_label_idx)]] += 1 
                else:
                    feature_id = self.num_features
                    self.feature_dict[feature_str][(prev_label_idx, cur_label_idx)] = feature_id
                    self.empirical_counts[feature_id] += 1
                    self.num_features += 1
                if (-1, cur_label_idx) in self.feature_dict[feature_str].keys():
                    self.empirical_counts[self.feature_dict[feature_str][(-1, cur_label_idx)]] += 1
                else:
                    feature_id = self.num_features
                    self.feature_dict[feature_str][(-1, cur_label_idx)] = feature_id
                    self.empirical_counts[feature_id] += 1
                    self.num_features += 1
            else:
                self.feature_dict[feature_str] = dict()
                # unigram
                feature_id = self.num_features
                self.feature_dict[feature_str][(-1, cur_label_idx)] = feature_id
                self.empirical_counts[feature_id] += 1
                self.num_features += 1
                # bigram
                feature_id = self.num_features
                self.feature_dict[feature_str][(prev_label_idx, cur_label_idx)] = feature_id
                self.empirical_counts[feature_id] += 1
                self.num_features += 1

    def _scan(self):
        """
        generate all feature functions and their counts in the training data
        """
        for sentence in self.data:
            prev_label_idx = START_LABEL_IDX
            for pos in range(len(sentence)):
                cur_label = sentence[pos][-1]
                try:
                    cur_label_idx = self.label_dict[cur_label]
                except KeyError:
                    cur_label_idx = len(self.label_dict.keys())
                    self.label_dict[cur_label] = cur_label_idx
                    self.label_array.append(cur_label)
                self._add_feature_func(prev_label_idx, cur_label_idx, sentence, pos)
                prev_label_idx = cur_label_idx

    def calc_inner_product_score(self, params, sentence, pos):
        inner_product = defaultdict(int)
        for feature_str in self.extract_feature_at_pos(sentence, pos):
            try:
                for (prev_label_idx, cur_label_idx), feature_id in self.feature_dict[feature_str].items():
                    inner_product[(prev_label_idx, cur_label_idx)] += params[feature_id]
            except KeyError:
                pass
        return [((prev_label_idx, cur_label_idx), score) for (prev_label_idx, cur_label_idx), score in inner_product.items()]
        
    def get_feature_func_id(self, sentence, pos):
        feature_func_id_dict = dict()
        for feature_str in self.extract_feature_at_pos(sentence, pos):
            for (prev_label_idx, cur_label_idx), feature_id in self.feature_dict[feature_str].items():
                if (prev_label_idx, cur_label_idx) in feature_func_id_dict.keys():
                    feature_func_id_dict[(prev_label_idx, cur_label_idx)].add(feature_id)
                else:
                    feature_func_id_dict[(prev_label_idx, cur_label_idx)] = {feature_id}
        return [((prev_label_idx, cur_label_idx), feature_ids) for (prev_label_idx, cur_label_idx), feature_ids in feature_func_id_dict.items()]

    def get_empirical_counts(self):
        empirical_counts = np.ndarray((self.num_features,))
        for feature_id, counts in self.empirical_counts.items():
            empirical_counts[feature_id] = counts
        return empirical_counts

    def serialize(self):
        serialized = dict()
        for feature_str in self.feature_dict.keys():
            serialized[feature_str] = dict()
            for (prev_label_idx, cur_label_idx), feature_id in self.feature_dict[feature_str].items():
                serialized[feature_str]['%d_%d' % (prev_label_idx, cur_label_idx)] = feature_id
        return serialized

    def deserialize(self, serialized):
        feature_dict = dict()
        for feature_str in serialized.keys():
            feature_dict[feature_str] = dict()
            for label_pair_str, feature_id in serialized[feature_str].items():
                prev_label, cur_label = label_pair_str.split('_')
                prev_label_idx, cur_label_idx = int(prev_label), int(cur_label)
                feature_dict[feature_str][(prev_label_idx, cur_label_idx)] = feature_id
        return feature_dict

    def load(self, feature_dict):
        self.feature_dict = self.deserialize(feature_dict)

