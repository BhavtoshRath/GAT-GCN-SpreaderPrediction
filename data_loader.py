from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import sklearn
from sklearn import preprocessing


class DatasetSampler(Sampler):

    def __init__(self, n_samples, start=0):
        self.n_samples = n_samples
        self.start = start

    def __iter__(self):
        return iter(range(self.start, self.start + self.n_samples))

    def __len__(self):
        return self.n_samples


class NewsEventDataset(Dataset):

    def __init__(self, seed, shuffle, adj_file, cred_file, label_file, nbr_file, trust_file):
        self.graphs = np.load(adj_file).astype(np.float32)
        identity = np.identity(self.graphs.shape[1])
        self.graphs += identity
        self.graphs[self.graphs != 0] = 1.0
        self.graphs = self.graphs.astype(np.dtype('B'))

        self.credibility_features = np.load(cred_file).astype(np.float32)

        self.labels = np.load(label_file)

        self.vertices = np.load(nbr_file)

        if shuffle:
            self.graphs, self.credibility_features, self.labels, self.vertices = \
                    sklearn.utils.shuffle(
                        self.graphs, self.credibility_features,
                        self.labels, self.vertices,
                        random_state=seed
                    )

        trust_features = np.load(trust_file)
        trust_features = preprocessing.scale(trust_features)  # Standardize/normalize trust features
        self.trust_features = torch.FloatTensor(trust_features)

        self.N = self.graphs.shape[0]

        n_classes = self.get_num_class()
        class_weight = self.N / (n_classes * np.bincount(self.labels))
        self.class_weight = torch.FloatTensor(class_weight)

    def get_trust_features(self):
        return self.trust_features

    def get_credibility_feature_dimension(self):
        return self.credibility_features.shape[-1]

    def get_num_class(self):
        return np.unique(self.labels).shape[0]

    def get_class_weight(self):
        return self.class_weight

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.graphs[idx], self.credibility_features[idx], self.labels[idx], self.vertices[idx]
