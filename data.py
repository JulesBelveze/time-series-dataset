import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, Subset


class TimeSeriesDataset(object):
    def __init__(self, data, index_output_col=-1, seq_length=1, prediction_window=1, ylog=True):
        '''Params:
        - data: numpy array of shape nb_obs * nb_features with target cols as last columns
        - index_output_col: index of the column we want to predict
        - seq_length: window length to use
        - prediction_time: window length to predict'''
        self.data = data
        self.index_output_col = index_output_col
        self.seq_length = seq_length
        self.prediction_window = prediction_window
        self.ylog = ylog
        self.frame_series()

    def frame_series(self):
        nb_obs = self.data.shape[0]
        X = self.data[:, :self.index_output_col]
        y = self.data[:, self.index_output_col]

        features, labels = [], []
        for i in range(nb_obs - self.seq_length - self.prediction_window):
            features.append(X[i:i + self.seq_length, :])

        for i in range(self.seq_len, nb_obs - self.prediction_window):
            labels.append(y[i: i + self.prediction_window])

        x_arr = np.array(features).reshape((len(features), features[0].shape[0], features[0].shape[1]))
        y_arr = np.array(labels).reshape(len(labels), labels[0].shape[0])
        if self.ylog:
            y_arr = np.log(y_arr)

        x_var, y_var = torch.Tensor(x_arr), torch.Tensor(y_arr)
        self.data = TensorDataset(x_var, y_var)

    def get_loaders(self, train_prop=.8, batch_size=batch_size):
        '''Get DataLoader for both training and testing sets'''
        len_data = len(self.data)
        train_max_index = self.batch_safe_len(train_prop * len_data)
        test_max_index = train_max_index + self.batch_safe_len(len_data - train_max_index)
        train_set = Subset(self.data, range(train_max_index))
        test_set = Subset(self.data, range(train_max_index, test_max_index))
        train_iter = DataLoader(train_set, batch_size=batch_size, shuffle=False)
        test_iter = DataLoader(test_set, batch_size=batch_size, shuffle=False)
        return train_iter, test_iter

    def batch_safe_len(self, num, batch_size=batch_size):
        return int(num - (num % batch_size))