import torch
from torch.utils.data import TensorDataset, DataLoader, Subset
from torch.autograd import Variable


class TimeSeriesDataset(object):
    def __init__(self, data, index_output_col=-1, seq_length=1, prediction_time=1):
        '''Params:
        - data: numpy array of shape nb_obs * nb_features with target cols
        as last columns
        - index_output_col: index of the column we want to predict
        - seq_length: window length to use
        - prediction_time: window length to predict'''
        self.data = data
        self.index_output_col = index_output_col
        self.seq_length = seq_length
        self.prediction_time = prediction_time
        self.frame_series()

    def frame_series(self):
        X = self.data[:-self.prediction_time]
        y = self.data[:, self.index_output_col]

        features, labels = [], []
        for i in range(self.data.shape[0] - self.seq_length - self.prediction_time):
            x_i = X[i: i + self.seq_length]
            y_i = y[i + self.seq_length: i + self.seq_length + self.prediction_time]
            features.append(x_i)
            labels.append(y_i)

        x_arr = np.array(features).reshape((len(features), features[0].shape[0], features[0].shape[1]))
        y_arr = np.array(labels).reshape(len(labels), labels[0].shape[0], 1)

        x_var = torch.from_numpy(x_arr)
        y_var = torch.from_numpy(y_arr)

        self.data = TensorData(x_var, y_var)

    def get_loaders(self, train_prop=.8, batch_size=32):
        '''Get DataLoader for both training and testing sets'''
        len_data = len(self.data)
        train_indices = range(int(train_prop * len_data))
        test_indices = range(int(train_prop * len_data), len_data)

        train_set = Subset(data, train_indices)
        test_set = Subset(data, test_indices)

        train_iter = DataLoader(train_set, batch_size=batch_size, shuffle=False)
        test_iter = DataLoader(test_set, batch_size=batch_size, shuffle=False)
        return train_iter, test_iter
