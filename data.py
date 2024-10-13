import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import TimeSeriesSplit
from typing import List, Optional, Tuple


class TimeSeriesDataset:
    """
    A class for preprocessing and loading time series data for pytorch models.

    attributes:
        data (pd.DataFrame): the input time series data.
        categorical_cols (List[str]): list of categorical column names.
        target_col (str): name of the target column.
        seq_length (int): length of the input sequence.
        prediction_window (int): length of the prediction window.
        numerical_cols (List[str]): list of numerical column names.
        preprocessor (ColumnTransformer): sklearn preprocessor for data transformation.
    """

    def __init__(self,
                 data: pd.DataFrame,
                 categorical_cols: List[str],
                 target_col: str,
                 seq_length: int,
                 prediction_window: int = 1):
        """
        Initialize the TimeSeriesDataset.

        args:
            data (pd.DataFrame): the input time series data.
            categorical_cols (List[str]): list of categorical column names.
            target_col (str): name of the target column.
            seq_length (int): length of the input sequence.
            prediction_window (int): length of the prediction window.
        """
        self.data = data
        self.categorical_cols = categorical_cols
        self.numerical_cols = list(set(data.columns) - set(categorical_cols) - {target_col})
        self.target_col = target_col
        self.seq_length = seq_length
        self.prediction_window = prediction_window
        self.preprocessor = None

    def preprocess_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Preprocess the data using sklearn ColumnTransformer.

        returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: preprocessed training and testing data.
        """
        X = self.data.drop(self.target_col, axis=1)
        y = self.data[self.target_col]

        self.preprocessor = ColumnTransformer(
            [("scaler", StandardScaler(), self.numerical_cols),
             ("encoder", OneHotEncoder(sparse=False, handle_unknown='ignore'), self.categorical_cols)],
            remainder="passthrough"
        )

        # use timeseriessplit for time series data :cite[c4]
        tscv = TimeSeriesSplit(n_splits=5)
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        X_train = self.preprocessor.fit_transform(X_train)
        X_test = self.preprocessor.transform(X_test)

        return X_train, X_test, y_train.values, y_test.values

    def frame_series(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> TensorDataset:
        """
        Create a TensorDataset from the input data.

        args:
            X (np.ndarray): input features.
            y (Optional[np.ndarray]): target values.

        returns:
            TensorDataset: dataset containing the framed series.
        """
        nb_obs, nb_features = X.shape
        features, target, y_hist = [], [], []

        for i in range(nb_obs - self.seq_length - self.prediction_window + 1):
            features.append(torch.FloatTensor(X[i:i + self.seq_length, :]))

        features_var = torch.stack(features)

        if y is not None:
            for i in range(nb_obs - self.seq_length - self.prediction_window + 1):
                target.append(torch.FloatTensor(y[i + self.seq_length:i + self.seq_length + self.prediction_window]))
                y_hist.append(
                    torch.FloatTensor(y[i + self.seq_length - 1:i + self.seq_length + self.prediction_window - 1]))

            target_var, y_hist_var = torch.stack(target), torch.stack(y_hist)
            return TensorDataset(features_var, target_var, y_hist_var)

        return TensorDataset(features_var)

    def get_loaders(self, batch_size: int) -> Tuple[DataLoader, DataLoader]:
        """
        Create DataLoader objects for training and testing data.

        args:
            batch_size (int): size of each batch.

        returns:
            Tuple[DataLoader, DataLoader]: DataLoader objects for training and testing data.
        """
        X_train, X_test, y_train, y_test = self.preprocess_data()
        train_dataset = self.frame_series(X_train, y_train)
        test_dataset = self.frame_series(X_test, y_test)

        train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
        test_iter = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

        return train_iter, test_iter
