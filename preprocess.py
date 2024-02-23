import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from torch.nn.functional import one_hot

class BCIDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def standardize_data(X_train, X_test, channels): 
    for j in range(channels):
        scaler = StandardScaler()
        scaler.fit(X_train[:, 0, j, :])
        X_train[:, 0, j, :] = scaler.transform(X_train[:, 0, j, :])
        X_test[:, 0, j, :] = scaler.transform(X_test[:, 0, j, :])

    return X_train, X_test

def get_data(path, subject, dataset='BCI2a', classes_labels='all', LOSO=False, isStandard=True, isShuffle=True):
    if LOSO:
        X_train, y_train, X_test, y_test = load_data_LOSO(path, subject, dataset)
    else:
        if dataset == 'BCI2a':
            path = path + 's{:}/'.format(subject+1)
            X_train, y_train = load_BCI2a_data(path, subject+1, True)
            X_test, y_test = load_BCI2a_data(path, subject+1, False)
        elif dataset == 'CS2R':
            X_train, y_train, _, _, _ = load_CS2R_data_v2(path, subject, True, classes_labels)
            X_test, y_test, _, _, _ = load_CS2R_data_v2(path, subject, False, classes_labels)
        else:
            raise Exception("'{}' dataset is not supported yet!".format(dataset))

    if isShuffle:
        X_train, y_train = shuffle(X_train, y_train, random_state=42)
        X_test, y_test = shuffle(X_test, y_test, random_state=42)

    N_tr, N_ch, T = X_train.shape 
    X_train = X_train.reshape(N_tr, 1, N_ch, T)
    y_train_onehot = one_hot(torch.tensor(y_train), num_classes=len(np.unique(y_train)))

    N_tr, N_ch, T = X_test.shape 
    X_test = X_test.reshape(N_tr, 1, N_ch, T)
    y_test_onehot = one_hot(torch.tensor(y_test), num_classes=len(np.unique(y_test)))    

    if isStandard:
        X_train, X_test = standardize_data(X_train, X_test, N_ch)

    train_dataset = BCIDataset(X_train, y_train_onehot)
    test_dataset = BCIDataset(X_test, y_test_onehot)

    return train_dataset, test_dataset
