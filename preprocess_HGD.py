import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.utils import shuffle
from torchvision import transforms

# Load High Gamma Dataset (HGD)
import numpy as np
import logging
from collections import OrderedDict
#from braindecode.datasets.bbci import BBCIDataset
""" from braindecode.datautil.trial_segment import create_signal_target
from braindecode.mne_ext.signalproc import mne_apply, resample_cnt
from braindecode.datautil.signalproc import exponential_running_standardize
from braindecode.datautil.signalproc import highpass_cnt """
import braindecode
import torch
from torch.utils.data import Dataset

class HGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def load_HGD_data(data_path, subject, training, low_cut_hz=0, debug=False):
    log = logging.getLogger(__name__)
    log.setLevel('DEBUG')

    if training:
        filename = (data_path + 'train/{}.mat'.format(subject))
    else:
        filename = (data_path + 'test/{}.mat'.format(subject))

    load_sensor_names = None
    if debug:
        load_sensor_names = ['C3', 'C4', 'C2']

    loader = BBCIDataset(filename, load_sensor_names=load_sensor_names)

    log.info("Loading data...")
    cnt = loader.load()

    log.info("Cutting trials...")
    marker_def = OrderedDict([('Right Hand', [1]), ('Left Hand', [2],),
                              ('Rest', [3]), ('Feet', [4])])
    clean_ival = [0, 4000]

    set_for_cleaning = create_signal_target(cnt, marker_def, clean_ival)
    clean_trial_mask = np.max(np.abs(set_for_cleaning.X), axis=(1, 2)) < 800

    log.info("Clean trials: {:3d}  of {:3d} ({:5.1f}%)".format(
        np.sum(clean_trial_mask),
        len(set_for_cleaning.X),
        np.mean(clean_trial_mask) * 100))

    C_sensors = ['FC5', 'FC1', 'FC2', 'FC6', 'C3', 'C4', 'CP5',
                 'CP1', 'CP2', 'CP6', 'FC3', 'FCz', 'FC4', 'C5', 'C1', 'C2',
                 'C6',
                 'CP3', 'CPz', 'CP4', 'FFC5h', 'FFC3h', 'FFC4h', 'FFC6h',
                 'FCC5h',
                 'FCC3h', 'FCC4h', 'FCC6h', 'CCP5h', 'CCP3h', 'CCP4h', 'CCP6h',
                 'CPP5h',
                 'CPP3h', 'CPP4h', 'CPP6h', 'FFC1h', 'FFC2h', 'FCC1h', 'FCC2h',
                 'CCP1h',
                 'CCP2h', 'CPP1h', 'CPP2h']
    if debug:
        C_sensors = load_sensor_names
    cnt = cnt.pick_channels(C_sensors)

    log.info("Resampling...")
    cnt = resample_cnt(cnt, 250.0)
    log.info("Highpassing...")
    cnt = mne_apply(
        lambda a: highpass_cnt(
            a, low_cut_hz, cnt.info['sfreq'], filt_order=3, axis=1),
        cnt)
    log.info("Standardizing...")
    cnt = mne_apply(
        lambda a: exponential_running_standardize(a.T, factor_new=1e-3,
                                                  init_block_size=1000,
                                                  eps=1e-4).T,
        cnt)

    ival = [-500, 4000]

    dataset = create_signal_target_from_raw_mne(cnt, marker_def, ival)
    dataset.X = dataset.X[clean_trial_mask]
    dataset.y = dataset.y[clean_trial_mask]

    return HGDataset(dataset.X, dataset.y)


# Define the dataset class
class EEGDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample = {'eeg': self.X[idx], 'label': self.y[idx]}

        if self.transform:
            sample = self.transform(sample)

        return sample

# Custom transform to standardize the EEG data
class StandardizeEEG(object):
    def __init__(self, channels):
        self.channels = channels
        self.scalers = [StandardScaler() for _ in range(channels)]

    def __call__(self, sample):
        eeg, label = sample['eeg'], sample['label']

        for j in range(self.channels):
            eeg[:, j, :] = self.scalers[j].transform(eeg[:, j, :])

        return {'eeg': eeg, 'label': label}

# Get EEG data for PyTorch
def get_data(path, subject, dataset='BCI2a', classes_labels='all', LOSO=False, isStandard=True, isShuffle=True):
    # Load and split the dataset into training and testing
    if LOSO:
        # Load data using the original function
        X_train, y_train, X_test, y_test = load_data_LOSO(path, subject, dataset)
    else:
        if dataset == 'BCI2a':
            path = path + 's{:}/'.format(subject + 1)
            X_train, y_train = load_BCI2a_data(path, subject + 1, True)
            X_test, y_test = load_BCI2a_data(path, subject + 1, False)
        elif dataset == 'CS2R':
            X_train, y_train, _, _, _ = load_CS2R_data_v2(path, subject, True, classes_labels)
            X_test, y_test, _, _, _ = load_CS2R_data_v2(path, subject, False, classes_labels)
        elif dataset == 'HGD':
            # Load HGD data using the provided function
            X_train, y_train = load_HGD_data(path, subject + 1, True)
            X_test, y_test = load_HGD_data(path, subject + 1, False)
        else:
            raise Exception("'{}' dataset is not supported yet!".format(dataset))

    # Shuffle the data
    if isShuffle:
        X_train, y_train = shuffle(X_train, y_train, random_state=42)
        X_test, y_test = shuffle(X_test, y_test, random_state=42)

    # Prepare training data
    N_tr, N_ch, T = X_train.shape
    X_train = X_train.reshape(N_tr, 1, N_ch, T)
    y_train_onehot = torch.from_numpy(y_train).long()

    # Prepare testing data
    N_tr, N_ch, T = X_test.shape
    X_test = X_test.reshape(N_tr, 1, N_ch, T)
    y_test_onehot = torch.from_numpy(y_test).long()

    # Standardize the data
    if isStandard:
        channels = N_ch
        transform = transforms.Compose([StandardizeEEG(channels)])
        dataset_train = EEGDataset(X_train, y_train_onehot, transform=transform)
        dataset_test = EEGDataset(X_test, y_test_onehot, transform=transform)
    else:
        dataset_train = TensorDataset(torch.Tensor(X_train), y_train_onehot)
        dataset_test = TensorDataset(torch.Tensor(X_test), y_test_onehot)

    return dataset_train, dataset_test

# Example usage
path = 'your_dataset_path/'
subject = 1
dataset_train, dataset_test = get_data(path, subject, dataset='BCI2a', classes_labels='all', LOSO=False, isStandard=True, isShuffle=True)
train_loader = DataLoader(dataset_train, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset_test, batch_size=32, shuffle=False)