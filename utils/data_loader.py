from re import T
from typing import Optional
from scipy import signal
from scipy.signal import resample
import scipy.io as scio
import numpy as np


def split(x, y, ratio=0.8, shuffle=True):
    idx = np.arange(len(x))
    if shuffle:
        idx = np.random.permutation(idx)
    train_size = int(len(x) * ratio)

    return x[idx[:train_size]], y[idx[:train_size]], x[idx[train_size:]], y[
        idx[train_size:]]


def MI4CLoad(id: int, setup: Optional[str] = 'within'):
    data_path = '../EEG_data/MI4C/processed/'
    x_train, y_train, x_test, y_test = [], [], [], []
    if setup == 'within':
        data = scio.loadmat(data_path + f's{id}.mat')
        x, y = data['x'], data['y']
        y = np.squeeze(y.flatten())
        x_train, y_train, x_test, y_test = split(x, y, ratio=0.8, shuffle=True)
    elif setup == 'cross':
        for i in range(9):
            data = scio.loadmat(data_path + f's{i}.mat')
            if i == id:
                x_test, y_test = data['x'], data['y']
            else:
                x_train.extend(data['x'])
                y_train.extend(data['y'])
        x_train = np.array(x_train)
        y_train = np.concatenate(y_train, axis=0)
    else:
        raise Exception('No such Experiment setup.')

    return x_train, y_train.squeeze(), x_test, y_test.squeeze()


def ERNLoad(id: int, setup: Optional[str] = 'within'):
    data_path = '../EEG_data/ERN/processed/'
    x_train, y_train, x_test, y_test = [], [], [], []
    if setup == 'within':
        data = scio.loadmat(data_path + f's{id}.mat')
        x, y = data['x'], data['y']
        y = np.squeeze(y.flatten())
        x_train, y_train, x_test, y_test = split(x, y, ratio=0.8, shuffle=True)
    elif setup == 'cross':
        for i in range(16):
            data = scio.loadmat(data_path + f's{i}.mat')
            if i == id:
                x_test, y_test = data['x'], data['y']
            else:
                x_train.extend(data['x'])
                y_train.extend(data['y'])
        x_train = np.array(x_train)
        y_train = np.concatenate(y_train, axis=0)
    else:
        raise Exception('No such Experiment setup')

    return x_train, y_train.squeeze(), x_test, y_test.squeeze()


def EPFLLoad(id: int, setup: Optional[str] = 'within'):
    data_path = '../EEG_data/EPFL/'
    x_train, y_train, x_test, y_test = [], [], [], []
    if setup == 'within':
        data = scio.loadmat(data_path + f's{id}.mat')
        x, y = data['x'], data['y']
        y = np.squeeze(y.flatten())
        x_train, y_train, x_test, y_test = split(x, y, ratio=0.8, shuffle=True)
    elif setup == 'cross':
        for i in range(8):
            data = scio.loadmat(data_path + f's{i}.mat')
            if i == id:
                x_test, y_test = data['x'], data['y']
            else:
                x_train.extend(data['x'])
                y_train.extend(data['y'])
        x_train = np.array(x_train)
        y_train = np.concatenate(y_train, axis=0)
    else:
        raise Exception('No such Experiment setup')
    
    return x_train, y_train.squeeze(), x_test, y_test.squeeze()


def BNCILoad(id: int, setup: Optional[str] = 'within'):
    data_path = '../data7/MIData/BNCI2014-001-4/'
    x_train, y_train, x_test, y_test = [], [], [], []
    if setup == 'within':
        data = scio.loadmat(data_path + f'A{id+1}.mat')
        x, y = data['X'], data['y']
        y = np.squeeze(y.flatten())
        x_train, y_train, x_test, y_test = split(x, y, ratio=0.8, shuffle=True)
    elif setup == 'cross':
        for i in range(9):
            data = scio.loadmat(data_path + f'A{i+1}.mat')
            if i == id:
                x_test, y_test = data['X'], data['y']
            else:
                x_train.extend(data['X'])
                y_train.extend(data['y'])
        x_train = np.array(x_train)
        y_train = np.array(y_train)
    else:
        raise Exception('No such Experiment setup.')

    # resample
    x_train = signal.resample(x_train,
                              num=int(x_train.shape[2] * 128 / 250),
                              axis=2)
    x_test = signal.resample(x_test,
                             num=int(x_test.shape[2] * 128 / 250),
                             axis=2)
    # replace label
    # label_dict = {'left_hand ': 0, 'right_hand': 1}
    label_dict = {
        'left_hand ': 0,
        'right_hand': 1,
        'feet      ': 2,
        'tongue    ': 3
    }
    y_train = np.array([label_dict[x] for x in y_train])
    y_test = np.array([label_dict[x] for x in y_test])

    x_train = x_train[:, np.newaxis, :, :]
    x_test = x_test[:, np.newaxis, :, :]

    return x_train, y_train.squeeze(), x_test, y_test.squeeze()
