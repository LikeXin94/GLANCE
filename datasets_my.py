import os, random, sys
import numpy as np
import scipy.io as sio
import util
from scipy import sparse
from sklearn.preprocessing import MinMaxScaler# 对数据标准化处理
tool = MinMaxScaler(feature_range=(0, 1))
import h5py


def load_data_my(config):
    """Load data """
    data_name = config['dataset']
    # main_dir = sys.path[0]
    X_list = []
    Y_list = []
    main_dir = os.path.dirname(os.getcwd())
    if data_name in ['Scene_15']:
        mat = sio.loadmat(os.path.join(main_dir, 'test_completer_20220410/data', 'Scene-15.mat'))
        X = mat['X'][0]
        X_list.append(X[0].astype('float32'))
        X_list.append(X[1].astype('float32'))
        Y_list.append(np.squeeze(mat['Y']))

    elif data_name in ['digit_6view']:
        mat = sio.loadmat(os.path.join(main_dir, 'test_completer_20220410/data', 'digit_6views.mat'))
        X = mat['tmpX'][0]
        for v in range(6):
            X_list.append(X[v].astype('float32'))
            X_list[v] = tool.fit_transform(X_list[v])
        Y_list.append(np.squeeze(mat['tmpY']))

    return X_list, Y_list

