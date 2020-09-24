import os
import multiprocessing as mp
from subprocess import call
import warnings
import numpy as np
import scipy.io as sio
import numpy as np
import keras.backend as K
from keras.datasets import mnist, cifar10, cifar100, fashion_mnist
from keras.utils import np_utils
from util import other_class
from tensorflow.python.lib.io import file_io
from io import BytesIO

# Set random seed
np.random.seed(123)

NUM_CLASSES = {'mnist': 10, 'fashion_mnist': 10, 'svhn': 10, 'cifar-10': 10, 'cifar-100': 100, 'celeb': 20}

def get_data(dataset='mnist', noise_ratio=0, data_ratio=100, random_shuffle=False):
    """
    Get training images with specified ratio of label noise
    :param data_ratio: percentage of data used, once use this parameter, the data ration will averagely be used on each class
    :param dataset:
    :param noise_ratio: 0 - 100 (%)
    :param random_shuffle:
    :return:
    """
    if dataset == 'mnist' or dataset == 'fashion_mnist':
        if dataset == 'mnist':
            (X_train, y_train), (X_test, y_test) = mnist.load_data()
        if dataset == 'fashion_mnist':
            (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

        selected_index = []
        un_selected_index = []

        selected_limit = np.zeros(NUM_CLASSES[dataset])

        for i in np.arange(NUM_CLASSES[dataset]):
            selected_limit[i] = X_train.shape[0] * data_ratio / 100.0 / NUM_CLASSES[dataset]

        selected_counter = np.zeros(NUM_CLASSES[dataset])

        for i in np.arange(len(y_train)):
            if selected_counter[y_train[i]] < selected_limit[y_train[i]]:
                selected_index.append(i)
                selected_counter[y_train[i]] += 1
            else:
                un_selected_index.append(i)
        X_train = X_train[selected_index]
        y_train = y_train[selected_index]

        X_train = X_train.reshape(-1, 28, 28, 1)
        X_test = X_test.reshape(-1, 28, 28, 1)

        X_train = X_train / 255.0
        X_test = X_test / 255.0





    elif dataset == 'celeb':
        # gs://ml_engine_data_bucket/data
        f = BytesIO(file_io.read_file_to_string('gs://ml_engine_data_bucket/data/image_train_20.npy', binary_mode=True))
        X_train = np.load(f)
        f = BytesIO(file_io.read_file_to_string('gs://ml_engine_data_bucket/data/image_test_20.npy', binary_mode=True))
        X_test = np.load(f)
        f = BytesIO(file_io.read_file_to_string('gs://ml_engine_data_bucket/data/label_train_20.npy', binary_mode=True))
        y_train = np.load(f)
        f = BytesIO(file_io.read_file_to_string('gs://ml_engine_data_bucket/data/label_test_20.npy', binary_mode=True))
        y_test = np.load(f)

        selected_index = []
        un_selected_index = []
        selected_limit = np.zeros(NUM_CLASSES[dataset])
        number_limit = int(X_train.shape[0] * data_ratio / 100.0)
        extra = number_limit % NUM_CLASSES[dataset]
        for i in np.arange(NUM_CLASSES[dataset]):
            if extra > 0:
                selected_limit[i] = int(X_train.shape[0] * data_ratio / 100.0 / NUM_CLASSES[dataset]) + 1
                extra -= 1
            else:
                selected_limit[i] = int(X_train.shape[0] * data_ratio / 100.0 / NUM_CLASSES[dataset])

        selected_counter = np.zeros(NUM_CLASSES[dataset])

        for i in np.arange(len(y_train)):
            if selected_counter[y_train[i]] < selected_limit[y_train[i]]:
                selected_index.append(i)
                selected_counter[y_train[i]] += 1
            else:
                un_selected_index.append(i)
        X_train = X_train[selected_index]
        y_train = y_train[selected_index]

        X_train = X_train.reshape(-1, 128, 128, 3)
        X_test = X_test.reshape(-1, 128, 128, 3)

        X_train = X_train / 255.0
        X_test = X_test / 255.0

        means = X_train.mean(axis=0)
        # std = np.std(X_train)
        X_train = (X_train - means)  # / std
        X_test = (X_test - means)  # / std

        # they are 2D originally in cifar
        y_train = y_train.ravel()
        y_test = y_test.ravel()

    elif dataset == 'cifar-10':
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        selected_index = []
        un_selected_index = []

        selected_limit = np.zeros(NUM_CLASSES[dataset])

        for i in np.arange(NUM_CLASSES[dataset]):
            selected_limit[i] = X_train.shape[0] * data_ratio / 100.0 / NUM_CLASSES[dataset]

        selected_counter = np.zeros(NUM_CLASSES[dataset])

        for i in np.arange(len(y_train)):
            if selected_counter[y_train[i]] < selected_limit[y_train[i]]:
                selected_index.append(i)
                selected_counter[y_train[i]] += 1
            else:
                un_selected_index.append(i)
        X_train = X_train[selected_index]
        y_train = y_train[selected_index]


        X_train = X_train.reshape(-1, 32, 32, 3)
        X_test = X_test.reshape(-1, 32, 32, 3)

        X_train = X_train / 255.0
        X_test = X_test / 255.0

        means = X_train.mean(axis=0)
        # std = np.std(X_train)
        X_train = (X_train - means)  # / std
        X_test = (X_test - means)  # / std

        # they are 2D originally in cifar
        y_train = y_train.ravel()
        y_test = y_test.ravel()

    elif dataset == 'cifar-100':
        (X_train, y_train), (X_test, y_test) = cifar100.load_data()
        selected_index = []
        un_selected_index = []
        selected_limit = np.zeros(NUM_CLASSES[dataset])

        for i in np.arange(NUM_CLASSES[dataset]):
            selected_limit[i] = X_train.shape[0] * data_ratio / 100.0 / NUM_CLASSES[dataset]

        selected_counter = np.zeros(NUM_CLASSES[dataset])

        for i in np.arange(len(y_train)):
            if selected_counter[y_train[i]] < selected_limit[y_train[i]]:
                selected_index.append(i)
                selected_counter[y_train[i]] += 1
            else:
                un_selected_index.append(i)
        X_train = X_train[selected_index]
        y_train = y_train[selected_index]

        X_train = X_train.reshape(-1, 32, 32, 3)
        X_test = X_test.reshape(-1, 32, 32, 3)

        X_train = X_train / 255.0
        X_test = X_test / 255.0

        means = X_train.mean(axis=0)
        # std = np.std(X_train)
        X_train = (X_train - means)  # / std
        X_test = (X_test - means)  # / std

        # they are 2D originally in cifar
        y_train = y_train.ravel()
        y_test = y_test.ravel()

    else:
        return None, None, None, None, None


    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    # generate random noisy labels
    if noise_ratio > 0:
        n_samples = y_train.shape[0]
        n_noisy = int(noise_ratio*n_samples/100)
        noisy_idx = np.random.choice(n_samples, n_noisy, replace=False)
        for i in noisy_idx:
            y_train[i] = other_class(n_classes=NUM_CLASSES[dataset], current_class=y_train[i])
        # data_file = "data/%s_train_labels_%s.npy" % (dataset, noise_ratio)
        # if os.path.isfile(data_file):
        #     y_train = np.load(data_file)
        # else:
        #     n_samples = y_train.shape[0]
        #     n_noisy = int(noise_ratio*n_samples/100)
        #     noisy_idx = np.random.choice(n_samples, n_noisy, replace=False)
        #     for i in noisy_idx:
        #         y_train[i] = other_class(n_classes=NUM_CLASSES[dataset], current_class=y_train[i])
        #     np.save(data_file, y_train)

    if random_shuffle:
        # random shuffle
        idx_perm = np.random.permutation(X_train.shape[0])
        X_train, y_train = X_train[idx_perm], y_train[idx_perm]

    # one-hot-encode the labels
    y_train = np_utils.to_categorical(y_train, NUM_CLASSES[dataset])
    y_test = np_utils.to_categorical(y_test, NUM_CLASSES[dataset])

    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
    print("X_test:", X_test.shape)
    print("y_test", y_test.shape)

    return X_train, y_train, X_test, y_test, un_selected_index

def get_training_data(dataset='mnist', noise_ratio=0, data_ratio=100, un_selected_index=[], random_shuffle=False):
    """
    Get training images with specified ratio of label noise
    :param data_ratio: percentage of data to be selected, not used this for this function, set a default value to 100.
    :param dataset:
    :param noise_ratio: 0 - 100 (%)
    :param random_shuffle:
    :return:
    """
    if dataset == 'mnist':
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X_train = X_train[un_selected_index]
        y_train = y_train[un_selected_index]


        X_train = X_train.reshape(-1, 28, 28, 1)
        X_test = X_test.reshape(-1, 28, 28, 1)

        X_train = X_train / 255.0
        X_test = X_test / 255.0
    elif dataset == 'cifar-10':
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()


        X_train = X_train[un_selected_index]
        y_train = y_train[un_selected_index]

        X_train = X_train.reshape(-1, 32, 32, 3)
        X_test = X_test.reshape(-1, 32, 32, 3)

        X_train = X_train / 255.0
        X_test = X_test / 255.0

        means = X_train.mean(axis=0)
        # std = np.std(X_train)
        X_train = (X_train - means)  # / std
        X_test = (X_test - means)  # / std

        # they are 2D originally in cifar
        y_train = y_train.ravel()
        y_test = y_test.ravel()

    elif dataset == 'cifar-100':
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()

        X_train = X_train[un_selected_index]
        y_train = y_train[un_selected_index]

        X_train = X_train.reshape(-1, 32, 32, 3)
        X_test = X_test.reshape(-1, 32, 32, 3)

        X_train = X_train / 255.0
        X_test = X_test / 255.0

        means = X_train.mean(axis=0)
        # std = np.std(X_train)
        X_train = (X_train - means)  # / std
        X_test = (X_test - means)  # / std

        # they are 2D originally in cifar
        y_train = y_train.ravel()
        y_test = y_test.ravel()

    else:
        return None, None, None, None


    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    # generate random noisy labels
    if noise_ratio > 0:
        n_samples = y_train.shape[0]
        n_noisy = int(noise_ratio*n_samples/100)
        noisy_idx = np.random.choice(n_samples, n_noisy, replace=False)
        for i in noisy_idx:
            y_train[i] = other_class(n_classes=NUM_CLASSES[dataset], current_class=y_train[i])
        # data_file = "data/%s_train_labels_%s.npy" % (dataset, noise_ratio)
        # if os.path.isfile(data_file):
        #     y_train = np.load(data_file)
        # else:
        #     n_samples = y_train.shape[0]
        #     n_noisy = int(noise_ratio*n_samples/100)
        #     noisy_idx = np.random.choice(n_samples, n_noisy, replace=False)
        #     for i in noisy_idx:
        #         y_train[i] = other_class(n_classes=NUM_CLASSES[dataset], current_class=y_train[i])
        #     np.save(data_file, y_train)

    if random_shuffle:
        # random shuffle
        idx_perm = np.random.permutation(X_train.shape[0])
        X_train, y_train = X_train[idx_perm], y_train[idx_perm]

    # one-hot-encode the labels
    # y_train = np_utils.to_categorical(y_train, NUM_CLASSES[dataset])
    #y_test = np_utils.to_categorical(y_test, NUM_CLASSES[dataset])

    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
    #print("X_test:", X_test.shape)
    #print("y_test", y_test.shape)

    return X_train, y_train

def validatation_split(X, y, split=0.1):
    """
    split data to train and validation set, based on the split ratios
    :param X:
    :param y:
    :param split:
    :return:
    """
    idx_val = np.round(split * X.shape[0]).astype(int)
    X_val, y_val = X[:idx_val], y[:idx_val]
    X_train, y_train = X[idx_val:], y[idx_val:]
    return X_train, y_train, X_val, y_val
