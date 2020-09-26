from __future__ import absolute_import
from __future__ import print_function

import os
import multiprocessing as mp
from subprocess import call
import warnings
import numpy as np
from numpy.testing import assert_array_almost_equal
from sklearn.preprocessing import MinMaxScaler
import keras.backend as K
from scipy.spatial.distance import pdist, cdist, squareform
from keras.callbacks import ModelCheckpoint, Callback
from keras.callbacks import LearningRateScheduler
import tensorflow as tf
from keras.utils import np_utils
import random

# Set random seed
np.random.seed(123)
NUM_CLASSES = {'mnist': 10, 'svhn': 10, 'cifar-10': 10, 'cifar-100': 100, 'celeb': 20}

def lid(logits, k=20):
    """
    Calculate LID for a minibatch of training samples based on the outputs of the network.

    :param logits:
    :param k:
    :return:
    """
    epsilon = 1e-12
    batch_size = tf.shape(logits)[0]
    # n_samples = logits.get_shape().as_list()
    # calculate pairwise distance
    r = tf.reduce_sum(logits * logits, 1)
    # turn r into column vector
    r1 = tf.reshape(r, [-1, 1])
    D = r1 - 2 * tf.matmul(logits, tf.transpose(logits)) + tf.transpose(r1) + \
        tf.ones([batch_size, batch_size])

    # find the k nearest neighbor
    D1 = -tf.sqrt(D)
    D2, _ = tf.nn.top_k(D1, k=k, sorted=True)
    D3 = -D2[:, 1:]  # skip the x-to-x distance 0 by using [,1:]

    m = tf.transpose(tf.multiply(tf.transpose(D3), 1.0 / D3[:, -1]))
    v_log = tf.reduce_sum(tf.log(m + epsilon), axis=1)  # to avoid nan
    lids = -k / v_log
    return lids


def mle_single(data, x, k):
    """
    lid of a single query point x.
    numpy implementation.

    :param data:
    :param x:
    :param k:
    :return:
    """
    data = np.asarray(data, dtype=np.float32)
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape((-1, x.shape[0]))
    # dim = x.shape[1]

    k = min(k, len(data) - 1)
    f = lambda v: - k / np.sum(np.log(v / v[-1] + 1e-8))
    a = cdist(x, data)
    a = np.apply_along_axis(np.sort, axis=1, arr=a)[:, 1:k + 1]
    a = np.apply_along_axis(f, axis=1, arr=a)
    return a[0]


def mle_batch(data, batch, k):
    """
    lid of a batch of query points X.
    numpy implementation.

    :param data:
    :param batch:
    :param k:
    :return:
    """
    data = np.asarray(data, dtype=np.float32)
    batch = np.asarray(batch, dtype=np.float32)

    k = min(k, len(data) - 1)
    f = lambda v: - k / np.sum(np.log(v / v[-1] + 1e-8))
    a = cdist(batch, data)
    a = np.apply_along_axis(np.sort, axis=1, arr=a)[:, 1:k + 1]
    a = np.apply_along_axis(f, axis=1, arr=a)
    return a


def other_class(n_classes, current_class):
    """
    Returns a list of class indices excluding the class indexed by class_ind
    :param nb_classes: number of classes in the task
    :param class_ind: the class index to be omitted
    :return: one random class that != class_ind
    """
    #print(current_class)
    if current_class < 0 or current_class >= n_classes:
        error_str = "class_ind must be within the range (0, nb_classes - 1)"
        raise ValueError(error_str)

    other_class_list = list(range(n_classes))
    other_class_list.remove(current_class)
    other_class = np.random.choice(other_class_list)
    return other_class


def get_lids_random_batch(model, X, k=20, batch_size=128):
    """
    Get the local intrinsic dimensionality of each Xi in X_adv
    estimated by k close neighbours in the random batch it lies in.
    :param model: if None: lid of raw inputs, otherwise LID of deep representations
    :param X: normal images
    :param k: the number of nearest neighbours for LID estimation
    :param batch_size: default 100
    :return: lids: LID of normal images of shape (num_examples, lid_dim)
            lids_adv: LID of advs images of shape (num_examples, lid_dim)
    """
    if model is None:
        lids = []
        n_batches = int(np.ceil(X.shape[0] / float(batch_size)))
        for i_batch in range(n_batches):
            start = i_batch * batch_size
            end = np.minimum(len(X), (i_batch + 1) * batch_size)
            X_batch = X[start:end].reshape((end - start, -1))

            # Maximum likelihood estimation of local intrinsic dimensionality (LID)
            lid_batch = mle_batch(X_batch, X_batch, k=k)
            lids.extend(lid_batch)

        lids = np.asarray(lids, dtype=np.float32)
        return lids

    # get deep representations
    funcs = [K.function([model.layers[0].input, K.learning_phase()], [out])
             for out in [model.get_layer("lid").output]]
    lid_dim = len(funcs)

    #     print("Number of layers to estimate: ", lid_dim)

    def estimate(i_batch):
        start = i_batch * batch_size
        end = np.minimum(len(X), (i_batch + 1) * batch_size)
        n_feed = end - start
        lid_batch = np.zeros(shape=(n_feed, lid_dim))
        for i, func in enumerate(funcs):
            X_act = func([X[start:end], 0])[0]
            X_act = np.asarray(X_act, dtype=np.float32).reshape((n_feed, -1))

            # Maximum likelihood estimation of local intrinsic dimensionality (LID)
            lid_batch[:, i] = mle_batch(X_act, X_act, k=k)

        return lid_batch

    lids = []
    n_batches = int(np.ceil(X.shape[0] / float(batch_size)))
    for i_batch in range(n_batches):
        lid_batch = estimate(i_batch)
        lids.extend(lid_batch)

    lids = np.asarray(lids, dtype=np.float32)

    return lids


def get_lr_scheduler(dataset):
    """
    customerized learning rate decay for training with clean labels.
     For efficientcy purpose we use large lr for noisy data.
    :param dataset:
    :param noise_ratio:
    :return:
    """
    if dataset in ['mnist', 'svhn']:
        def scheduler(epoch):
            if epoch > 40:
                return 0.001
            elif epoch > 20:
                return 0.01
            else:
                return 0.1

        return LearningRateScheduler(scheduler)
    elif dataset in ['cifar-10']:
        def scheduler(epoch):
            if epoch > 80:
                return 0.001
            elif epoch > 40:
                return 0.01
            else:
                return 0.1

        return LearningRateScheduler(scheduler)
    elif dataset in ['cifar-100']:
        def scheduler(epoch):
            if epoch > 120:
                return 0.001
            elif epoch > 80:
                return 0.01
            else:
                return 0.1

        return LearningRateScheduler(scheduler)


def uniform_noise_model_P(num_classes, noise):
    """ The noise matrix flips any class to any other with probability
    noise / (num_classes - 1).
    """

    assert (noise >= 0.) and (noise <= 1.)

    P = noise / (num_classes - 1) * np.ones((num_classes, num_classes))
    np.fill_diagonal(P, (1 - noise) * np.ones(num_classes))

    assert_array_almost_equal(P.sum(axis=1), 1, 1)
    return P


def get_deep_representations(model, X, batch_size=128):
    """
    Get the deep representations before logits.
    :param model:
    :param X:
    :param batch_size:
    :return:
    """
    # last hidden layer is always at index -4
    output_dim = model.layers[-3].output.shape[-1].value
    get_encoding = K.function(
        [model.layers[0].input, K.learning_phase()],
        [model.layers[-3].output]
    )

    n_batches = int(np.ceil(X.shape[0] / float(batch_size)))
    output = np.zeros(shape=(len(X), output_dim))
    for i in range(n_batches):
        output[i * batch_size:(i + 1) * batch_size] = \
            get_encoding([X[i * batch_size:(i + 1) * batch_size], 0])[0]

    return output

# make prediction by quality_model, compare the predicted label and y_train
def select_clean_uncertain_by_quality(X_train, y_train, quality_model, return_prediction = False):
    y_predict_label = quality_model.predict(X_train, batch_size=128, verbose=0, steps=None)
    y_predict_label = y_predict_label.argmax(axis=-1)
    clean_list_quality = []
    dirty_list_quality = []
    for i in np.arange(len(y_predict_label)):
        if (np.argmax(y_train[i]) == y_predict_label[i]):
	        clean_list_quality.append(i)
        else:
            dirty_list_quality.append(i)

    # y_predict_label_classifier = classifier.predict(X_train[dirty_list_quality], batch_size=128, verbose=0, steps=None)
    # y_predict_label_classifier = y_predict_label_classifier.argmax(axis=-1)
    # clean_list_classifier = []
    # dirty_list_classifier = []
    # y_train_classifier = y_train[dirty_list_quality]
    # for i in np.arange(len(dirty_list_quality)):
    #     if (np.argmax(y_train_classifier[i]) == y_predict_label_classifier[i]):
	#         clean_list_classifier.append(i)
    #     else:
    #         dirty_list_classifier.append(i)
    #
    # clean_list = np.append(clean_list_classifier, clean_list_quality)
    # uncertain_list = dirty_list_classifier

    #return clean_list, uncertain_list
    if return_prediction:
        return clean_list_quality, dirty_list_quality, quality_model.predict(X_train, batch_size=128, verbose=0, steps=None)
    else:
        return clean_list_quality, dirty_list_quality

def select_clean_uncertain(X_train, y_train, quality_model, classifier):
    y_predict_label = quality_model.predict(X_train, batch_size=128, verbose=0, steps=None)
    y_predict_label = y_predict_label.argmax(axis=-1)
    clean_list_quality = []
    dirty_list_quality = []
    for i in np.arange(len(y_predict_label)):
        if (np.argmax(y_train[i]) == y_predict_label[i]):
	        clean_list_quality.append(i)
        else:
            dirty_list_quality.append(i)

    y_predict_label_classifier = classifier.predict(X_train[dirty_list_quality], batch_size=128, verbose=0, steps=None)
    y_predict_label_classifier = y_predict_label_classifier.argmax(axis=-1)
    clean_list_classifier = []
    dirty_list_classifier = []
    y_train_classifier = y_train[dirty_list_quality]
    for i in np.arange(len(dirty_list_quality)):
        if (np.argmax(y_train_classifier[i]) == y_predict_label_classifier[i]):
	        clean_list_classifier.append(dirty_list_quality[i])
        else:
            dirty_list_classifier.append(dirty_list_quality[i])

    clean_list = np.append(clean_list_classifier, clean_list_quality)
    uncertain_list = dirty_list_classifier
    # clean_list.sort()
    # uncertain_list.sort()
    #return clean_list, uncertain_list
    return clean_list, uncertain_list




# make prediction by classifier
# def select_clean_quality(X_train, y_train, quality_model, dataset):
#     y_predict_label = quality_model.predict(X_train, batch_size=128, verbose=0, steps=None)
#     y_predict_label = y_predict_label.argmax(axis=-1)
#     clean_list = []
#     for i in np.arange(len(y_predict_label)):
#         if (np.argmax(y_train[i]) == y_predict_label[i]):
# 	    clean_list.append(i)
#     X_clean_train = X_train[clean_list]
#     y_clean_train = np_utils.to_categorical(y_train[clean_list], NUM_CLASSES[dataset])
#     print("X_clean_train:", X_clean_train.shape)
#     print("y_clean_train:", y_clean_train.shape)
#     return X_clean_train, y_clean_train, clean_list;

# make prediction by quality_model, compare the predicted label and y_train
def select_clean_knn(X_train, y_train, quality_model,dataset):
    X_train_linear = np.reshape(X_train, (X_train.shape[0], -1))
    y_predict_label = quality_model.predict(X_train_linear, k=10)
    #y_predict_label = y_predict_label.argmax(axis=-1)
    clean_list = []
    for i in np.arange(len(y_predict_label)):
        if (y_train[i].argmax(axis=-1) == y_predict_label[i].argmax(axis=-1)):
	        clean_list.append(i)
    X_clean_train = X_train[clean_list]
    #y_clean_train = np_utils.to_categorical(y_train[clean_list], NUM_CLASSES[dataset])
    print("X_clean_train:", X_clean_train.shape)
    print("y_clean_train:", y_train[clean_list].shape)
    return X_clean_train, y_train[clean_list];

# combine two h
def combine_result(h, h_training_epoch):
    h.history['acc'] += h_training_epoch.history['acc']
    h.history['loss'] += h_training_epoch.history['loss']
    h.history['val_acc'] += h_training_epoch.history['val_acc']
    h.history['val_loss'] += h_training_epoch.history['val_loss']
    return h;

def inject_noise(dataset, y, noise_level):
    if noise_level > 100 or noise_level < 0:
        raise ValueError('Noise level can not be bigger than 100 or smaller than 0')

    noisy_idx = np.random.choice(len(y), int(len(y)*noise_level/100.0), replace = False)
    for i in noisy_idx:
        y[i] = np_utils.to_categorical(other_class(NUM_CLASSES[dataset], y[i].argmax(axis=-1)), NUM_CLASSES[dataset])

    return y

def statistic_oracle(y_noise, y_clean):
    assert y_clean.shape[0] == y_noise.shape[0]
    correct = 0.0
    wrong = 0.0
    for i in range(y_clean.shape[0]):
        if np.argmax(y_noise[i]) == np.argmax(y_clean[i]):
            correct += 1.0
        else:
            wrong += 1.0
    return correct, wrong, correct/(correct+wrong)

def cross_entropy(predictions, targets, epsilon=1e-12):
    """
    Computes cross entropy between targets (encoded as one-hot vectors)
    and predictions.
    Input: predictions (N, k) ndarray
           targets (N, k) ndarray
    Returns: scalar
    """
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    N = predictions.shape[0]
    ce = -np.sum(targets*np.log(predictions+1e-9))/N
    return ce

def uncertain_ranking(uncertain_list, y_noisy_iteration, y_predict, max_n):
    assert max_n <= len(uncertain_list)
    distance_list = []
    for i in uncertain_list:
        #distance_list.append(np.linalg.norm(y_predict[i]-y_noisy_iteration[i]))
        distance_list.append(cross_entropy(y_predict[i],y_noisy_iteration[i]))
    distance_arr = np.array(distance_list)
    return np.array(uncertain_list)[distance_arr.argsort()[-max_n:][::-1]]


def flip_label(y_train, noise_level = 1):
    print("noise level: ", noise_level)
    y = y_train.copy()
    n_class = y.shape[1]
    for i in range(y_train.shape[0]):
        if random.uniform(0, 1) < noise_level:
            y[i] = np_utils.to_categorical(other_class(n_class, y[i].argmax(axis=-1)), n_class)
    return y
def set_to_one(y_train):
    y = y_train.copy()
    n_class = y.shape[1]
    for i in range(y.shape[0]):
        y[i] = np_utils.to_categorical(1, n_class)
    return y
def set_to_zero(y_train):
    y = y_train.copy()
    n_class = y.shape[1]
    for i in range(y.shape[0]):
        y[i] = np_utils.to_categorical(0, n_class)
    return y
