#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import math
import itertools
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.models import clone_model
from datasets import get_data, get_training_data
from models import get_model, resnet_v1, resnet_v2
from util import select_clean_uncertain, combine_result, inject_noise, flip_label, set_to_one, set_to_zero
import time
import argparse
from tensorflow.python.lib.io import file_io
from keras.utils import np_utils, multi_gpu_model
from keras import backend as K
from io import BytesIO
from loss_acc_plot import loss_acc_plot
from keras.datasets import mnist, cifar10, cifar100, fashion_mnist
import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
# In[2]:


NUM_CLASSES = {'mnist': 10, 'svhn': 10, 'cifar-10': 10, 'cifar-100': 100, 'celeb': 20}
dataset = "cifar-10"
init_noise_ratio = 0
data_ratio = 100.0
X_train, y_train, X_test, y_test, un_selected_index = get_data(dataset, init_noise_ratio, data_ratio, random_shuffle=False)


# In[3]:


n_client = 4
malicious_client = sys.argv[2]
if malicious_client == '[]':
    malicious_client = []
else:
    malicious_client = map(float, malicious_client.strip('[]').split(','))
print("noise level, malicious client: ", sys.argv[1], sys.argv[2])
client_data_number = [6000, 6000, 6000, 6000]
clients = []
clients_train_data = []
clients_train_label = []


# In[4]:


cursor = 0
image_shape = X_train.shape[1:]
server =  get_model(dataset, input_tensor=None, input_shape=image_shape, num_classes=NUM_CLASSES[dataset])
# initialize n_client models, and the correspondante training data for each client
# each client has different data size
for i in range(n_client):
    clients.append(get_model(dataset, input_tensor=None, input_shape=image_shape, num_classes=NUM_CLASSES[dataset]))
    clients_train_data.append(X_train[cursor: cursor+client_data_number[i]])
    clients_train_label.append(y_train[cursor: cursor+client_data_number[i]])
    cursor = cursor + client_data_number[i]


# In[5]:


optimizer = SGD(lr=0.01, decay=1e-4, momentum=0.9)
server.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])
for i in range(n_client):
    clients[i].compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])


# In[6]:


datagen = ImageDataGenerator(
        featurewise_center = False,  # set input mean to 0 over the dataset
        samplewise_center = False,  # set each sample mean to 0
        featurewise_std_normalization = False,  # divide inputs by std of the dataset
        samplewise_std_normalization = False,  # divide each input by its std
        zca_whitening = False,  # apply ZCA whitening
        rotation_range = 0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range = 0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range = 0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip = True,  # randomly flip images
        vertical_flip = False,  # randomly flip images
        )
datagen.fit(X_train)


# In[8]:


batch_size = 64
results = []
# 2 is only for quick test, need to set a large number
epochs = 100
for i in range(n_client):
    if i not in malicious_client:
        results.append(
            clients[i].fit_generator(datagen.flow(clients_train_data[i], clients_train_label[i], batch_size=batch_size),
                                    steps_per_epoch=clients_train_data[i].shape[0]//batch_size, epochs=epochs,
                                    validation_data=(X_test, y_test)
                                    )
        )
    else:
        print("malicious client: ", str(i))
        results.append(
            clients[i].fit_generator(datagen.flow(clients_train_data[i], flip_label(clients_train_label[i], float(sys.argv[1])), batch_size=batch_size),
                                    steps_per_epoch=clients_train_data[i].shape[0]//batch_size, epochs=epochs,
                                    validation_data=(X_test, y_test)
                                    )
        )

# In[106]:


def aggregate_weights(client_list):
    n_client = len(client_list)
    n_layers = len(client_list[0].get_weights())
    if n_client == 0:
        print('empty input')
        return
    # copy the weights and structure from last client
    result = client_list[n_client-1].get_weights().copy()
    for k in range(n_client):
        result[k] = result[k]*(1.0/n_client)
    for i in range(n_layers):
        # Using n_client -1: because result contains already last client's weights
        for j in range(n_client-1):
            result[i] = result[i] + client_list[j].get_weights()[i]*(1.0/n_client)
    return result


# In[69]:


weights = aggregate_weights(clients)


# In[77]:


server.set_weights(weights)


# In[100]:


print("Evaluate on test data with all model aggregation")
results = server.evaluate(X_test, y_test, batch_size=batch_size)
print("test loss, test acc:", results)
# accuracy_aggregate is the baseline to calculate influence
# influence is the difference between accurate_aggregatea and accuracy without one of client
accuracy_aggregate = results[1]


print("Conducting Shapley Value")


# In[105]:

def make_all_subsets(n_client):
    client_list = list(np.arange(n_client))
    set_of_all_subsets = set([])
    for i in range(len(client_list),-1,-1):
        for element in itertools.combinations(client_list,i):
            set_of_all_subsets.add(frozenset(element))
    return sorted(set_of_all_subsets)

def calculate_shapley_values(n_client):
        print("*************")
        client_list = list(np.arange(n_client))
        shapley = []
        clientShapley = 0
        total = 0
        factorialTotal = math.factorial(n_client)
        set_of_all_subsets = make_all_subsets(n_client)
        for client in client_list:
            for subset in set_of_all_subsets:
                if client in subset:
                    remainderSet = subset.difference(set([client]))
                    remainder_res = [0,0]
                    sub_weights = aggregate_weights(np.array(clients)[list(subset)])
                    server.set_weights(sub_weights)
                    sub_res = server.evaluate(X_test, y_test, batch_size=batch_size)
                    b = len(remainderSet)
                    factValue = (len(client_list) - b -1)
                    if remainderSet != frozenset():
                        remainder_weights = aggregate_weights(np.array(clients)[list(remainderSet)])
                        server.set_weights(remainder_weights)
                        remainder_res = server.evaluate(X_test, y_test, batch_size=batch_size)
                    difference = sub_res[1] - remainder_res[1]
                    divisor = (math.factorial(factValue) * math.factorial(b) * 1.0) / (factorialTotal * 1.0)
                    weightValue = divisor * difference
                    clientShapley += weightValue
            shapley.append(clientShapley)
            print("Shapley Value of Client " + str(client) + ": " + str(clientShapley))
            total = total + clientShapley
            clientShapley = 0
        print("Shapley Value in list: "+ str(shapley))
        #print("Total: " + str(total) + " *** Note: This should equal to 1.0")
        print("*************")


calculate_shapley_values(n_client)
