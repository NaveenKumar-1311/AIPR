
# coding: utf-8

# Objective:
# ========
# To perform object classification(using GTSRB digit data) using federated learning conept. 
# 
# Idea is to mimic and understand how federated learning works, so that It can be further extended to perform more complex computer vision related tasks.
# 
# 
# How:
# ====
# > Use GTSRB data to prepare a feature and label list.
# 
# > Split it into 90%-10% train and test data to judge the performance later.
# 
# > For fedearated clients-
#      1. mimic client as data shards
#      2. essentially we will be creating a dictionary object; where keys: client name and value: map of features data and label
#      
# > Train using a NN based algorithm (each client)
# 
# > Performing Federated average: 
#     https://arxiv.org/pdf/1602.05629.pdf
# `
# 
# > Repeat for few epochs

# In[1]:

import torch
from torchvision import utils as vutils, datasets, transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torch import nn
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
import numpy as np
import pandas as pd
import random
from imutils import paths
from random import randint
import os
from PIL import Image as Image


# In[2]:

dataset_path = '.'


# In[3]:

torch.manual_seed(42)
np.random.seed(42)

cuda3 = torch.device('cuda:7')
# In[4]:



train_path = dataset_path+'/Train'
test_csv_file_path = dataset_path+'/Test.csv'


# In[6]:

NUM_CLASSES = 62
IMG_SIZE = 150
BATCH_SIZE = 64


# In[7]:

data_transforms = transformations = transforms.Compose([
    transforms.Resize((IMG_SIZE,IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


# In[8]:

train_dataset = datasets.ImageFolder(train_path, transform=data_transforms)
num_train = len(train_dataset)
print('number of training samples :', num_train)


# In[ ]:

# test_dataset = GTSRB(root_dir=dataset_path, csv_file=test_csv_file_path, transform= data_transforms)
# print('number of training samples :', len(test_dataset))
train_sampler, test_dataset = torch.utils.data.random_split(train_dataset, (round(0.8*len(train_dataset)), round(0.2*len(train_dataset))))
print('number of testing samples :', len(test_dataset))


# In[ ]:

def randomize_clients_data(train_size, num_clients, minimum_size):


    data_per_client = []
    max = 20000
    for idx in range(num_clients):
      data_per_client.append(randint(minimum_size,max))
    #   max = (train_size-sum(data_per_client))//(num_clients-len(data_per_client))
    # data_per_client.append(train_size-sum(data_per_client))
    return data_per_client

# shard_size = randomize_clients_data(100,4, 20)
# shard_size


# In[ ]:

def create_clients(data, num_clients=10, initial='clients'):
    ''' return: a dictionary with keys clients' names and value as 
                data shards - tuple of images and label lists.
        args: 
            image_list: a list of numpy arrays of training images
            label_list:a list of binarized labels for each image
            num_client: number of fedrated members (clients)
            initials: the clients'name prefix, e.g, clients_1 
            
    '''
    #create a list of client names
    client_names = ['{}_{}'.format(initial, i+1) for i in range(num_clients)]
    training_data_size = len(data)

    train_idx = list(range(training_data_size))
    # np.random.shuffle(train_idx)

    #shard data and place at each client
    # size = len(data)//num_clients
    # shards = [torch.utils.data.Subset(data,list(range(i, i + size))) for i in range(0, size*num_clients, size)]
    # Mimic presence of hetrogenous data size on each client
    shard_size = randomize_clients_data(training_data_size, num_clients, 18000)
    # print('data per client: ', shard_size)
    # shards = [torch.utils.data.Subset(data,list(range(i, i + shard_size[i])))  for i in range(num_clients)]
    shards=[]
    for i in range(num_clients):
        r = random.randint(0, training_data_size - shard_size[i])
        #r = 0
        shards.append(torch.utils.data.Subset(data,train_idx[r : r + shard_size[i]]))
    for i in range(len(shards)):
      print('client ' , i , ' : data size: ', len(shards[i]))

    #number of clients must equal number of shards
    # assert(len(shards) == len(client_names))

    return {client_names[i] : shards[i] for i in range(len(client_names))} 


clients = create_clients(train_sampler, num_clients=1)
print('Clients created..done')


# In[ ]:

def batch_data(data_shard, bs=32):
    '''Takes in a clients data shard and create a tf pytorch dataloaderds object off it
    args:
        shard: a data, label constituting a client's data shard
        bs:batch size
    return:
        pytorch dataloader object'''

    
    trainloader = torch.utils.data.DataLoader(data_shard, batch_size=BATCH_SIZE,
                                            shuffle=False, drop_last= True, num_workers=2)
    
    return trainloader


# In[ ]:

clients_batched = dict()
for (client_name, data) in clients.items():
    clients_batched[client_name] = batch_data(data)
    
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE,
                                            shuffle=False, num_workers=2, drop_last=True)


# ### Model

# In[ ]:

class CustomCNN(nn.Module):

    def __init__(self):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.conv1_bn = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.conv2_bn = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 1)
        self.conv3_bn = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 256, 1)
        self.conv4_bn = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(256 * 8 * 8, 512)
        self.fc1_bn = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 62)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout(self.conv1_bn(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(self.conv2_bn(x))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.conv3_bn(x)
        x = self.pool(F.relu(self.conv4(x)))
        x = self.dropout(self.conv4_bn(x))
        x = x.view(-1, 256 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.dropout(self.fc1_bn(x))
        x = F.softmax(self.fc2(x), -1)
        return x


def scale_model_weights(client_models, weight_multiplier_list):
    '''function for scaling a models weights'''
    client_model = client_models[0].state_dict()
    for i in range(len(client_models)):
      for k in client_model.keys():
        client_models[i].state_dict()[k].float()*weight_multiplier_list[i]

    return client_models


# In[ ]:

def server_aggregate(global_model, client_models):
    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        global_dict[k] = torch.stack([client_models[i].state_dict()[k].float() for i in range(len(client_models))], 0).mean(0)
    global_model.load_state_dict(global_dict)
    return global_model


# In[ ]:

def test(global_model, test_loader):
    """This function test the global model on test data and returns test loss and test accuracy """
    global_model.eval()
    loss = nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    test_size = len(test_loader.dataset)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(cuda3), target.to(cuda3)
            output = global_model(data)
            test_loss += loss(output, target) 
            pred = output.argmax(dim=1, keepdim=True)  
            correct += pred.eq(target.view_as(pred)).sum().item()
            del data
            del target

    test_loss /= test_size
    acc = correct / test_size

    return acc, test_loss

import torch
import numpy as np
import time

def get_probs(model, x, y):
    probs = (model(x))[:, y]
    return torch.diag(probs.data)

# SIMBA
def SimBA(model, x, y, num_iters, epsilon):
    
    x_org = x.clone()
    n_dims = x.reshape(1, -1).size(1)
    perm = torch.randperm(n_dims)
    org_probs = model(x)
    last_prob = get_probs(model, x_org, y)
    new_class = y.clone()
    i = 0
    k = 1

    while ((i < num_iters) and ((y == new_class) or (torch.abs((output)[:, y] - (output)[:, new_class]) <= 0.1))):
        
        diff = torch.zeros(n_dims).to(cuda3)
        diff[perm[i % len(perm)]] = epsilon
        perturbation = diff.reshape(x.size())

        left_prob = get_probs(model, (x - perturbation), y)
        
        if left_prob < last_prob:
            x = (x - perturbation)
            last_prob = left_prob
    
        else:
            right_prob = get_probs(model, (x + perturbation).to(cuda3), y)
            if right_prob < last_prob:
                x = (x + perturbation)
                last_prob = right_prob

        output = model(x)
        new_class = torch.argmax(output)

        i += 1
        
    return x, model(x), i


# In[ ]:

import torch
import numpy as np
import time

def get_probs(model, x, confused_class):
    probs = (model(x))[:, confused_class]
    return torch.diag(probs.data)

def MSimBA(model, x, y, num_iters, epsilon):
    
    x_org = x.clone()
    n_dims = x.reshape(1, -1).size(1)
    perm = torch.randperm(n_dims)
    org_probs = model(x)
    confused_class = torch.topk(org_probs.squeeze(), 2, dim=0, largest=True, sorted=True).indices[1]
    confused_prob = org_probs[0, confused_class]
    last_prob = get_probs(model, x_org, confused_class)
    new_class = y.clone()
    i = 0
    k = 1
    while ((i < num_iters) and ((y == new_class) or (torch.abs((output)[:, y] - (output)[:, new_class]) <= 0.1))):
        
        diff = torch.zeros(n_dims).to(cuda3)
        diff[perm[i % len(perm)]] = epsilon
        perturbation = diff.reshape(x.size())

        left_prob = get_probs(model, (x - perturbation), confused_class)
        
        if left_prob > last_prob:
            x = (x - perturbation)
            last_prob = left_prob
    
        else:
            right_prob = get_probs(model, (x + perturbation).to(cuda3), confused_class)
            if right_prob > last_prob:
                x = (x + perturbation)
                last_prob = right_prob

        output = model(x)
        new_class = torch.argmax(output)

        i += 1
        
    return x, model(x), i


import copy
num = 64  
iterations = []
acc = []

model_path = "./FL_KBTS_3C.pt"


global_model = CustomCNN().to(cuda3)
# global_model.load_state_dict(torch.load(model_path, map_location='cpu'))
# global_model.eval()
# global_acc, global_loss = test(global_model, test_loader)
# print(global_acc)
learning_rate = 0.01 
global_epochs = 200
local_epochs = 1
GTAA=[]

optimizer  = torch.optim.SGD(global_model.parameters(), lr=learning_rate, momentum=0.9)
criterion  = nn.CrossEntropyLoss()
        
for g_epoch in range(global_epochs):

    global_weights = global_model.parameters()
    
    scaled_local_weight_list = list()

    client_names= list(clients_batched.keys())

    client_names_sel = client_names
    print(client_names_sel)
    
    client_models = []
    scaling_factor = []
    clients_data_list = []

    for client in tqdm(client_names_sel):
        local_model = CustomCNN().to(cuda3)
        local_model.load_state_dict(global_model.state_dict())
        local_model = copy.deepcopy(global_model)
        
        optimizer  = torch.optim.SGD(local_model.parameters(), lr=learning_rate, momentum=0.9)
        loss_func  = nn.CrossEntropyLoss()
        for epoch in range(local_epochs):
            success = 0
            total = 0
            client_data_size = 0
            for batch_idx, (data, target) in enumerate(clients_batched[client]):

                images_batch = Variable(data)
                labels_batch = Variable(target)
                # if client == "clients_1":
                #     org_img = images_batch.clone()
                #     org_label = labels_batch.clone()
                #     images_batch = torch.zeros(num, 3, 150, 150)
                #     labels_batch = torch.zeros(num)
                #     for j in range (0, num):
                #         images_batch[j], adv_prob, itera = MSimBA(local_model.to(cuda3), org_img[j].clone().to(cuda3).unsqueeze(0), org_label[j].clone().to(cuda3).unsqueeze(0), 5, 0.7)
                #         labels_batch[j] = torch.argmax(adv_prob)
                #         iterations.append(itera)
                #     client_data_size += len(target)
                # else:
                client_data_size += len(target)
                optimizer.zero_grad()
                output = local_model(images_batch.to(cuda3))
                loss = criterion(output, labels_batch.to(cuda3).long())
                loss.backward()
                optimizer.step()
                del images_batch
                del labels_batch

            client_models.append(local_model)
            clients_data_list.append(client_data_size)
            del local_model
 
    tot_data = sum(clients_data_list)
    scaling_factor = [client_data / tot_data for client_data in clients_data_list]

    client_models = scale_model_weights(client_models, scaling_factor)
    global_model = server_aggregate(global_model,client_models)
    
    global_acc, global_loss = test(global_model, test_loader)
    print('global_epochs: {} | global_loss: {} | global_accuracy: {}'.format(g_epoch+1, global_loss, global_acc))
    GTAA.append(global_acc)

    # np.save('./GTAA_40C1AMSimBAKBTS3112022.npy', GTAA)

print('Federated learning process completed...ok')
print('--------------------------------')



