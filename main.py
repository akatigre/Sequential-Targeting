#!/usr/bin/env python3

import torch.utils.data
import wandb
from torch import optim
import utils
from model import Net, MLP
from train import train, evaluate
import numpy as np
import os
from data import Data

architecture = "CNN"
dataset_id = "MNIST"
BATCH_SIZE = 64
# SAMPLE_SIZE = 1500
EARLY_STOPPING = 50
N_EPOCHS = 300
PATH = "/content/gdrive/My Drive/SeqBoost-image/"
P = 10
M = 0.8
E = "1:1"


config = dict(
    learning_rate=0.001,
    weight_decay=0.00001,
    epoch=N_EPOCHS,
    momentum=0.9,
    architecture="CNN",
    dataset_id="cifar-10",
    batch_size=BATCH_SIZE,
    # sample_size=SAMPLE_SIZE,
    early_stopping=EARLY_STOPPING
)

if __name__ == '__main__':

    cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    d = Data(BATCH_SIZE)
    train_dataset, test_dataset, train_loader, valid_loader, test_loader,\
    trainA_loader, validA_loader, trainB_loader, validB_loader, ros_train_loader=\
    d.train_dataset, d.test_dataset, d.train_loader, d.valid_loader, d.test_loader, d.trainA_loader, d.validA_loader, d.trainB_loader, d.validB_loader, d.ros_train_loader 

    
    loaders = [train_loader, valid_loader, test_loader, trainA_loader, trainB_loader, validA_loader, validB_loader]
    names = ['train_loader','valid_loader', 'test_loader',"trainA_loader", "trainB_loader", "validA_loader", "validB_loader"]
    for loader, name in zip(loaders, names):
        train_iter = iter(loader)
        for _ in range(2):
            _, target = train_iter.next()
            print(f'{name}', ': Classes {}, counts: {}'.format(
                *np.unique(target.numpy(), return_counts=True)))

   
    #############################
    #########Base Line############
    ##############################
    model = MLP()
    model = model.to(device)
    for name, param in model.named_parameters():
        if param.device.type != 'cuda':
            print('param {}, not on GPU'.format(name))

    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    wandb.init(
        project='Seq Boost2',
        config=config,
        name="Baseline p={} mu={} eta={}".format(P,M,E))

    model, train_loss, valid_loss = train(model, train_loader, valid_loader, batch_size=BATCH_SIZE, wandb_log=True,
                                          consolidate=False, patience=EARLY_STOPPING, n_epochs=config['epoch'])
    evaluate(model, test_loader, batch_size = BATCH_SIZE)
