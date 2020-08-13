#!/usr/bin/env python3

import torch.utils.data
import wandb
from torch import optim
import utils
from model import Net
from train import train, evaluate
from data import data_balancer

architecture = "CNN"
dataset_id = "cifar-10"
batch_size= 128

config = dict(
  learning_rate = 0.001,
  weight_decay = 0.00001,
  epoch = 100,
  momentum = 0.9,
  architecture = "CNN",
  dataset_id = "cifar-10",
  batch_size= 128,
)


if __name__ == '__main__':

    # decide whether to use cuda or not.
    cuda = torch.cuda.is_available()

    train_dataset, test_dataset, train_loader, valid_loader, test_loader, trainA_loader, validA_loader, trainB_loader, validB_loader = data_balancer()

    print(train_loader.dataset)
    print(trainA_loader.dataset)
    # prepare the model.
    model = Net()

    # initialize the parameters.
    utils.xavier_initialize(model)

    # prepare the cuda if needed.
    # if cuda:
    #     model.cuda()
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model = model.to(device)
    # optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    # print(model)
    # wandb.init(
    #     project='CIFAR!0',
    #     config=config,
    #     name="Baseline")
    #
    # model, train_loss, valid_loss = train(model, train_loader, valid_loader, wandb_log = False, patience =5, n_epochs=10)
    # evaluate(model, test_loader)
    #

    ##############################
    #########Seq Boost############
    ##############################
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Net()
    model = model.to(device)

    for name, param in model.named_parameters():
        if param.device.type != 'cuda':
            print('param {}, not on GPU'.format(name))

    optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=config['momentum'])
    mid_model, train_loss, valid_loss = train(model, trainA_loader, validA_loader, wandb_log = False, patience = 15, consolidate = False, n_epochs=config['epoch'])
    wandb.init(
        project='CIFAR!0',
        config=config,
        name='SeqBoost(EWC) p=2 mu=0.9 eta=4:6')
    model, train_loss, valid_loss = train(mid_model, trainB_loader, validB_loader, patience = 15, consolidate = True, n_epochs=config['epoch'])
    evaluate(model, test_loader)
