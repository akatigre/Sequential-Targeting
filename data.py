import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms


def data_loader(batch_size = 128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # Training dataset
    train_dataset = datasets.CIFAR10(
        root='.', train=True, download=True, transform=transform)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available())

    # Test dataset
    test_dataset = datasets.CIFAR10(
        root='.', train=False, download=True, transform=transform)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available())

    class_names = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
                   'ship', 'truck')
    nb_classes = len(class_names)

    return train_dataset, test_dataset, train_loader, test_loader, class_names, nb_classes

def data_balancer(proportion = ([0.5] * 9, [1]), batch_size = 128,A_idx = [800,200,2200], B_idx = [1600,4000,4400]):
    train_dataset, test_dataset, train_loader, test_loader, class_names, nb_classes = data_loader()
    train_targets = np.array(train_dataset.targets)
    test_targets = np.array(test_dataset.targets)
    _, train_class_counts = np.unique(train_targets, return_counts=True)
    _, test_class_counts = np.unique(test_targets, return_counts=True)
    targets = np.array(train_dataset.targets)

    imbal_class_prop = np.hstack(proportion)

    train_class_indices = [
        np.where(train_targets == i)[0] for i in range(nb_classes)
    ]
    test_class_indices = [
        np.where(test_targets == i)[0] for i in range(nb_classes)
    ]

    train_class_counts = [
        int(count * prop)
        for count, prop in zip(train_class_counts, imbal_class_prop)
    ]

    test_class_counts = [
        int(count * prop)
        for count, prop in zip(test_class_counts, imbal_class_prop)
    ]

    train_imbal_class_indices = []
    test_imbal_class_indices = []
    for i in range(nb_classes):
        train_class_count = train_class_counts[i]
        train_imbal_class_indices.append(
            train_class_indices[i][:train_class_count])

        test_class_count = test_class_counts[i]
        test_imbal_class_indices.append(test_class_indices[i][:test_class_count])

        print('Class {} reduced to {} training and {} test samples'.format(
            class_names[i], train_class_count, test_class_count))

    train_imbal_class_indices = np.hstack(train_imbal_class_indices)
    test_imbal_class_indices = np.hstack(test_imbal_class_indices)

    # Resample datasets
    train_dataset.train_labels = train_targets[train_imbal_class_indices]
    train_dataset.train_data = train_dataset.data[train_imbal_class_indices]

    test_dataset.test_labels = test_targets[test_imbal_class_indices]
    test_dataset.test_data = test_dataset.data[test_imbal_class_indices]

    assert len(train_dataset.train_labels) == len(train_dataset.train_data)
    assert len(test_dataset.test_labels) == len(test_dataset.test_data)

    train_indices = []
    valid_indices = []

    for i in range(nb_classes):
        thres = int(round(len(train_class_indices[i])) * 0.8)
        train_indices.extend(train_class_indices[i][:thres])
        valid_indices.extend(train_class_indices[i][thres:])

    train_indices = np.hstack(train_indices)
    valid_indices = np.hstack(valid_indices)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        sampler=SubsetRandomSampler(train_indices),
        pin_memory=torch.cuda.is_available())

    valid_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        sampler=SubsetRandomSampler(valid_indices),
        pin_memory=torch.cuda.is_available())

    # Get class indices for reduced class count
    trainA_indices = []
    trainB_indices = []
    valA_indices = []
    valB_indices = []

    # TODO : eta 4:6
    for i in range(nb_classes):
        if i <= 4:

            trainA_indices.extend(
                train_class_indices[i][:A_idx[0]])

            trainB_indices.extend(
                train_class_indices[i][A_idx[0]:A_idx[1]])

            valA_indices.extend(
                train_class_indices[i][A_idx[1]:A_idx[2]])

            valB_indices.extend(
                train_class_indices[i][A_idx[2]:])

        else:
            trainA_indices.extend(
                train_class_indices[i][:B_idx[0]])
            trainB_indices.extend(
                train_class_indices[i][B_idx[0]:B_idx[1]])
            valA_indices.extend(
                train_class_indices[i][B_idx[1]:B_idx[2]])
            valB_indices.extend(
                train_class_indices[i][B_idx[2]:])

    trainA_indices = np.hstack(trainA_indices)
    trainB_indices = np.hstack(trainB_indices)
    valA_indices = np.hstack(valA_indices)
    valB_indices = np.hstack(valB_indices)

    trainA_sampler = SubsetRandomSampler(trainA_indices)
    trainB_sampler = SubsetRandomSampler(trainB_indices)
    valA_sampler = SubsetRandomSampler(valA_indices)
    valB_sampler = SubsetRandomSampler(valB_indices)


    trainA_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        sampler=trainA_sampler,
        pin_memory=torch.cuda.is_available())

    trainB_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        sampler=trainB_sampler,
        pin_memory=torch.cuda.is_available())

    validA_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        sampler=valA_sampler,
        pin_memory=torch.cuda.is_available())

    validB_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        sampler=valB_sampler,
        pin_memory=torch.cuda.is_available())

    return train_dataset, test_dataset, train_loader, valid_loader, test_loader, trainA_loader, validA_loader, trainB_loader, validB_loader


def _permutate_image_pixels(image, permutation):
    if permutation is None:
        return image

    c, h, w = image.size()
    image = image.view(-1, c)
    image = image[permutation, :]
    image.view(c, h, w)
    return image


def get_dataset(name, train=True, download=True, permutation=None):
    dataset_class = AVAILABLE_DATASETS[name]
    dataset_transform = transforms.Compose([
        *AVAILABLE_TRANSFORMS[name],
        transforms.Lambda(lambda x: _permutate_image_pixels(x, permutation)),
    ])

    return dataset_class(
        './datasets/{name}'.format(name=name), train=train,
        download=download, transform=dataset_transform,
    )


AVAILABLE_DATASETS = {
    'mnist': datasets.MNIST,
    'cifar': datasets.CIFAR10
}

AVAILABLE_TRANSFORMS = {
    'mnist': [
        transforms.ToTensor(),
        transforms.ToPILImage(),
        transforms.Pad(2),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ],
    'cifar': [
        transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                            ])
    ]
}

DATASET_CONFIGS = {
    'mnist': {'size': 32, 'channels': 1, 'classes': 10},
    'cifar': {'size': 32, 'channels': 3, 'classes': 10}
}
