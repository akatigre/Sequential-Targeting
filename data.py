
import numpy as np
import torch.utils.data
from torch.utils.data import DataLoader, Subset
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
from torchvision import datasets, transforms


class Data(object):
    def __init__(self, batch_size, type = "mnist"):
        self.type = type
        self.batch_size = batch_size
        self.train_dataset, self.test_dataset, self.train_loader, self.valid_loader, self.test_loader, self.trainA_loader, self.validA_loader, self.trainB_loader, self.validB_loader = self.data_balancer(
            # proportion=([0.2] * 6, [1] * 4), A_idx=[400, 800, 900, 1000], B_idx=[4400, 4800, 4900, 5000],
            proportion=([0.1] * 8, [1] * 2), A_idx=[215, 430, 465, 500], B_idx=[4715, 4930, 4965, 5000],
            imbal_class_count=6)
        self.ros_train_loader = self.oversampling(self.train_dataset)
        self.rus_train_loader = self.undersampling(self.train_dataset)
        if self.type=="mnist":
            train_dataset = datasets.MNIST(root='.', train=True, download=True, transform=self.transform)
        else:
            train_dataset = datasets.CIFAR10(root='.', train=True, download=True, transform=self.transform)
        ros_trainA = Subset(train_dataset, self.trainA_indices).dataset
        self.ros_trainA_loader = self.oversampling(ros_trainA)
    def data_loader(self, batch_size):
        if self.type=="mnist":
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))])
            
            train_dataset = datasets.MNIST(root='.', train=True, download=True, transform=self.transform)
            test_dataset  = datasets.MNIST(root='.', train=False, download=True, transform=self.transform)
            class_names = ('zero','one','two','three','four','five','six','seven','eight','nine')
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            train_dataset = datasets.CIFAR10(root='.', train=True, download=True, transform=self.transform)
            test_dataset = datasets.CIFAR10(root='.', train=False, download=True, transform=self.transform)
            class_names = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
                       'ship', 'truck')
            ros_trainA = Subset(train_dataset, self.trainA_indices).dataset
            self.ros_trainA_loader = self.oversampling(ros_trainA)
        
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=torch.cuda.is_available())
        
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=torch.cuda.is_available())

        nb_class = len(class_names)
        return train_dataset, test_dataset, train_loader, test_loader, class_names, nb_class

    def data_balancer(self, proportion=([0.1] * 8, [1] * 2), A_idx=[175,350,400,450],
                      B_idx=[4675,4850, 4900, 4950], imbal_class_count=8):
        train_dataset, test_dataset, train_loader, self.test_loader, self.class_names, self.nb_class = self.data_loader(
            self.batch_size)
        
        train_targets = np.array(train_dataset.targets)
        test_targets = np.array(test_dataset.targets)
        _, train_class_counts = np.unique(train_targets, return_counts=True)
        _, test_class_counts = np.unique(test_targets, return_counts=True)
        
        imbal_class_prop = np.hstack(proportion)

        train_class_indices = [
            np.where(train_targets == i)[0] for i in range(self.nb_class)
        ]
        test_class_indices = [
            np.where(test_targets == i)[0] for i in range(self.nb_class)
        ]

        self.train_class_counts = [
            int(count * prop)
            for count, prop in zip(train_class_counts, imbal_class_prop)
        ]
        self.test_class_counts = [int(min(train_class_counts)*0.2)]*self.nb_class

        train_imbal_class_indices = []
        valid_imbal_class_indices = []
        test_bal_class_indices = []

        for i in range(self.nb_class):
            train_imbal_class_indices.append(
                train_class_indices[i][:int(self.train_class_counts[i] * 0.85)])
            valid_imbal_class_indices.append(
                train_class_indices[i][int(self.train_class_counts[i] * 0.85):self.train_class_counts[i]])
            test_bal_class_indices.append(
                test_class_indices[i][:test_class_counts[i]])

            print('Baseline : Class {} reduced to {} training {} validation {} test'.format(
                self.class_names[i], int(self.train_class_counts[i] * 0.85),
                self.train_class_counts[i] - int(self.train_class_counts[i] * 0.85),self.test_class_counts[i]))

        train_imbal_class_indices = np.hstack(train_imbal_class_indices)
        valid_imbal_class_indices = np.hstack(valid_imbal_class_indices)
        test_bal_class_indices = np.hstack(test_bal_class_indices)

        train_sampler = SubsetRandomSampler(train_imbal_class_indices)
        valid_sampler = SubsetRandomSampler(valid_imbal_class_indices)
        test_sampler = SubsetRandomSampler(test_bal_class_indices)

        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            sampler=train_sampler,
            pin_memory=torch.cuda.is_available())

        valid_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            sampler=valid_sampler,
            pin_memory=torch.cuda.is_available())

        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            sampler=test_sampler,
            pin_memory=torch.cuda.is_available())

        ##########################################################
        ########Get class indices for reduced class count#########
        ##########################################################

        trainA_indices = []
        trainB_indices = []
        valA_indices = []
        valB_indices = []

        for i in range(self.nb_class):

            if i < imbal_class_count:

                trainA_indices.extend(
                    train_class_indices[i][:A_idx[0]])

                trainB_indices.extend(
                    train_class_indices[i][A_idx[0]:A_idx[1]])

                valA_indices.extend(
                    train_class_indices[i][A_idx[1]:A_idx[2]])

                valB_indices.extend(
                    train_class_indices[i][A_idx[2]:A_idx[3]])

            else:
                trainA_indices.extend(
                    train_class_indices[i][:B_idx[0]])
                trainB_indices.extend(
                    train_class_indices[i][B_idx[0]:B_idx[1]])
                valA_indices.extend(
                    train_class_indices[i][B_idx[1]:B_idx[2]])
                valB_indices.extend(
                    train_class_indices[i][B_idx[2]:B_idx[3]])

        self.trainA_indices = np.hstack(trainA_indices)
        self.trainB_indices = np.hstack(trainB_indices)
        valA_indices = np.hstack(valA_indices)
        valB_indices = np.hstack(valB_indices)

        trainA_sampler = SubsetRandomSampler(trainA_indices)
        trainB_sampler = SubsetRandomSampler(trainB_indices)
        valA_sampler = SubsetRandomSampler(valA_indices)
        valB_sampler = SubsetRandomSampler(valB_indices)

        trainA_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            sampler=trainA_sampler,
            pin_memory=torch.cuda.is_available())

        trainB_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            sampler=trainB_sampler,
            pin_memory=torch.cuda.is_available())

        validA_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            sampler=valA_sampler,
            pin_memory=torch.cuda.is_available())

        validB_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            sampler=valB_sampler,
            pin_memory=torch.cuda.is_available())

        return train_dataset, test_dataset, train_loader, valid_loader, test_loader, trainA_loader, validA_loader, trainB_loader, validB_loader

    def oversampling(self, train_dataset):
        weight = np.zeros(self.nb_class)
        class_weights = 1. / torch.tensor(self.train_class_counts, dtype=torch.float)
        target_list = torch.tensor(train_dataset.targets)
        
        target_list = target_list[torch.randperm(len(target_list))]
        class_weights_all = class_weights[target_list]
        weightsampler = WeightedRandomSampler(class_weights_all, num_samples=len(class_weights_all))
        ros_train_loader = DataLoader(
          train_dataset,
          batch_size=self.batch_size,
          sampler = weightsampler,
          num_workers=2,
          pin_memory=torch.cuda.is_available()
          )

        return ros_train_loader

    def undersampling(self, train_dataset):
        weight = np.zeros(self.nb_class)
        class_weights = 1. / torch.tensor(self.train_class_counts, dtype=torch.float)
        target_list = torch.tensor(train_dataset.targets)
        
        target_list = target_list[torch.randperm(len(target_list))]
        class_weights_all = class_weights[target_list]
        weightsampler = WeightedRandomSampler(class_weights_all, num_samples=len(class_weights_all), replacement=False)
        rus_train_loader = DataLoader(
          train_dataset,
          batch_size=self.batch_size,
          sampler = weightsampler,
          num_workers=2,
          pin_memory=torch.cuda.is_available()
          )

        return rus_train_loader

    def _permutate_image_pixels(self, image, permutation):
        if permutation is None:
            return image

        c, h, w = image.size()
        image = image.view(-1, c)
        image = image[permutation, :]
        image.view(c, h, w)
        return image

    def get_dataset(self, name, train=True, download=True, permutation=None):
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



DATASET_CONFIGS = {
    'mnist': {'size': 32, 'channels': 1, 'classes': 10},
    'cifar': {'size': 32, 'channels': 3, 'classes': 10}
}
