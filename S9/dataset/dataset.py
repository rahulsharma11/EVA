from cProfile import label
import importlib
import torchvision.transforms as transforms
import torchvision
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from torchvision.datasets import CIFAR10
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset, DataLoader
import numpy as np
from numpy import asarray



# def get_data_loaders(batch_size, use_cuda):

#     kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

#     transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#     trainset = torchvision.datasets.CIFAR10(root='./dataset', train=True,
#                                             download=True, transform=transform)
#     train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
#                                             shuffle=True, **kwargs)

#     testset = torchvision.datasets.CIFAR10(root='./dataset', train=False,
#                                         download=True, transform=transform)
#     test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
#                                             shuffle=True, **kwargs)

#     classes = ('plane', 'car', 'bird', 'cat',
#             'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
            
#     return train_loader, test_loader, classes



def get_new_data_loaders(batch_size, use_cuda):
        kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

        train_transform = A.Compose(
        [
                A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.02, rotate_limit=10, p=0.5),
                # A.RandomCrop(height=16, width=16),
                A.RandomBrightnessContrast(p=0.5),
                A.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ToTensorV2()
        ]
        )

        test_transform = A.Compose(
        [
                A.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ToTensorV2()  
        ]
        )



        transfroms = lambda x: train_transform(image=np.array(x))['image']
        val_transforms = lambda x: test_transform(image=np.array(x))['image']

        trainset = torchvision.datasets.CIFAR10(root='./dataset', train=True,
                                            download=True, transform=transfroms)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, **kwargs)

        testset = torchvision.datasets.CIFAR10(root='./dataset', train=False,
                                        download=True, transform=val_transforms)
        val_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=True, **kwargs)

        classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        
            
        return train_loader, val_loader, classes