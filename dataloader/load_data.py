import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.utils as utils
from codewrapper.dataloader.albumentation import *
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import notebook
from PIL import Image
import os
import requests
import zipfile
from io import BytesIO
import glob
import csv
import numpy as np
import random


class Cifar10DataLoader:
    def __init__(self, config): #get the config from the config.yaml file
        self.config = config
        self.augmentation = config['data_augmentation']['type']
        
    def calculate_mean_std(self):
        train_transform = transforms.Compose([transforms.ToTensor()])
        train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
        mean = train_set.data.mean(axis=(0,1,2))/255
        std = train_set.data.std(axis=(0,1,2))/255
        return mean, std

    def classes(self):
        return ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'], None
        
    def get_dataloader(self): 
        
        cifar_albumentation = eval(self.augmentation)()
        mean,std = self.calculate_mean_std()
        
        train_transforms, test_transforms = cifar_albumentation.train_transform(mean,std),cifar_albumentation.test_transform(mean,std)
                                                                              
        trainset = datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=train_transforms)  
            
        testset  = datasets.CIFAR10(root='./data', train=False,
                                             transform=test_transforms)

        self.train_loader = torch.utils.data.DataLoader(trainset, 
                                                      batch_size=self.config['data_loader']['args']['batch_size'], 
                                                      shuffle=True,
                                                      num_workers=self.config['data_loader']['args']['num_workers'], 
                                                      pin_memory=self.config['data_loader']['args']['pin_memory'])
        self.test_loader = torch.utils.data.DataLoader(testset, 
                                                     batch_size=self.config['data_loader']['args']['batch_size'],  
                                                     shuffle=False,
                                                     num_workers=self.config['data_loader']['args']['num_workers'], 
                                                     pin_memory=self.config['data_loader']['args']['pin_memory'])
        return self.train_loader,self.test_loader