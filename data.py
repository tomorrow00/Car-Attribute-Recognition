import os
from os.path import exists, join, basename
from os import makedirs, remove
from six.moves import urllib
import tarfile
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize
from torchvision.transforms import Normalize

from dataset import DatasetFromFolder

data_dir = "/home/wcc/data/car_attribute"
root_dir = os.path.join(data_dir, "processedImages")

def get_training_set():
    train_dir = join(root_dir, "train")
    crop_size = (224, 224)
    # return DatasetFromFolder(train_dir,
    #                          input_transform=Compose([CenterCrop(crop_size), Resize(crop_size), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406],
    #                      std=[0.229, 0.224, 0.225]),]))
    return DatasetFromFolder(train_dir,
                             input_transform=Compose([Resize(crop_size), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])]))

def get_val_set():
    val_dir = join(root_dir, "validation")
    crop_size = (224, 224)
    # return DatasetFromFolder(val_dir,
    #                          input_transform=Compose([CenterCrop(crop_size), Resize(crop_size), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406],
    #                      std=[0.229, 0.224, 0.225]),]))
    return DatasetFromFolder(val_dir,
                             input_transform=Compose([Resize(crop_size), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])]))

def get_test_set():
    test_dir = join(root_dir, "test")
    crop_size = (224, 224)
    # return DatasetFromFolder(test_dir,
    #                          input_transform=Compose([CenterCrop(crop_size), Resize(crop_size), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406],
    #                      std=[0.229, 0.224, 0.225]),]))
    return DatasetFromFolder(test_dir,
                             input_transform=Compose([Resize(crop_size), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])]))
                      
