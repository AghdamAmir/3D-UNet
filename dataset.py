import copy
import nibabel as nib
import numpy as np
import os
import tarfile
import json
from sklearn.utils import shuffle
from torch.utils.data import Dataset, DataLoader
import torch
from torch.utils.data import random_split
from config import (
    DATASET_PATH, TASK_ID, TRAIN_VAL_TEST_SPLIT,
    TRAIN_BATCH_SIZE, VAL_BATCH_SIZE, TEST_BATCH_SIZE
)

#Utility function to extract .tar file formats into ./Datasets directory
def ExtractTar(Directory):
        try:
            print("Extracting tar file ...")
            tarfile.open(Directory).extractall('./Datasets')
        except:
            raise "File extraction failed!"
        print("Extraction completed!")
        return 


#The dict representing segmentation tasks along with their IDs
task_names = {
    "01": "BrainTumour",
    "02": "Heart",
    "03": "Liver",
    "04": "Hippocampus",
    "05": "Prostate",
    "06": "Lung",
    "07": "Pancreas",
    "08": "HepaticVessel",
    "09": "Spleen",
    "10": "Colon"
}


class MedicalSegmentationDecathlon(Dataset):
    """
    The base dataset class for Decathlon segmentation tasks
    -- __init__()
    :param task_number -> represent the organ dataset ID (see task_names above for hints)
    :param dir_path -> the dataset directory path to .tar files
    :param transform -> optional - transforms to be applied on each instance
    """
    def __init__(self, task_number, dir_path, split_ratios = [0.8, 0.1, 0.1], transforms = None, mode = None) -> None:
        super(MedicalSegmentationDecathlon, self).__init__()
        #Rectify the task ID representaion
        self.task_number = str(task_number)
        if len(self.task_number) == 1:
            self.task_number = "0" + self.task_number
        #Building the file name according to task ID
        self.file_name = f"Task{self.task_number}_{task_names[self.task_number]}"
        #Extracting .tar file
        if not os.path.exists(os.path.join(os.getcwd(), "Datasets", self.file_name)):
            ExtractTar(os.path.join(dir_path, f"{self.file_name}.tar"))
        #Path to extracted dataset
        self.dir = os.path.join(os.getcwd(), "Datasets", self.file_name)
        #Meta data about the dataset
        self.meta = json.load(open(os.path.join(self.dir, "dataset.json")))
        self.splits = split_ratios
        self.transform = transforms
        #Calculating split number of images
        num_training_imgs =  self.meta["numTraining"]
        train_val_test = [int(x * num_training_imgs) for x in split_ratios]
        if(sum(train_val_test) != num_training_imgs): train_val_test[0] += (num_training_imgs - sum(train_val_test))
        train_val_test = [x for x in train_val_test if x!=0]
        # train_val_test = [(x-1) for x in train_val_test]
        self.mode = mode
        #Spliting dataset
        samples = self.meta["training"]
        shuffle(samples)
        self.train = samples[0:train_val_test[0]]
        self.val = samples[train_val_test[0]:train_val_test[0] + train_val_test[1]]
        self.test = samples[train_val_test[1]:train_val_test[1] + train_val_test[2]]

    def set_mode(self, mode):
        self.mode = mode

    def __len__(self):
        if self.mode == "train":
            return len(self.train)
        elif self.mode == "val":
            return len(self.val)
        elif self.mode == "test":
            return len(self.test)
        return self.meta["numTraining"]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        #Obtaining image name by given index and the mode using meta data
        if self.mode == "train":
            name = self.train[idx]['image'].split('/')[-1]
        elif self.mode == "val":
            name = self.val[idx]['image'].split('/')[-1]
        elif self.mode == "test":
            name = self.test[idx]['image'].split('/')[-1]
        else:
            name = self.meta["training"][idx]['image'].split('/')[-1]
        img_path = os.path.join(self.dir, "imagesTr", name)
        label_path = os.path.join(self.dir, "labelsTr", name)
        img_object = nib.load(img_path)
        label_object = nib.load(label_path)
        img_array = img_object.get_fdata()
        #Converting to channel-first numpy array
        img_array = np.moveaxis(img_array, -1, 0)
        label_array = label_object.get_fdata()
        label_array = np.moveaxis(label_array, -1, 0)
        proccessed_out = {'name': name, 'image': img_array, 'label': label_array} 
        if self.transform:
            if self.mode == "train":
                proccessed_out = self.transform[0](proccessed_out)
            elif self.mode == "val":
                proccessed_out = self.transform[1](proccessed_out)
            elif self.mode == "test":
                proccessed_out = self.transform[2](proccessed_out)
            else:
                proccessed_out = self.transform(proccessed_out)
        
        #The output numpy array is in channel-first format
        return proccessed_out



def get_train_val_test_Dataloaders(train_transforms, val_transforms, test_transforms):
    """
    The utility function to generate splitted train, validation and test dataloaders
    
    Note: all the configs to generate dataloaders in included in "config.py"
    """

    dataset = MedicalSegmentationDecathlon(task_number=TASK_ID, dir_path=DATASET_PATH, split_ratios=TRAIN_VAL_TEST_SPLIT, transforms=[train_transforms, val_transforms, test_transforms])

    #Spliting dataset and building their respective DataLoaders
    train_set, val_set, test_set = copy.deepcopy(dataset), copy.deepcopy(dataset), copy.deepcopy(dataset)
    train_set.set_mode('train')
    val_set.set_mode('val')
    test_set.set_mode('test')
    train_dataloader = DataLoader(dataset= train_set, batch_size= TRAIN_BATCH_SIZE, shuffle= False)
    val_dataloader = DataLoader(dataset= val_set, batch_size= VAL_BATCH_SIZE, shuffle= False)
    test_dataloader = DataLoader(dataset= test_set, batch_size= TEST_BATCH_SIZE, shuffle= False)
    
    return train_dataloader, val_dataloader, test_dataloader
