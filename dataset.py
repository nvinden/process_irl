import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

import torchvision

from PIL import Image

from saliency import dataset

DATASET_CONFIG = {
	'data_path' : "Datasets",
	'dataset_json': 'saliency/data/dataset.json',
	'auto_download' : True
}


class ProcessDataDataset(Dataset):
    #seq, stim, img_emb, seq_patch
    def __init__(self, dataset_name):
        self.data_path =  "Dataset_" + dataset_name

        self.dataset_name = dataset_name

        self.height = 320
        self.width = 512

        self.ds = dataset.SaliencyDataset(config = DATASET_CONFIG)
        self.ds.load(self.dataset_name)

        self.stimuli_directory = os.path.join(self.data_path, "stimuli")
        self.blurred_stimuli_directory = os.path.join(self.data_path, "blurred_stimuli")

        if not os.path.isdir(self.data_path):
            os.mkdir(self.data_path)
        if not os.path.isdir(self.stimuli_directory):
            os.mkdir(self.stimuli_directory)
        if not os.path.isdir(self.blurred_stimuli_directory):
            os.mkdir(self.blurred_stimuli_directory)

    def __len__(self):
        sample_length = len(self.ds.get('stimuli_path'))
        return sample_length

    def __getitem__(self, idx):
        filename = "pathformer_" + str(idx).rjust(5, '0') + ".npy"
        
        stimuli_file_name = os.path.join(self.stimuli_directory, filename)
        if os.path.isfile(stimuli_file_name):
            stimuli = np.load(stimuli_file_name, allow_pickle = True)
        else:
            stimuli = self.save_stimuli(stimuli_file_name, idx)

        blurred_stimuli_file_name = os.path.join(self.blurred_stimuli_directory, filename)
        if os.path.isfile(blurred_stimuli_file_name):
            blurred_stimuli = np.load(blurred_stimuli_file_name, allow_pickle = True)
        else:
            blurred_stimuli = self.save_blurred_stimuli(blurred_stimuli_file_name, idx)

        if not torch.is_tensor(stimuli):
            stimuli = torch.from_numpy(stimuli)
        if not torch.is_tensor(blurred_stimuli):
            blurred_stimuli = torch.from_numpy(blurred_stimuli)

        return {"stimuli": stimuli, "blurred_stimuli": blurred_stimuli}

    def save_stimuli(self, filename, idx, save = True):
        stim = self.ds.get("stimuli", index = range(idx, idx + 1), start = idx)
        stim = np.squeeze(stim, axis = 0)

        stim = torch.from_numpy(stim)
        stim = stim.permute(2, 0, 1)
        resize = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size = (self.height, self.width)),
        ])
        stim = resize(stim)
        stim = stim.permute(1, 2, 0)

        if save == True:
            np.save(filename, stim)

        return stim

    def save_blurred_stimuli(self, filename, idx, save = True):
        stim = self.ds.get("stimuli", index = range(idx, idx + 1), start = idx)
        stim = np.squeeze(stim, axis = 0)

        stim = torch.from_numpy(stim)
        stim = stim.permute(2, 0, 1)
        resize_and_blur = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size = (self.height, self.width)),
            torchvision.transforms.GaussianBlur(kernel_size = 7, sigma = 2)
        ])
        stim = resize_and_blur(stim)
        stim = stim.permute(1, 2, 0)

        if save == True:
            np.save(filename, stim)

        return stim
