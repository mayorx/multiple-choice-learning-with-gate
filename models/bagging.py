import torch

import torchvision
import torchvision.transforms as transforms

from random import randint


class BagDataset(torch.utils.data.Dataset):
    def __init__(self, original_dataset):
        self.od = original_dataset
        len = self.od.__len__()
        self.mapping = {}
        for idx in range(len):
            map_idx = randint(0, len-1)
            self.mapping[idx] = map_idx

    def __getitem__(self, index):
        # super(BagDataset, self).__getitem__()
        return self.od.__getitem__(self.mapping[index])


    def __len__(self):
        return self.od.__len__()