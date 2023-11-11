import os
from torch.utils import data
import torch
import pandas as pd
from osgeo import gdal

def readTif(fileName):
    dataset = gdal.Open(fileName)
    if dataset == None:
        print(fileName+"ÎÄ¼þÎÞ·¨´ò¿ª")
        return
    im_width = dataset.RasterXSize 
    im_height = dataset.RasterYSize 
    im_bands = dataset.RasterCount 
    im_data = dataset.ReadAsArray(0,0,im_width,im_height)
    return im_data

class RemoteDataset(data.Dataset):
    def __init__(self, root_dir, split="train", order=0, img_size=128, to_tensor=True):
        super(RemoteDataset, self).__init__()
        self.root_dir = root_dir
        self.to_tensor = to_tensor
        csv_file = os.path.join(root_dir, str(order), f"{split}.csv")
        self.csv_file = pd.read_csv(csv_file, header=None)


    def __getitem__(self, index):
        filename = self.csv_file.iloc[index, 0]
        img_P_path = os.path.join(self.root_dir, "P", f"{filename}")
        img_S_path = os.path.join(self.root_dir, "S", f"{filename}")
        label_img_path = os.path.join(self.root_dir, "gt", f"{filename}")
        img_P = torch.from_numpy(readTif(img_P_path).astype(float)).type(torch.FloatTensor)
        img_S = torch.from_numpy(readTif(img_S_path).astype(float)).type(torch.FloatTensor)
        label = torch.from_numpy(readTif(label_img_path).astype(float)).type(torch.FloatTensor)
        labels = [label]
        return {"name": filename, "P": img_P, "S": img_S, "L": labels[0].long()}

    def __len__(self):
        return len(self.csv_file)
