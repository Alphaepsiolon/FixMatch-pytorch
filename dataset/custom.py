import glob
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd 
from sklearn.model_selection import train_test_split
from PIL import Image
import cv2 as cv

class DataPreprocess(object):
    def __init__(self,
                 path:str):
        self.path = path
        self.data_df = self.preprocess()

        # create splits
        self.td,self.vd,self.ted = self.create_splits()

    def preprocess(self):
        img_paths = glob.glob(f"{self.path}/*")
        data = {
            'img':[],
            'label_str':[],
            'label':[]
        }
        for imp in img_paths:
            data['img'].append(imp)
            if 'COVID' in imp:
                data['label'].append(0)
                data['label_str'].append('COVID')
            elif 'NORMAL' in imp:
                data['label'].append(1)
                data['label_str'].append('NORMAL')
            elif 'PNEUMONIA' in imp:
                data['label'].append(2)
                data['label_str'].append('PNEUMONIA')
            else:
                print(imp)
                raise ValueError
        
        data_df = pd.DataFrame(data)
        return data_df
    
    def create_splits(self):
        shuf_data = self.data_df.sample(frac=1).reset_index(drop=True)

        # split data
        train_data,test_data = train_test_split(shuf_data, test_size=0.1, random_state=1)

        # only take 20 of each
        df_samples_train = []
        df_samples_val = []
        for l in shuf_data.label_str.unique().tolist():
            subset = train_data[train_data['label_str'] == l]
            temp_train,temp_val = train_test_split(subset, test_size=len(subset)-20,random_state=1)
            df_samples_train.append(temp_train)
            df_samples_val.append(temp_val)

        train_data = pd.concat(df_samples_train)
        val_data = pd.concat(df_samples_val)
        
        return train_data,val_data,test_data
    
    @property
    def train_data(self):
        return self.td
    @property
    def val_data(self):
        return self.vd
    @property
    def test_data(self):
        return self.ted

class ClassificationDataset(Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 transform:transforms.Compose = None,
                 target_transform:transforms.Compose = None) -> None:
        self.data = df
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image = Image.fromarray(cv.imread(self.data['img'].iloc[idx]))
        label = self.data['label'].iloc[idx]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.transform(label)

        return image, label