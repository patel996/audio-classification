from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transforms import tensor_transform
import os
import torchaudio
import csv
from os import path
from pathlib import Path
import torch

class CustomDataset(Dataset):
    def __init__(self, dataset_path, transform=tensor_transform):

        self.dataset = []
        self.male = 0
        self.female = 0
        self.kid = 0
        
        walker = sorted(str(p) for p in Path(dataset_path).glob(f'*.wav'))
        for i, file_path in enumerate(walker):
            path,filename = os.path.split(file_path)
            speaker, _ = os.path.splitext(filename)
            _, label = os.path.split(dataset_path)
            
            
            # Load audio
            waveform, sample_rate = torchaudio.load(file_path)
        
            if ((label == 'Child speech, kid speaking' and self.kid < 3500) or (label == 'Male speech, man speaking' and self.male < 1750) or (label == 'Female speech, woman speaking' and self.female < 1750)):
                if(waveform.size()[1] == 160000):
                    if label == 'Child speech, kid speaking':
                        self.kid += 1
                        x = 0
                    elif label == 'Female speech, woman speaking':
                        self.female += 1
                        x = 1
                    elif label == 'Male speech, man speaking':
                        self.male += 1
                        x = 1
            
                    if waveform.size()[0] == 1:
                        (self.dataset).append((waveform, x))
                    else:
                        waveform = torch.mean(waveform, dim=0).unsqueeze(0)
                        (self.dataset).append((waveform, x))
                
                
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        (img, label_id) = self.dataset[idx]
        #fname = self.path_dataset[idx][0]
        transformed = self.transform(img)
        #transformed = torch.squeeze(transformed, 0)
        return transformed, label_id

def load_data(dataset_path, transforms=tensor_transform, num_workers=0, batch_size=32, random_seed = 40):
    '''
    this data loading proccedure assumes dataset/train/ dataset/val/ folders
    also assumes transform dictionary with train and val
    '''
    dataset_train_path_child = dataset_path + "/Child speech, kid speaking"
    dataset_train_child = CustomDataset(dataset_train_path_child, transform=transforms['train'])
    
    dataset_train_path_male = dataset_path + "/Male speech, man speaking"
    dataset_train_male = CustomDataset(dataset_train_path_male, transform=transforms['train'])
    
    dataset_train_path_female = dataset_path + "/Female speech, woman speaking"
    dataset_train_female = CustomDataset(dataset_train_path_female, transform=transforms['train'])
    
    
    #print(dataset_train_child[0])
    #dataset_val_path = dataset_path + "/val"
    #dataset_val = Dataset(dataset_val_path, transform=transforms['valid'])
    train_child_dataset, test_child_dataset = torch.utils.data.random_split(dataset_train_child, [3131, 348])
    train_male_dataset, test_male_dataset = torch.utils.data.random_split(dataset_train_male, [1575, 175])
    train_female_dataset, test_female_dataset = torch.utils.data.random_split(dataset_train_female, [1575, 175])
    
    train_dataset = train_child_dataset + train_male_dataset + train_female_dataset
    validation_dataset = test_child_dataset + test_male_dataset + test_female_dataset
    
    print(len(train_child_dataset))
    print(len(train_male_dataset))
    print(len(test_female_dataset))

    print("Size of train dataset: ",len(train_dataset))
    #print("Size of val dataset: ",len(dataset_val))

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True),
        'val': DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    }
    return dataloaders

# def load_data(dataset_path, transforms=tensor_transform, num_workers=0, batch_size=32, random_seed = 40):
#     return 2