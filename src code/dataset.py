import torch
import os
import numpy as np
import csv
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


default_transform = transforms.Compose([
    transforms.ToTensor(),
    ])

class bair_robot_pushing_dataset(Dataset):
    def __init__(self, args,mode='train', transform=default_transform):
        assert mode == 'train' or mode == 'test' or mode == 'validate'
        self.mode=mode
        self.root=args.data_root
        
        self.seed_is_set=False
        if self.mode=='train':
            self.dirPath=f'{self.root}/train'
            
            pass
        elif self.mode=='test':
            self.dirPath=f'{self.root}/test/traj_0_to_255.tfrecords'
            pass
        else:
            self.dirPath=f'{self.root}/validate/traj_256_to_511.tfrecords'
        
        self.dirList=[d for d in os.listdir(self.dirPath) if os.path.isdir((os.path.join(self.dirPath,d)))]
        #print(self.dirList)
        
        self.transform=transform

        self.n_per_seq=args.n_past+args.n_future

        #raise NotImplementedError
        
    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)
            
    def __len__(self):
        if self.mode=='train':
            return len(self.dirList)*256
        else:
            return len(self.dirList)
        
        
    def get_seq(self,filepath):
        seq=torch.empty(0,3,64,64)
        for i in range(self.n_per_seq):
            img=Image.open(f'{filepath}/{i}.png')
            img=torch.unsqueeze(self.transform(img),0)
            seq=torch.cat((seq,img), 0)
        # return [2+10,3,64,64]
        return seq
    
    def get_csv(self,filepath):
        # [30,4]
        action = torch.from_numpy(
            np.genfromtxt(f'{filepath}/actions.csv', delimiter=",")
        )
        # [30,3]
        endeffector_positions =torch.from_numpy(
            np.genfromtxt(f'{filepath}/endeffector_positions.csv', delimiter=",")
        )
        # [30,7]
        cond=torch.cat((action,endeffector_positions),1)[:self.n_per_seq,:]
        
        return cond
    
    def __getitem__(self, index):
        self.set_seed(index)
        if self.mode=='train':
            filepath=f'{self.dirPath}/{self.dirList[index//256]}/{index%256}'
        else:
            filepath=f'{self.dirPath}/{index}'
            pass
        seq = self.get_seq(filepath)
        cond =  self.get_csv(filepath)
        return seq, cond
