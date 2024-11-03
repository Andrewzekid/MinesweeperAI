#file for ai models
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import codecs
import json
import torchvision
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision.transforms import ToTensor
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder
import torch.nn.functional as F
from torch.utils.data import random_split

# ============== Model 1 - Base Model ==================
class MineSweeperAI(nn.Module):
    def __init__(self,blocks,hidden_units,data_folder=None,batch_size=64,learning_rate=0.01,shuffle=True,input_shape=(11,5,5),**kwargs):
        super().__init__()
        self.gamesPlayed = 0
        self.gamesWon = 0
        self.data_folder = data_folder
        self.batch_size=batch_size
        self.shuffle=shuffle
        self.dataset_folder = None
        #key hyperparameters
        self.learning_rate=learning_rate

        print(f"Detected data folder {data_folder}")
        ###Hyperparameters
        self.blocks = blocks
        self.hidden_units = hidden_units
        # print("alive3")
        # assert self.blocks != None, "Recheck input, got self.blocks == None"
        # assert self.hidden_units != None, "Please provide the number of hidden units for the classifier"
        # print("alive4")
        self.network = nn.Sequential()
        # print("Alive1")
        for (in_channels,out_channels,kernel_size,padding,stride) in self.blocks:
          block = self._block(in_channels,out_channels,kernel_size,padding,stride)
          self.network.append(block) #add block to network
          self.network.append(nn.ReLU())
        # print("Alive2")
        # self.classifier = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(in_features=self.hidden_units,out_features=1)
        # )

        # self.network.append(self.classifier)

        self.flatten = nn.Flatten()
        self.linear =  nn.Linear(in_features=self.hidden_units,out_features=1)
        #Append classifier
        self.network.append(
            self.flatten
        )
        self.network.append(self.linear)
        self.network.append(nn.Sigmoid())
        print(self.network)

        #Train test split ratio
        self.train_test_split = 0.8 #80/20 train test split.


    def _block(self,in_channels,out_channels,kernel_size,padding,stride):
      return nn.Conv2d(in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding)


      
      # #      x = self.conv_block_1(x)
      # # print(x.shape)
      # x = self.conv_block_2(x)
      # # print(x.shape)
      # x = self.classifier(x)
      # # print(x.shape)
      # return x

    def forward(self,x):
    #   print(f"Input shape to forward layer: {x.shape}")

      return self.network(x)

    def getNextMove(self,moves):
        """Returns the most probable move given the list of next moves"""
        # print(f"Moves: {moves}")
        moves_np = np.array(moves)
        return moves_np.argmax(axis=0) #gets the most probable to be safe, least likely to be a mine
    def loadJson(self,filepath):
        obj_text = codecs.open(filepath,"r",encoding="utf-8").read()
        b_new = json.loads(obj_text)
        return torch.tensor(b_new,dtype=torch.float32) #return the list loaded
    def getProbability(self,one_hot):
        """Given a one-hot encoding of a 5x5 window, get the probability that the surrounding cell is a mine"""
        with torch.inference_mode():
            return self.forward(one_hot)


    def createDataLoader(self):
        self.createDatasetFolder(self.data_folder) #create dataset_folder

        #Split into training and test dataset
        self.train_dataset,self.test_dataset = random_split(self.dataset_folder,lengths=[self.train_test_split,1-self.train_test_split],generator=torch.Generator().manual_seed(42))
        if self.dataset_folder is not None:
            self.data_loader = DataLoader(self.train_dataset,batch_size=self.batch_size,shuffle=self.shuffle,sampler=self.sampler,pin_memory=True)
            self.test_data_loader = DataLoader(self.test_dataset,batch_size=self.batch_size,shuffle=self.shuffle,pin_memory=True)
        else:
            raise ValueError("Dataset folder was not found")

    def createDatasetFolder(self,data_path):
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        data_path = Path(data_path)
        if data_path.is_dir():
            self.dataset_folder = DatasetFolder(
                root = data_path,
                loader=self.loadJson,
                extensions=("json",),#convert file to tensor
            )
            self.class_weights = [5,1]
            self.sample_weights = [0] * len(self.dataset_folder) #specify exactly the weight for each example in our dataset
            for idx, (data,label) in enumerate(self.train_dataset):
                class_weight = self.class_weights[label]
                self.sample_weights[idx] = class_weight
                #Take out what class weight is for that particular set
            self.sampler = WeightedRandomSampler(self.sample_weights,num_samples=len(self.sample_weights),replacement=True)
        else:
            raise FileNotFoundError(f"Path {str(data_path)} does not exist!")

    def getWinRate(self):
        return self.gamesWon/self.gamesPlayed
    