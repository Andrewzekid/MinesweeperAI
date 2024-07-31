#file for ai models
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import codecs
import json
import torchvision
from torchvision.transforms import ToTensor
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder

# ============== Model 1 - CNN Model ==================
class MineSweeperAI(nn.Module):
    def __init__(self,data_folder=None,batch_size=32,learning_rate=0.01,shuffle=True,input_shape=(11,5,5)):
        super().__init__()
        self.gamesPlayed = 0
        self.gamesWon = 0
        self.data_folder = data_folder
        self.batch_size=batch_size
        self.shuffle=shuffle
        self.dataset_folder = None
        #key hyperparameters
        self.learning_rate=learning_rate
        self.kernel_size=(5,5)
        print(f"Detected data folder {data_folder}")
        # if self.data_folder is not None:
        #     #create dataloader
        #     self.createDataLoader()
        #     print(f"Created data loader of length {len(self.data_loader)}")


        # #define optimizer
        # self.optimizer = torch.optim.SGD(parameters)
        #Define NN layers
        self.relu = nn.ReLU()
        self.conv25_1 = nn.Conv2d(in_channels=11,out_channels=25,kernel_size=self.kernel_size,padding="same") #conv layer with 25 5x5 filters
        self.conv25_2 = nn.Conv2d(in_channels=25,out_channels=25,kernel_size=self.kernel_size,padding="same")
        self.conv50 = nn.Conv2d(in_channels=25,out_channels=64,kernel_size=self.kernel_size,padding="same")
        self.maxpool_1 = nn.MaxPool2d(kernel_size=(2,2),stride=2) #downsamples to 64 x 2 x2
        self.output_1 = nn.Linear(in_features=256,out_features=1)
        
    
    def forward(self,one_hot):
        """Forward pass for model"""
        layer_1 = self.relu(self.conv25_1(one_hot))
        layer_2 = self.relu(self.conv25_2(layer_1))
        layer_3 = self.relu(self.conv50(layer_2))
        layer_4 = self.maxpool_1(layer_3)
        output = self.output_1(layer_4)
        return output

    def getNextMove(self,moves):
        """Returns the most probable move given the list of next moves"""
        moves_np = np.array(moves)
        return moves_np.argmax(axis=0) #gets the most probable argument
    def loadJson(self,filepath):
        obj_text = codecs.open(filepath,"r",encoding="utf-8").read()
        b_new = json.loads(obj_text)
        return torch.tensor(b_new,dtype=torch.float32) #return the list loaded
    def getProbability(self,one_hot):
        """Given a one-hot encoding of a 5x5 window, get the probability that the surrounding cell is a mine"""
        with torch.inference_mode():
            probs = self.forward(one_hot)
            return probs

    def createDataLoader(self):
        self.createDatasetFolder(self.data_folder) #create dataset_folder
        if self.dataset_folder is not None:
            self.data_loader = DataLoader(self.dataset_folder,batch_size=self.batch_size,shuffle=self.shuffle)

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
        else:
            raise FileNotFoundError(f"Path {str(data_path)} does not exist!")
    
    def getWinRate(self):
        return self.gamesWon/self.gamesPlayed
    
    def train(self,data):
        """Trains the solver on the examples so far"""

        raise NotImplemented #TODO
    