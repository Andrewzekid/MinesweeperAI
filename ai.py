#file for ai models
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import codecs
import json
import torchvision
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader,DatasetFolder

# ============== Model 1 - CNN Model ==================
class MineSweeperAI(nn.Module):
    def __init__(self,data_folder=None,batch_size=32,shuffle=True):
        super().__init__(self)
        self.gamesPlayed = 0
        self.gamesWon = 0
        self.data_folder = data_folder
        self.batch_size=batch_size
        self.shuffle=shuffle
        self.dataset_folder = None
        if self.data_folder is not None:
            #create dataloader
            self.createDataLoader()

    def getNextMove(moves):
        """Returns the most probable move given the list of next moves"""
        moves_np = np.array(moves)
        return moves_np.argmax(axis=0) #gets the most probable argument
    def loadJson(filepath):
        obj_text = codecs.open(filepath,"r",encoding="utf-8").read()
        b_new = json.loads(obj_text)
        return b_new #return the list loaded
    def getProbability(self,one_hot):
        """Given a one-hot encoding of a 5x5 window, get the probability that the surrounding cell is a mine"""
        return self.forward(one_hot)
    
    def createDataLoader(self):
        self.createDatasetFolder(self.data_folder) #create dataset_folder
        if self.dataset_folder is not None:
            self.data_loader = DataLoader(self.dataset_folder,batch_size=self.batch_size,shuffle=self.shuffle)
        else:
            raise ValueError("Dataset folder was not found")

    def createDatasetFolder(self,data_path):
        data_path = Path(data_path)
        if data_path.is_dir():
            self.dataset_folder = DatasetFolder(
                root = data_path,
                loader=self.loadJson,
                extensions=("json",),
                transform=ToTensor #convert file to tensor
            )
        else:
            raise FileNotFoundError(f"Path {str(data_path)} does not exist!")
    
    def getWinRate(self):
        return self.gamesWon/self.gamesPlayed
    
    def train(self,data):
        """Trains the solver on the examples so far"""

        raise NotImplemented #TODO
    