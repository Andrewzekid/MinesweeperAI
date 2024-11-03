

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

class MineSweeperAI(nn.Module):
    def __init__(self, blocks, hidden_units, data_folder=None, batch_size=64, 
                 learning_rate=0.01, shuffle=True, input_shape=(11,5,5), **kwargs):
        super().__init__()
        # Game statistics
        self.gamesPlayed = 0
        self.gamesWon = 0
        
        # Data loading parameters
        self.data_folder = data_folder
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dataset_folder = None
        self.train_test_split = 0.8
        self.train_dataset = None
        self.test_dataset = None
        self.data_loader = None
        self.test_data_loader = None
        
        # Training parameters
        self.learning_rate = learning_rate
        self.blocks = blocks
        self.hidden_units = hidden_units
        
        print(f"Detected data folder {data_folder}")
        
        # Build the network
        self.network = nn.Sequential()
        for (in_channels, out_channels, kernel_size, padding, stride) in self.blocks:
            block = self._block(in_channels, out_channels, kernel_size, padding, stride)
            self.network.append(block)
            self.network.append(nn.ReLU())
        
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=self.hidden_units, out_features=1)
        
        self.network.append(self.flatten)
        self.network.append(self.linear)
        self.network.append(nn.Sigmoid())
        print(self.network)

    def _block(self, in_channels, out_channels, kernel_size, padding, stride):
        return nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )

    def forward(self, x):
        return self.network(x)

    def getNextMove(self, moves):
        """Returns the most probable move given the list of next moves"""
        moves_np = np.array(moves)
        return moves_np.argmax(axis=0)

    def loadJson(self, filepath):
        """Load and decode JSON file into tensor"""
        print(f"Now decoding: {filepath}")
        with codecs.open(filepath, "r", encoding="utf-8") as f:
            obj_text = f.read()
        b_new = json.loads(obj_text)
        return torch.tensor(b_new, dtype=torch.float32)

    def getProbability(self, one_hot):
        """Given a one-hot encoding of a 5x5 window, get the probability that the surrounding cell is a mine"""
        with torch.inference_mode():
            return self.forward(one_hot)

    def createDatasetFolder(self, data_path):
        """Create and initialize the dataset folder with proper weights"""
        data_path = Path(data_path)
        if not data_path.is_dir():
            raise FileNotFoundError(f"Path {str(data_path)} does not exist!")
        
        self.dataset_folder = DatasetFolder(
            root=data_path,
            loader=self.loadJson,
            extensions=("json",)
        )
        
        # Initialize weights for balancing classes
        self.class_weights = [5, 1]  # Higher weight for mines
        num_samples = len(self.dataset_folder)
        self.sample_weights = []
        
        # Calculate weights for each sample
        for _, label in self.dataset_folder:
            class_weight = self.class_weights[label]
            self.sample_weights.append(class_weight)
            
        # Create weighted sampler
        self.sampler = WeightedRandomSampler(
            weights=self.sample_weights,
            num_samples=num_samples,
            replacement=True
        )

    def createDataLoader(self):
        """Create train and test data loaders with proper splitting"""
        if self.dataset_folder is None:
            self.createDatasetFolder(self.data_folder)
        
        # Calculate exact split sizes
        total_size = len(self.dataset_folder)
        train_size = int(total_size * self.train_test_split)
        test_size = total_size - train_size
        
        # Create the splits
        self.train_dataset, self.test_dataset = random_split(
            self.dataset_folder,
            lengths=[train_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Create the data loaders
        self.data_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # False because we're using sampler
            sampler=self.sampler,
            pin_memory=True
        )
        
        self.test_data_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            pin_memory=True
        )

    def getWinRate(self):
        """Calculate the win rate"""
        if self.gamesPlayed == 0:
            return 0.0
        return self.gamesWon/self.gamesPlayed