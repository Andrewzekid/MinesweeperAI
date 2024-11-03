from piece import Piece
import random
import numpy as np
import json,codecs
from pathlib import Path
class Board():
    def __init__(self,size,prob,seed=None):
        self.size = size
        self.prob = prob #probability each piece has a bomb
        self.lost = False
        self.won = False
        self.numClicked = 0
        self.numNonBombs = 0
        self.seed = seed
        self.setBoard(seed=self.seed)

    def setSizes(self,pieceSize,screenSize):
        for row in self.board:
            for piece in row:
                piece.pieceSize = pieceSize
                piece.screenSize = screenSize

    def setBoard(self,seed=None):
        self.board = []
        if seed:
            random.seed(seed)
        for row in range(self.size[0]):
            row_num = row
            hrow = [] #hrow = horiziontal row
            for col in range(self.size[1]):
                hasBomb = random.random() < self.prob
                if (not hasBomb):
                    self.numNonBombs += 1
                piece = Piece(hasBomb)
                piece.position = (row_num,col)
                hrow.append(piece) #each piece represents a piece on minesweeper game
            self.board.append(hrow)
        self.setNeighbors()
    def setNeighbors(self):
        for row in range(self.size[0]):
            for col in range(self.size[1]):
                piece = self.getPiece((row,col))
                neighbors = self.getListOfNeighbors((row,col))
                piece.setNeighbors(neighbors)
    def getListOfNeighbors(self,index):
        neighbors = []
        for row in range(index[0] - 1, index[0] + 2):
            for col in range(index[1] -1, index[1]+2):
                outOfBounds = row < 0 or row >= self.size[0] or col < 0 or col >= self.size[1] #check for OOB error
                same = row == index[0] and col == index[1]
                if (same or outOfBounds):
                    continue
                neighbors.append(self.getPiece((row,col)))
        return neighbors
# ========== START AI FUNCTIONS ================    
    def getWindow(self,index,window_size:tuple=(1,1)):
        """Get all neighbors in a wxh window"""
        # print(f"Window size: {window_size}")
        width = window_size[0]
        height = window_size[1]
        neighbors = []
        for row in range(index[0] - height, index[0] + height + 1):
            current_row = []
            for col in range(index[1] - width, index[1]+ width + 1):
                outOfBounds = row < 0 or row >= self.size[0] or col < 0 or col >= self.size[1] #check for OOB error
                same = row == index[0] and col == index[1]
                if same:
                    current_row.append(9) #-1 for a covered cell
                elif outOfBounds:
                    current_row.append(10) #10 for OOB
                else:
                    current_row.append(self.getPiece((row,col)).getDisplay())
            neighbors.append(current_row)
        # print(f"{width}x{height} Window at {index}: {neighbors}")
        return neighbors


    def window_to_one_hot(self,window):
        """
        Gets a window and turns it into a one hot encoding with on dimensions C X H X W. 10 colors channels for 10, -1 and 0 to 8.
        Args:
            window: 5x5 or any dimensional window with values from 10 to 8. 
            10 = OOB
            -1 = covered
            0 - 8 = number of mines around
        Returns:
            numpy.ndarray (one-hot encoded window with shape (10,5,5))
        """
        #convert to numpy

        window = np.array(window)
        classes = np.arange(11)  # Classes 0 to 10
        
        # Reshape for broadcasting
        window_expanded = window[np.newaxis, :, :]
        classes_expanded = classes[:, np.newaxis, np.newaxis]
        
        # Create one-hot encoding
        one_hot = (window_expanded == classes_expanded).astype(int)
        
        return one_hot
    
    def save_one_hot_as_json(self,one_hot,filename,folder_name="0"):
        ls = one_hot.tolist() #change to nested list (1,11,5,5)
        json_path = Path("data")
        file_path = json_path / folder_name / filename
        
        # #save
        # if not json_path.is_dir():
        #     print(f"Folder {json_path} doesn't exist, creating it....")
        #     json_path.mkdir(parents=True,exist_ok=False) #make the folder if it doesnt exist
        #folder exists
        json.dump(ls, codecs.open(file_path, 'w', encoding='utf-8'), 
          separators=(',', ':'), 
          sort_keys=True, 
          indent=4)
    
    # def save_label_as_json(value,filename,folder_name="labels"):
    #     json_path = Path("data")
    #     file_path = json_path / folder_name / filename

    #     # Save the integer into the JSON file
    #     with file_path.open("w") as file:
    #         json.dump(value, file)
    
    def load_one_hot(filename,folder_name="json"):
        json_path = Path("data") 
        file_path = json_path / folder_name / filename
        if not file_path.is_file():
            raise FileNotFoundError(f"File {filename} not found at {file_path}")
        else:
            obj_text = codecs.open(file_path,"r",encoding="utf-8").read()
            one_hot = json.loads(obj_text)
            return np.array(one_hot)
# ========================== END AI FUNCTIONS =============
    def getSize(self):
        return self.size
    def getPiece(self,index):
        return self.board[index[0]][index[1]]
    def getNumAround(self):
        return self.numAround
    def handleClick(self,piece,flag):
        if (piece.getClicked() or (not flag and piece.getFlagged())):
            return #cant click a piece thats flagged, can only toggle the flag
        if (flag):
            piece.toggleFlag()
            return
        piece.click()
        if (piece.getHasBomb()):
            self.lost = True
            return
        self.numClicked += 1
        if (piece.getNumAround() != 0):
            return
        for neighbor in piece.getNeighbors():
            if (not neighbor.getHasBomb() and not neighbor.getClicked()): #if neighbor doesnt have a bomb and is not clicked, recursively click neighbors
                self.handleClick(neighbor,False) #make a neighbor clicked
    def handleClickIndex(self,piece):
        """Handle click function for AI"""
        if (piece.getClicked()):
            print("already clicked")
            return #cant click a piece thats flagged, can only toggle the flag
        piece.click()
        if (piece.getHasBomb()):
            # print("LOST!!")
            self.lost = True
            return
        self.numClicked += 1
        if (piece.getNumAround() != 0):
            return
        for neighbor in piece.getNeighbors():
            if (not neighbor.getHasBomb() and not neighbor.getClicked()): #if neighbor doesnt have a bomb and is not clicked, recursively click neighbors
                self.handleClickIndex(neighbor) #make a neighbor clicked

    def getLost(self):
        return self.lost
        
    def getWon(self):
        return self.numNonBombs == self.numClicked
    
    def getAvailableMoves(self):
        """Returns a list of all unclicked cells as tuples (colNum,rowNum)"""
        available_moves = []
        size = self.getSize()
        #loop through all rows
        for row in range(size[0]):
            for col in range(size[1]):
                piece = self.getPiece((row,col))
                if not piece.getClicked():
                    # print(f"piece {piece}")
                    available_moves.append((row,col))
        return available_moves
    
                
