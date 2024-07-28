from piece import Piece
import random
import numpy as np
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

    

    def setBoard(self,seed=None):
        self.board = []
        if seed:
            random.seed(seed)
        for row in range(self.size[0]):
            row = []
            for col in range(self.size[1]):
                hasBomb = random.random() < self.prob
                if (not hasBomb):
                    self.numNonBombs += 1
                piece = Piece(hasBomb)
                row.append(piece) #each piece represents a piece on minesweeper game
            self.board.append(row)
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
    
    def getWindow(self,index,window_size=(1,1)):
        """Get all neighbors in a wxh window"""
        width = window_size[0]
        height = window_size[1]
        neighbors = []
        for row in range(index[0] - height, index[0] + height + 1):
            current_row = []
            for col in range(index[1] - width, index[1]+ width + 1):
                outOfBounds = row < 0 or row >= self.size[0] or col < 0 or col >= self.size[1] #check for OOB error
                same = row == index[0] and col == index[1]
                if same:
                    current_row.append(-1) #-1 for a covered cell
                elif outOfBounds:
                    current_row.append(-2) #-2 for OOB
                else:
                    current_row.append(self.getPiece((row,col)).getDisplay())
            neighbors.append(current_row)
        print(f"{width}x{height} Window: {neighbors}")
        return neighbors


    def window_to_one_hot(window):
        """
        Gets a window and turns it into a one hot encoding with on dimensions C X H X W. 10 colors channels for -2, -1 and 0 to 8.
        Args:
            window: 5x5 or any dimensional window with values from -2 to 8. 
            -2 = OOB
            -1 = covered
            0 - 8 = number of mines around
        Returns:
            numpy.ndarray (one-hot encoded window with shape (10,5,5))
        """
        #convert to numpy
        window_np = np.array(window)
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
    
    def getLost(self):
        return self.lost
        
    def getWon(self):
        return self.numNonBombs == self.numClicked
    