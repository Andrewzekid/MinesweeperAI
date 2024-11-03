import pygame

class Piece():
    def __init__(self,hasBomb):
        self.hasBomb = hasBomb
        self.clicked = False
        self.flagged = False
        self.pieceSize = None
        self.screenSize = None
        self.position = None
    def getHasBomb(self):
        return self.hasBomb
    
    def getClicked(self):
        return self.clicked
    
    def getFlagged(self):
        return self.flagged
    
    def setNeighbors(self,neighbors):
        self.neighbors = neighbors
        self.setNumAround()
    
    def setNumAround(self):
        self.numAround = 0
        for piece in self.neighbors:
            if (piece.getHasBomb()):
                self.numAround += 1
    @staticmethod
    def get_window_position():
        pass
    
    @property
    def window_coordinates(self):
        """Returns the coordinates of a piece in tuples, (left,top,right,bottom)"""
        left,top,right,bottom = self.get_window_position()
        window_tl = (left,top)
        window_br = (right,bottom)

        #Calculate the navbar size
        navbar_size = (window_br[1] - window_tl[1]) - self.screenSize[1]
        return self.position[0]+left, top + navbar_size + self.position[1],self.position[0]+left + self.pieceSize[0],top + navbar_size + self.position[1] + self.pieceSize[1]
    def getNumAround(self):
        return self.numAround
    def toggleFlag(self):
        self.flagged = not self.flagged
    def click(self):
        self.clicked = True
    def getNeighbors(self):
        return self.neighbors
    def getDisplay(self):
        """Returns what the bot is able to see on that cell, depending on if the cell has been clicked or not.
        If covered, returns -1, else, return the number of mines around itself"""
        if self.clicked:
            return self.getNumAround()
        else:
            return 9 #-1 for a covered cell
    