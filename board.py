class Board():
    def __init__(self,size):
        self.size = size
        self.setBoard()
    def setBoard(self):
        self.board = []
        for row in range(self.size[0]):
            row = []
            for col in range(self.size[1]):
                piece = None
                row.append(piece) #each piece represents a piece on minesweeper game
            self.board.append(row)
    def getSize(self):
        return self.size