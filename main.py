from game import Game
from board import Board
size = (8,8)
prob = 0.126 #12.6% is easy, #18.1% is average for intermeditate, 20.6% is average for expert        
board = Board(size,prob,seed=42)
screenSize = (800,800)
game = Game(board,screenSize)
game.run()

