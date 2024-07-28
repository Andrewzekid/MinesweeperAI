from game import Game
from board import Board
size = (8,8)
prob = 0.156
board = Board(size,prob,seed=42)
screenSize = (800,800)
game = Game(board,screenSize)
game.run()

