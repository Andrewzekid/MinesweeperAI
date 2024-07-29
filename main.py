from game import Game
from board import Board
from ai import MineSweeperAI
size = (8,8)
prob = 0.126 #12.6% is easy, #18.1% is average for intermeditate, 20.6% is average for expert        
screenSize = (800,800)
gamesPlayed = 0
gamesWon = 0
maxGames = 100
#initializer the solver
solver = MineSweeperAI()

for game in range(maxGames):
    board = Board(size,prob)
    game = Game(board,screenSize,solver=solver,mode="ai")
    result = game.run()
    gamesPlayed += 1
    if result is None:
        #error
        print(f"Error encountered when playing game {game}, got result None")
        continue
    gamesWon += result

