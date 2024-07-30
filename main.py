from game import Game
from board import Board
import torch
from ai import MineSweeperAI
size = (8,8)
prob = 0.126 #12.6% is easy, #18.1% is average for intermeditate, 20.6% is average for expert        
screenSize = (800,800)
mode = "ai"
if mode == "ai":
    gamesPlayed = 0
    gamesWon = 0
    maxGames = 100
    #initializer the solver
    device = "cuda" if torch.cuda.is_available() else "cpu"
    solver = MineSweeperAI().to(device)
    global dataloader

    #define optimizer and loss function
    optimizer = torch.optim.SGD(params=solver.parameters(),lr=solver.learning_rate)
    loss_fn = torch.nn.MSELoss()


    def train(model,optimizer,dataloader,loss_fn,verbose=False):
        #main training loop for model
        total_loss = 0
        for batch, (X,y) in enumerate(dataloader):
            X_train,y_train = X.to(device),y.to(device)
            #forward_pass
            y_preds = model(X_train)
            #calculate loss
            loss = loss_fn(y_preds,y_train)
            total_loss += loss
            #optimizer zero grad
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss /= len(dataloader)

        #if verbose, print result
        print("Train MSE Loss: {total_loss}")


    trainable = False
    for game in range(maxGames):
        board = Board(size,prob)
        game = Game(board,screenSize,solver=solver,mode="ai")
        result = game.run(mode="ai")
        gamesPlayed += 1
        gamesWon += result

        if result is None:
            #error
            print(f"Error encountered when playing game {game}, got result None")
            continue
        else:
            if game == 2:
                if solver.data_folder == None:
                    solver.data_folder="data"
                    solver.createDataLoader() #create dataloader
                    dataloader = solver.data_loader
                    traniable = True
                #train loop
            if trainable:
                if gamesPlayed % 10 == 0:
                    print(f"Game {gamesPlayed}: ---------------- \n ")
                    print(f"Win Rate: {(gamesPlayed/gamesWon) * 100}%")
                    train(solver,optimizer,loss_fn,verbose=True)
                else:
                    train(solver,optimizer,loss_fn,verbose=False)

        #train solver on learnt data after every game
        





