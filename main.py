from game import Game
from board import Board
import torch
from ai import MineSweeperAI
# from torchsummary import summary
from pathlib import Path
size = (8,8)
import random
prob = 0.7 #12.6% is easy, #18.1% is average for intermeditate, 20.6% is average for expert        
screenSize = (800,800)
mode = "ai-display"
seeds = [random.randint(0,3688880) for i in range(200)]
winning_seeds = []
import time
global trainable 
trainable = False
if mode == "ai":
    gamesPlayed = 0
    gamesWon = 0
    start_games = 3801
    maxGames = 200

    #initializer the solver
    print(f"[INFO] Starting new game!")
#initializer the solver
    device = "cuda" if torch.cuda.is_available() else "cpu"
    solver = MineSweeperAI(learning_rate=0.01,blocks= [(11,22,5,1,1),(22,44,5,1,1)],hidden_units=44).to(device)
    
    #load weights
    model_path=Path("checkpoints") / "CNN_epoch_25_accuracy_0.723.pth"
    solver.load_state_dict(torch.load(model_path,weights_only=True,map_location=torch.device("cpu")))
    board = Board(size,prob)
    game = Game(board,screenSize,solver=solver,mode="ai-display")
    result = game.run(mode="ai-display")
    # with torch.inference_mode():
    #     summary(solver, (11, 5, 5),batch_size=32)

    # #load model
    # modelpath = Path("checkpoints") / "model_3800.pth"
    # solver.load_state_dict(torch.load(modelpath,map_location ='cpu'))
    global dataloader

    #define optimizer and loss function
    optimizer = torch.optim.SGD(params=solver.parameters(),lr=solver.learning_rate)
    loss_fn = torch.nn.MSELoss()


    def train(model,optimizer,dataloader,loss_fn,verbose=False):
        #main training loop for model
        total_loss = 0
        for batch, (X,y) in enumerate(dataloader):
            X_train,y_train = X.to(device),y.to(device).float()
            # print("train x",X_train.dtype)
            # print("ytrain",y_train.dtype)
            #forward_pass
            # print("X train",X_train.shape)
            y_preds = model(X_train)
            #calculate loss
            loss = loss_fn(y_preds,y_train)
            # print("loss fn",loss.dtype)
            total_loss += loss
            #optimizer zero grad
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss /= len(dataloader)

        #if verbose, print result
        print(f"Train MSE Loss: {total_loss}")


    for game_no in range(maxGames):
        seed = seeds[game_no]
        board = Board(size,prob)
        game = Game(board,screenSize,solver=solver,mode="ai")
        result = game.run(mode="ai")
        
        print(f"Starting new game! Game No.{game_no}")
        print(f"Outcome of the game: {result}")
        
        #Check if win
        if result == 1:
            winning_seeds.append(seed)
        # print(f"Trainable: {trainable}")
        # if result is None:
        #     #error
        #     print(f"Error encountered when playing game {game_no}, got result None")
        #     continue
        # else:
        gamesPlayed += 1
        gamesWon += result



        if solver.data_folder == None:
            print("MODEL IS NOW TRAINABLE")
            solver.data_folder="data"
            solver.createDataLoader() #create dataloader
            dataloader = solver.data_loader
            
            #train loop
        if trainable==True:
            if game_no % 5 == 0:
        # print("AAAA")
                print(f"Game {gamesPlayed}: ---------------- \n ")
                print(f"Win Rate: {(gamesWon/gamesPlayed) * 100}%")
                time.sleep(5)
                train(solver,optimizer,solver.data_loader,loss_fn,verbose=True)
            else:
                train(solver,optimizer,solver.data_loader,loss_fn,verbose=False)
            if game_no % 25 == 0:
                # print("BBB")
                print(f"Saving model...")
                checkpoint_path = Path("checkpoints")
                torch.save(solver.state_dict(),checkpoint_path / f"model_{gamesPlayed}_{int(gamesWon/gamesPlayed)*100}.pth")
            
    with open("seeds.txt","w") as seeds_file:
        for seed in winning_seeds:
            seeds_file.write(seed)
        #train solver on learnt data after every game
else:
    print(f"[INFO] Starting new game!")
#initializer the solver
    device = "cuda" if torch.cuda.is_available() else "cpu"
    solver = MineSweeperAI(learning_rate=0.01,blocks= [(11,22,5,1,1),(22,44,5,1,1)],hidden_units=44).to(device)
    
    #load weights
    model_path=Path("checkpoints") / "CNN_epoch_25_accuracy_0.723.pth"
    solver.load_state_dict(torch.load(model_path,weights_only=True,map_location=torch.device("cpu")))
    board = Board(size,prob)
    game = Game(board,screenSize,solver=solver,mode="ai-display")
    #set key constants
    board.pieceSize = game.pieceSize
    board.screenSize = screenSize
    
    result = game.run(mode="ai-display")
    print(f"[INFO] Outcome of the game: {result}")
    

    # print(f"Trainable: {trainable}")
    # if result is None:
    #     #error182343
    #     print(f"Error encountered when playing game {game_no}, got result None")
    #     continue
    # else:
    # gamesPlayed += 1
    # gamesWon += result
        





