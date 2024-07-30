import pygame
import os
import time
import string
import torch
import random
import numpy as np
class Game():
    def __init__(self,board,screenSize,window_size=(2,2),solver=None,mode="human"):
        self.board = board
        self.window_size = window_size
        self.screenSize = screenSize
        self.pieceSize = self.screenSize[0] // self.board.getSize()[1], self.screenSize[1] // self.board.getSize()[0] #no of pixels in x direction divided by no columns
        self.modes = ["human","ai","ai-realistic"]
        self.mode = mode
        self.solver= solver
        if mode != "ai":
            #on ai mode, optimizer for training
            self.loadImages()
        
    def run(self,mode):

        running = True
        if self.mode == "ai":
            while running:
                #print out all available moves
                available_moves = self.board.getAvailableMoves()
                # print(f"Available moves: {available_moves}")
                # print(available_moves)
                #get windows for those available moves (OHvector,x,y)
                windows = [self.board.getWindow(index,self.window_size) for index in available_moves]
                #one hot encoding
                # print(windows[0])
                one_hot = [self.board.window_to_one_hot(window) for window in windows]
                # one_hot_with_coords = zip(one_hot,available_moves) #(OHvector, (rownum,colnum))

                #Run all coords through ai
                probabilities = [] #(0.86, (X,y))
                for one_hot_vec in one_hot:
                    # print(one_hot_vec)
                    probability = self.solver.getProbability(torch.tensor(one_hot_vec).float())
                    probabilities.append(probability)
                probabilities_with_coords = zip(probabilities,available_moves) #(Probability, (rownum,colnum))

                #get next move
                # print(f"probabilities: {probabilities}")
                safest_cell = self.solver.getNextMove(probabilities) #get index of safest move (move with safe probability closest to 1)
                next_move = available_moves[safest_cell] #get coordinates of next move
                # print(f"Selected move: {next_move}")
                #click on next move
                #save window
                self.handleClickIndex(next_move)
                print(f"Clicked on {next_move}!")
                # print(self.board.numClicked)
                # if self.board.numClicked > 0:
                #prevent saving noise to the dataset e.g: first move blunders
                uuid = self.generateFileUUID() + ".json"
    
                result = 0
                # print("handling result!")
                #Check result
                if not self.board.getLost():
                    result = 1 #safe cell
                    # print(f"One hot encoding for cell: {one_hot[safest_cell]}")
                    self.board.save_one_hot_as_json(one_hot=np.array(one_hot[safest_cell]).astype(float),filename=uuid,folder_name="1") #save in 1's folder if safe
                    # print("Safe!")
                    if self.board.getWon():
                        print("Game won!")
                        running = False
                        return result #return a 1 for a win
                else:
                    print("game lost!")
                    self.board.save_one_hot_as_json(one_hot=np.array(one_hot[safest_cell]).astype(float),filename=uuid,folder_name="0") #save in 0s folder if unsafe
                    running = False
                    return result #return a 0 for a loss
                    
                   
                    # #save result as json
                    # self.board.save_label_as_json(result,filename=uuid)
            return #in the end that the code is glitched, return None
        else:
            pygame.init()
            self.screen = pygame.display.set_mode(self.screenSize)
            while running:
                #listen for events
                for event in pygame.event.get():
                    if (event.type == pygame.QUIT):
                        running = False
                    if (event.type == pygame.MOUSEBUTTONDOWN):
                        position = pygame.mouse.get_pos()
                        rightClick = pygame.mouse.get_pressed()[2]
                        self.handleClick(position, rightClick)
                self.draw()
                pygame.display.flip()
                if (self.board.getWon()):
                    #won
                    sound = pygame.mixer.Sound("win.wav")
                    sound.play()
                    time.sleep(3)
                    running = False
            pygame.quit()
    
    def draw(self):
        topLeft = (0,0)
        for row in range(self.board.getSize()[0]):
            for col in range(self.board.getSize()[1]):
                piece = self.board.getPiece((row,col))
                image = self.getImage(piece)
                self.screen.blit(image,topLeft)
                topLeft = topLeft[0] + self.pieceSize[0], topLeft[1]
            topLeft = 0, topLeft[1] + self.pieceSize[1]

    def loadImages(self):
        self.images = {} #dictionary with mapping of each image to image object
        for fileName in os.listdir("images"):
            if (not fileName.endswith(".png")):
                continue
            image = pygame.image.load(r"images/"+fileName)
            image = pygame.transform.scale(image,self.pieceSize)
            self.images[fileName.split(".")[0]] = image #only the part before .png
    def getImage(self,piece):
        string = None
        if (piece.getClicked()):
            string = "bomb-at-clicked-block" if piece.getHasBomb() else str(piece.getNumAround())
        else:
            string = "flag" if piece.getFlagged() else "empty-block"
        # if piece.getHasBomb() else str(piece.getNumAround())
        return self.images[string]

    def handleClick(self,position,rightClick):
        if (self.board.getLost()):
            return #if we lost, then dont let us click
        index = position[1] // self.pieceSize[1], position[0] // self.pieceSize[0] #integer division by number of pixels in y direction #grab y position of mouse click divided by the pieceSize will yield index on board
        piece = self.board.getPiece(index)
        self.board.getWindow(index,window_size=(2,2))
        self.board.handleClick(piece,rightClick)

    def handleClickIndex(self,index):
        #function to handle ai clicks
        if (self.board.getLost()):
            return #if we lost, then dont let us click
        piece = self.board.getPiece(index)
        self.board.handleClickIndex(piece)
    
    def generateFileUUID(self,length=16):
        chars = string.ascii_letters + string.digits
        return ''.join(random.choices(chars,k=length))
