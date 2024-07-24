import pygame

class Game():
    def __init__(self,board,screenSize):
        self.board = board
        self.pieceSize = self.screenSize[0] // self.board.getSize()[1], self.screenSize[1] // self.board.getSize()[0] #no of pixels in x direction divided by no columns
        self.loadImages
        self.screenSize = screenSize
    def run(self):
        pygame.init()
        screen = pygame.display.set_mode(self.screenSize)
        running = True
        while running:
            #listen for events
            for event in pygame.event.get():
                if (event.type == pygame.QUIT):
                    running = False
                    self.draw()
            pygame.display.flip()
        pygame.quit()
    
    def draw(self):
        topLeft = (0,0)
        for row in range(self.board.getSize()[0]):
            for col in range(self.board.getSize()[1]):
                pygame.draw()

    def loadImages(self):
        self.images = {} #dictionary with mapping of each image to image object
        for fileName in os.listdir("images"):
            if (not fileName.endswith(".png")):
                continue
            image = pygame.Image.load(r"images/"+fileName)
            image = pygame.transform.scale(image,self.pieceSize)
            self.images[fileName.split(".")[0]] = image #only the part before .png

