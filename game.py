import numpy as np
from copy import deepcopy
from collections import deque
import time
import random

import pygame.display
from tqdm import tqdm
import os
import cairo
import math
from PIL import Image
import cv2
filePath = "C://Users//arif1//PycharmProjects//deepqlearning//inputs.txt"
class GameBoard():
    WIDTH = HEIGHT = 224
    DIMENSION = 7
    SQUARE_SIZE = WIDTH//DIMENSION
    RED = (255,0,0)
    WHITE = (255,255,255)
    BLACK = (0,0,0)
    BLUE = (0, 0, 255)

    sCount = 4
    cCount = 4
    turn = 1
    roundNum = 1
    def __init__(self):
        self.board = [
            ["E", "E", "E", "E", "E", "E", "E", "E", "E"],
            ["E", "S", "-", "-", "-", "-", "-", "C", "E"],
            ["E", "-", "-", "-", "-", "-", "-", "-", "E"],
            ["E", "S", "-", "-", "-", "-", "-", "C", "E"],
            ["E", "-", "-", "-", "-", "-", "-", "-", "E"],
            ["E", "C", "-", "-", "-", "-", "-", "S", "E"],
            ["E", "-", "-", "-", "-", "-", "-", "-", "E"],
            ["E", "C", "-", "-", "-", "-", "-", "S", "E"],
            ["E", "E", "E", "E", "E", "E", "E", "E", "E"]]
    def reset(self):
        self.board = [
            ["E", "E", "E", "E", "E", "E", "E", "E", "E"],
            ["E", "S", "-", "-", "-", "-", "-", "C", "E"],
            ["E", "-", "-", "-", "-", "-", "-", "-", "E"],
            ["E", "S", "-", "-", "-", "-", "-", "C", "E"],
            ["E", "-", "-", "-", "-", "-", "-", "-", "E"],
            ["E", "C", "-", "-", "-", "-", "-", "S", "E"],
            ["E", "-", "-", "-", "-", "-", "-", "-", "E"],
            ["E", "C", "-", "-", "-", "-", "-", "S", "E"],l
            ["E", "E", "E", "E", "E", "E", "E", "E", "E"]]
        sCount = 4
        cCount = 4
        turnNumber = 1

    def GUI(self):
        #NOT USED
        WIN = pygame.display.set_mode((self.WIDTH,self.HEIGHT))
        pygame.display.set_caption("geym")
        gameExit = False
        WIN.fill(self.WHITE)
        for i in range(len(self.board)-2):
            for j in range(len(self.board)-2):
                pygame.draw.rect(WIN,self.RED,(i*self.SQUARE_SIZE,j*self.SQUARE_SIZE,self.SQUARE_SIZE,self.SQUARE_SIZE))
        pygame.display.update()

    def boardDrawer(self):
        env = np.zeros((self.HEIGHT, self.WIDTH, 4), dtype=np.uint8)
        surface = cairo.ImageSurface.create_for_data(
            env, cairo.FORMAT_ARGB32, self.HEIGHT, self.WIDTH)
        cr = cairo.Context(surface)
        cr.set_source_rgb(1.0, 1.0, 1.0)
        cr.paint()

        for i in range(len(self.board)-1):
            for j in range(len(self.board)-1):
                if(self.board[i+1][j+1] == 'C'):
                    #cr.rectangle(16+(i*32), 16+(j*32), 8, 8)
                    cr.arc(16 + (j * 32), 16 + (i * 32), 8, 0, 2 * math.pi)
                    cr.set_line_width(3)
                    cr.set_source_rgb(1.0, 0.0, 0.0)
                    cr.fill()
                    cr.stroke()
                elif(self.board[i+1][j+1] == 'S'):
                    cr.rectangle(8+(j*32), 8+(i*32), 16, 16)
                    cr.set_line_width(3)
                    cr.set_source_rgb(1.0, 0.0, 0.0)
                    cr.fill()
                    cr.stroke()

        #surface.write_to_png("circle.png")
        img = Image.fromarray(env, 'RGBA')

        img = img.resize((224, 224))
        return img
    def updateBoardCounters(self,board):
        cCount = 0
        sCount = 0
        for i in range(len(board)):
            for j in range(len(self.board)):
                if (board[i][j] == "C"):
                    cCount = cCount + 1
                elif (board[i][j] == "S"):
                    sCount = sCount + 1
        return sCount,cCount
    def resultFunc(self,board,roundNum):
        sCount,cCount = self.updateBoardCounters(board)
        if (cCount == 0):
            return 5
        if (sCount == 0):
            return -5
        if (roundNum == 50):
            return 2
        return False

    def render(self):
        img = self.boardDrawer()
        img = img.resize((256, 256))  # resizing so we can see our agent in all its glory.
        cv2.imshow("Game Image", np.array(img))  # show it!
        cv2.waitKey(0)

    def isMovePossible(self,i,j,direction,board):
        if(direction == "up" and board[i-1][j] == "-"):
            return True
        elif(direction == "down" and board[i+1][j] == "-"):
            return True
        elif (direction == "right" and board[i][j+1] == "-"):
            return True
        elif (direction == "left" and board[i][j-1] == "-"):
            return True
    def listEveryMove(self,str,board,iTmp,jTmp):
        possibleMoves = []
        for i in range(len(board)):
            for j in range(len(board)):
                if(board[i][j] == str and (i!=iTmp or j!=jTmp)):
                    if(self.isMovePossible(i,j,"up",board)):
                        possibleMoves.append([i,j,"up"])
                    if (self.isMovePossible(i, j, "down",board)):
                        possibleMoves.append([i,j,"down"])
                    if (self.isMovePossible(i, j, "right",board)):
                        possibleMoves.append([i,j,"right"])
                    if (self.isMovePossible(i, j, "left",board)):
                        possibleMoves.append([i,j,"left"])

        return possibleMoves
    def move(self,i,j,direction,board):
        type = board[i][j]
        board[i][j] = "-"
        if (direction == "up"):
            board[i-1][j] = type
        if (direction == "down"):
            board[i+1][j] = type
        if (direction == "right"):
            board[i][j+1] = type
        if (direction == "left"):
            board[i][j-1] = type
        self.updateBoard(board)

    def clearToBeDeleted(self,type,board):
        for i in range(len(board)):
            for j in range(len(board)):
                if(board[i][j] == "toBeDeleted"):
                    board[i][j] = type
    def deleteToBeDeleted(self,board):
        for i in range(len(board)):
            for j in range(len(board)):
                if(board[i][j] == "toBeDeleted"):
                    board[i][j] = "-"

    def updateCounters(self):
        self.cCount = 0
        self.sCount = 0
        for i in range(len(self.board)):
            for j in range(len(self.board)):
                if (self.board[i][j] == "C"):
                    self.cCount = self.cCount + 1
                elif (self.board[i][j] == "S"):
                    self.sCount = self.sCount + 1


    def updateBoard(self,board):
        for i in range(len(board)):
            for j in range(len(board)):
                tmpI = i
                tmpJ = j
                leftFlag = False
                if (board[tmpI][tmpJ] == 'C'):
                    type = "C"
                    oppositeType = "S"
                    while (board[tmpI][tmpJ] == type):
                        board[tmpI][tmpJ] = "toBeDeleted"
                        tmpJ = tmpJ -1
                    if(board[tmpI][tmpJ] == oppositeType or board[tmpI][tmpJ] == "E"):
                        tmpI = i
                        tmpJ = j + 1
                        while (board[tmpI][tmpJ] == type):
                            board[tmpI][tmpJ] = "toBeDeleted"
                            tmpJ = tmpJ + 1
                        if(board[tmpI][tmpJ] == oppositeType or board[tmpI][tmpJ] == "E"):
                            self.deleteToBeDeleted(board)
                            leftFlag = True;
                        else:
                           self.clearToBeDeleted(type,board)
                    else:
                        self.clearToBeDeleted(type,board)
                    if(leftFlag == False):
                        tmpI = i
                        tmpJ = j
                        while (board[tmpI][tmpJ] == type):
                            board[tmpI][tmpJ] = "toBeDeleted"
                            tmpI = tmpI - 1
                        if (board[tmpI][tmpJ] == oppositeType or board[tmpI][tmpJ] == "E"):
                            tmpI = i + 1
                            tmpJ = j
                            while (board[tmpI][tmpJ] == type):
                                board[tmpI][tmpJ] = "toBeDeleted"
                                tmpI = tmpI + 1
                            if (board[tmpI][tmpJ] == oppositeType or board[tmpI][tmpJ] == "E"):
                                self.deleteToBeDeleted(board)
                            else:
                                self.clearToBeDeleted(type,board)
                        else:
                            self.clearToBeDeleted(type,board)
                elif(board[tmpI][tmpJ] == 'S'):
                    type = "S"
                    oppositeType = "C"
                    while (board[tmpI][tmpJ] == type):
                        board[tmpI][tmpJ] = "toBeDeleted"
                        tmpJ = tmpJ -1
                    if(board[tmpI][tmpJ] == oppositeType or board[tmpI][tmpJ] == "E"):
                        tmpI = i
                        tmpJ = j + 1
                        while (board[tmpI][tmpJ] == type):
                            board[tmpI][tmpJ] = "toBeDeleted"
                            tmpJ = tmpJ + 1
                        if(board[tmpI][tmpJ] == oppositeType or board[tmpI][tmpJ] == "E"):
                            leftFlag = True
                            self.deleteToBeDeleted(board)
                        else:
                            self.clearToBeDeleted(type,board)
                    else:
                        self.clearToBeDeleted(type,board)
                    if (leftFlag == False):
                        tmpI = i
                        tmpJ = j
                        while (board[tmpI][tmpJ] == type):
                            board[tmpI][tmpJ] = "toBeDeleted"
                            tmpI = tmpI - 1
                        if (board[tmpI][tmpJ] == oppositeType or board[tmpI][tmpJ] == "E"):
                            tmpI = i + 1
                            tmpJ = j
                            while (board[tmpI][tmpJ] == type):
                                board[tmpI][tmpJ] = "toBeDeleted"
                                tmpI = tmpI + 1
                            if (board[tmpI][tmpJ] == oppositeType or board[tmpI][tmpJ] == "E"):
                                self.deleteToBeDeleted(board)
                            else:
                                self.clearToBeDeleted(type,board)
                        else:
                            self.clearToBeDeleted(type,board)
    def evaluate(self,board):
        sCount,cCount = self.updateBoardCounters(board)
        return (sCount * 0.5) - (cCount * 0.5)
    def tempminimax(self,board,depth,isMaximizing,roundNum,alpha,beta,avoidedI,avoidedJ,firstmoveFlag): #this is not optimized
        sCount, cCount = self.updateBoardCounters(board)
        if(depth == 0):
            return self.evaluate(board),board
        elif self.resultFunc(board,roundNum) != False:
            return self.resultFunc(board,roundNum),board
        tmpBoard = deepcopy(board)
        if(isMaximizing):
            if(firstmoveFlag):
                maxEval = float('-inf')
                best_move = None
                for move in self.listEveryMove("S", board, avoidedI, avoidedJ):
                    i = int(move[0])
                    j = int(move[1])
                    direction = str(move[2])
                    self.move(i, j, direction, tmpBoard)
                    if (i == "up"):
                        i = i - 1
                    if (direction == "down"):
                        i = i + 1
                    if (direction == "right"):
                        j = j + 1
                    if (direction == "left"):
                        j = j - 1
                    evaluation = self.tempminimax(tmpBoard, depth, True, roundNum, alpha, beta, i, j, False)[0]
                    if (evaluation > maxEval):
                        maxEval = evaluation
                        best_move = move
                    # maxEval = max(maxEval,evaluation)
                    # if maxEval == evaluation:
                    #    best_move = move
                    tmpBoard = deepcopy(board)
                    alpha = max(alpha, maxEval)
                    if (beta <= alpha):
                        break
                return maxEval, best_move
            else:
                maxEval = float('-inf')
                best_move = None
                for move in self.listEveryMove("S",board,avoidedI,avoidedJ):
                    i = int(move[0])
                    j = int(move[1])
                    direction = str(move[2])
                    self.move(i,j,direction,tmpBoard)
                    evaluation= self.tempminimax(tmpBoard,depth-1,False,roundNum+1,alpha,beta,0,0,True)[0]
                    if(evaluation > maxEval):
                        maxEval = evaluation
                        best_move = move
                    #maxEval = max(maxEval,evaluation)
                    #if maxEval == evaluation:
                    #    best_move = move
                    tmpBoard = deepcopy(board)
                    alpha = max(alpha,maxEval)
                    if(beta<=alpha):
                        break
                return maxEval , best_move
        else:
            if(firstmoveFlag):
                minEval = float('inf')
                best_move = None
                for move in self.listEveryMove("C", board, 0, 0):
                    i = int(move[0])
                    j = int(move[1])
                    direction = str(move[2])
                    self.move(i, j, direction, tmpBoard)
                    if (i == "up"):
                        i = i - 1
                    if (direction == "down"):
                        i = i + 1
                    if (direction == "right"):
                        j = j + 1
                    if (direction == "left"):
                        j = j - 1
                    evaluation = self.tempminimax(tmpBoard, depth, False, roundNum, alpha, beta, i, j, False)[0]
                    # minEval = min(minEval, evaluation)
                    if (evaluation < minEval):
                        minEval = evaluation
                        best_move = move
                    tmpBoard = deepcopy(board)
                    beta = min(beta, minEval)
                    if beta <= alpha:
                        break
                return minEval, best_move
            else:
                minEval = float('inf')
                best_move = None
                for move in self.listEveryMove("C", board,0,0):
                    i = int(move[0])
                    j = int(move[1])
                    direction = str(move[2])
                    self.move(i, j, direction, tmpBoard)
                    evaluation = self.tempminimax(tmpBoard, depth - 1, True, roundNum + 1,alpha,beta,0,0,True)[0]
                    #minEval = min(minEval, evaluation)
                    if(evaluation < minEval):
                        minEval = evaluation
                        best_move = move
                    tmpBoard = deepcopy(board)
                    beta = min(beta, minEval)
                    if beta <= alpha:
                        break
                return minEval, best_move
    def minimax(self,board,depth,isMaximizing,roundNum,alpha,beta,avoidedI,avoidedJ):
        sCount, cCount = self.updateBoardCounters(board)
        if(depth == 0):
            return self.evaluate(board),board
        elif self.resultFunc(board,roundNum) != False:
            return self.resultFunc(board,roundNum),board
        tmpBoard = deepcopy(board)
        if(isMaximizing):
            maxEval = float('-inf')
            best_move = None
            for move in self.listEveryMove("S",board,avoidedI,avoidedJ):
                i = int(move[0])
                j = int(move[1])
                direction = str(move[2])
                self.move(i,j,direction,tmpBoard)
                evaluation= self.minimax(tmpBoard,depth-1,False,roundNum+1,alpha,beta,0,0)[0]
                if(evaluation > maxEval):
                    maxEval = evaluation
                    best_move = move
                #maxEval = max(maxEval,evaluation)
                #if maxEval == evaluation:
                #    best_move = move
                tmpBoard = deepcopy(board)
                alpha = max(alpha,maxEval)
                if(beta<=alpha):
                    break
            return maxEval , best_move
        else:
            minEval = float('inf')
            best_move = None
            for move in self.listEveryMove("C", board,0,0):
                i = int(move[0])
                j = int(move[1])
                direction = str(move[2])
                self.move(i, j, direction, tmpBoard)
                evaluation = self.minimax(tmpBoard, depth - 1, True, roundNum + 1,alpha,beta,0,0)[0]
                #minEval = min(minEval, evaluation)
                if(evaluation < minEval):
                    minEval = evaluation
                    best_move = move
                tmpBoard = deepcopy(board)
                beta = min(beta, minEval)
                if beta <= alpha:
                    break
            return minEval, best_move

    def readLastLine(self):
        with open(filePath,"r") as file:
            for line in file:
                pass
            last_line = line
        return line
    def writeToLastLine(self,moveId,playerId,y,x,direction):
        tmpStr = None
        destionationY = y
        destionationX = x
        if (direction == "up"):
            destionationY = y - 1
        if (direction == "down"):
            destionationY = y + 1
        if (direction == "right"):
            destionationX = x + 1
        if (direction == "left"):
            destionationX = x - 1
        tmpStr = str(moveId) + ";" + str(playerId) + ";" + str(y)+ ";" + str(x) + ";" + str(destionationY) + ";" + str(destionationX)
        with open(filePath, 'a+') as file:
            for lines in file:
                pass
            file.write(tmpStr + '\n')

    def startGame(self):
        depth = 6
        while(self.resultFunc(self.board,self.roundNum) == False):
            tmpBoard = deepcopy(self.board)
            if(self.turn == 2):

                moves = self.listEveryMove("C",self.board,0,0)
                i = 1
                for move in moves:
                    print(i," : ",move)
                    i += 1
                print("Enter your move: ")
                moveNr = int(input()) - 1
                print("You selected: ",moves[moveNr])
                i = moves[moveNr][0]
                j = moves[moveNr][1]
                direction = moves[moveNr][2]
                self.move(i,j,direction,self.board)
                self.updateCounters()
                self.render()
                if(self.cCount > 1 and self.resultFunc(self.board,self.roundNum) == False):
                    if (direction == "up"):
                        i = i -1
                    if (direction == "down"):
                        i = i + 1
                    if (direction == "right"):
                        j = j + 1
                    if (direction == "left"):
                        j = j - 1

                    moves = self.listEveryMove("C", self.board,i,j)
                    i = 1
                    for move in moves:
                        print(i, " : ", move)
                        i += 1
                    print("Enter your move: ")
                    moveNr = int(input()) - 1
                    print("You selected: ", moves[moveNr])
                    i = moves[moveNr][0]
                    j = moves[moveNr][1]
                    direction = moves[moveNr][2]
                    self.move(i, j, direction, self.board)
                    self.updateCounters()
                    self.render()

                self.turn = 1
                self.roundNum = self.roundNum + 1

            else:
                print("AI is making it's move")
                maxVal = float("-inf")
                best_move = None
                best_move = best_move = self.minimax(tmpBoard, depth, True, self.roundNum, float("-inf"), float("inf"),0,0)[1]
                #best_move = self.tempminimax(tmpBoard,depth,True,self.roundNum,float("-inf"),float("inf"),0,0,True)[1]
                print(best_move)
                self.writeToLastLine((self.roundNum * 2) - 1 ,1,best_move[0], best_move[1], best_move[2])
                self.move(best_move[0], best_move[1], best_move[2], self.board)
                self.updateCounters()
                self.render()
                if (self.sCount > 1 and self.resultFunc(self.board,self.roundNum) == False):
                    first_move = best_move
                    if (first_move[2] == "up"):
                        first_move[0] = first_move[0] - 1
                    if (first_move[2] == "down"):
                        first_move[0] = first_move[0] + 1
                    if (first_move[2] == "right"):
                        first_move[1] = first_move[1] + 1
                    if (first_move[2] == "left"):
                        first_move[1] = first_move[1] - 1

                    tmpBoard = deepcopy(self.board)
                    maxVal = float("-inf")
                    best_move = self.minimax(tmpBoard, depth, True, self.roundNum, float("-inf"), float("inf"),first_move[0],first_move[1])[1]
                    #best_move = self.tempminimax(tmpBoard, depth, True, self.roundNum, float("-inf"), float("inf"),first_move[0],first_move[1],False)[1]
                    print(best_move)
                    self.writeToLastLine((self.roundNum * 2), 1, best_move[0], best_move[1],best_move[2])
                    self.move(best_move[0], best_move[1], best_move[2], self.board)
                    self.render()
                    self.updateCounters()
                self.turn = 2
                self.roundNum = self.roundNum + 1

        if(self.resultFunc(self.board,self.roundNum) == 5):
            print("AI WON")
        elif(self.resultFunc(self.board,self.roundNum) == -5):
            print("PLAYER WON")
        else:
            print("DRAW")







#check if object,
   #check left spot if out of index or other type of object
        #if other type of object add left side to toBeDeleted list

def main():
    gameboard = GameBoard()
    #gameboard.render()
    gameboard.startGame()
    #gameboard.GUI()


if __name__ == "__main__":
    main()