from inspect import currentframe
from PIL import Image
import tkinter.font
import math, copy, random, time, tkinter
import cv2 as cv
# import numpy as np

from cmu_112_graphics import *

import main

def appStarted(app):
    #Viewport
    app.widthMargin = app.width // 7
    app.heightMargin = app.height // 10
    app.maxWidth = app.width - app.widthMargin
    app.maxHeight = app.height - 4 * app.heightMargin

    #https://docs.python.org/3/library/tkinter.font.html
    app.font = tkinter.font.Font(family='Helvetica', size=36, weight='bold')
    app.titleFont = tkinter.font.Font(family='Helvetica', size=48, weight='bold')

    #Photo init
    app.selectedFile = None
    app.photoSelected = False
    app.rawPhoto = None
    app.photoArray = None
    app.photo = None
    photoInit(app)

#https://www.pythontutorial.net/tkinter/tkinter-open-file-dialog/

def photoInit(app):
    if (app.selectedFile != None):
        try:
            app.rawPhoto = cv.imread(app.selectedFile)
            app.photoArray = resizeImage(app)
            #https://docs.opencv.org/3.4.13/d8/d01/group__imgproc__color__
            # conversions.html#gga4e0972be5de079fed4e3a10e24ef5ef0a95d70bf0c
            # 1b5aa58d1eb8abf03161f21
            app.photoArray = cv.cvtColor(app.photoArray, cv.COLOR_BGR2RGBA)
            app.photo = Image.fromarray(app.photoArray)
            app.photoSelected = True
        except:
            print("Oh no")

def resizeImage(app):

    #Size up
    if (len(app.rawPhoto) > (app.maxHeight - 4 * app.heightMargin)
        or len(app.rawPhoto) > (app.maxWidth - 3 * app.widthMargin)):
        interpolation = cv.INTER_AREA

    #Size down
    else:
        interpolation = cv.INTER_CUBIC

    #Create destination array
    app.stage = (app.maxWidth - 4 * app.widthMargin, 
    app.maxHeight - 3 * app.heightMargin)

    #Resize image into destination array
    img = cv.resize(app.rawPhoto, app.stage, 
         interpolation)

    return  img

def selectFile(app):
    filename = tkinter.filedialog.askopenfilename()
    app.selectedFile = filename
    photoInit(app)

def mousePressed(app, event):
    cx, cy = event.x, event.y

    if (app.photoSelected):
        if ((app.widthMargin < cx < app.width - app.widthMargin) and
            (app.heightMargin * 2 < cy < app.maxHeight)):
            selectFile(app)
        elif ((app.widthMargin < cx < app.width - app.widthMargin) and
            (app.heightMargin * 0.25 + app.maxHeight < cy 
            < app.heightMargin + app.maxHeight)):
            main.playPuzzle(app.selectedFile)

    else:
        if ((app.widthMargin < cx < app.width - app.widthMargin) and
            (app.heightMargin * 2 < cy < app.maxHeight - app.heightMargin)):
            selectFile(app)

def redrawAll(app, canvas):
    canvas.create_text(app.width / 2, app.heightMargin, text="Main Menu", fill="red",
        font=app.titleFont)

    if (app.photoSelected):

        canvas.create_rectangle(app.widthMargin, app.heightMargin * 2, app.maxWidth, 
            app.maxHeight, fill="blue")
        
        canvas.create_image(app.width // 2, app.height // 2 - app.heightMargin, 
            image=ImageTk.PhotoImage(app.photo))

        canvas.create_rectangle(app.widthMargin, app.heightMargin * 0.25 + app.maxHeight, 
        app.maxWidth, app.heightMargin + app.maxHeight, fill="green")

        canvas.create_text(app.width / 2, (app.maxHeight  + app.heightMargin * 0.25)  + 
        ((app.heightMargin + app.maxHeight) - (app.heightMargin * 0.25 + app.maxHeight)) / 2, 
        text="Play", fill="white", font=app.font)

    else:

        canvas.create_rectangle(app.widthMargin, app.heightMargin * 2, app.maxWidth, 
            app.maxHeight - app.heightMargin, fill="blue")
        
        canvas.create_text(app.width / 2, 
        app.heightMargin + (app.maxHeight - app.heightMargin) / 2,
        text="Click to upload photo", fill="white", font=app.font)
        
        canvas.create_rectangle(app.widthMargin, 
        app.heightMargin * 0.25 + app.maxHeight - app.heightMargin, 
        app.maxWidth, app.maxHeight, fill="gray")

        canvas.create_text(app.width / 2, 
        (app.maxHeight - app.heightMargin * 1.5 + app.heightMargin * 0.25)  + 
        ((app.maxHeight) - 
        (app.heightMargin * 0.25 + app.maxHeight - app.heightMargin * 2)) / 2, 
        text="Play", fill="black", font=app.font)

    # canvas.create_rectangle(app.widthMargin, app.heightMargin, 
    #     app.width - app.widthMargin, app.height - app.heightMargin, fill="blue")

def playGame():
    runApp(width=800, height=850)
    

playGame()
