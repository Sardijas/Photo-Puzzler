from PIL import Image
import tkinter.font
import math, copy, random, time, tkinter
import cv2 as cv

from cmu_112_graphics import *

import main

#Image Processing##############################################################################

def appStarted(app):
    #Viewport
    app.widthMargin = app.width // 7
    app.heightMargin = app.height // 10
    app.maxWidth = app.width - app.widthMargin
    app.maxHeight = app.height - 4 * app.heightMargin

    #Fonts
    #References:
    # https://docs.python.org/3/library/tkinter.font.html
    app.font = tkinter.font.Font(family='Helvetica', size=36, weight='bold')
    app.titleFont = tkinter.font.Font(family='Helvetica', size=48, weight='bold')

    #Photo init
    app.selectedFile = None
    app.photoSelected = False
    app.rawPhoto = None
    app.photoArray = None
    app.photo = None
    photoInit(app)

    #Algorithm chooser init
    app.algorithm = False


#Image Processing##############################################################################

#Resizes image to be displayed in photo stage
#References:
# https://docs.opencv.org/3.4/da/d54/group__imgproc__
# transform.html#ga47a974309e9102f5f08231edc7e7529d
# https://www.pyimagesearch.com/2021/01/20/opencv-resize-image-cv2-resize/
# https://www.tutorialspoint.com/using-opencv-with-tkinter
def resizeImage(app):

    #Size down
    if (len(app.rawPhoto) > (app.maxHeight - 4 * app.heightMargin)
        or len(app.rawPhoto) > (app.maxWidth - 3 * app.widthMargin)):
        interpolation = cv.INTER_AREA

    #Size up
    else:
        interpolation = cv.INTER_CUBIC

    #Create destination array
    app.stage = (app.maxWidth - 4 * app.widthMargin, 
    app.maxHeight - 3 * app.heightMargin)

    #Resize image into destination array
    img = cv.resize(app.rawPhoto, app.stage, 
         interpolation)

    return  img


#File Management##############################################################################

#Opens a file manager dialog for the user to select a photo
def selectFile(app):
    #References:
    # #https://www.pythontutorial.net/tkinter/tkinter-open-file-dialog/
    filename = tkinter.filedialog.askopenfilename()
    app.selectedFile = filename
    photoInit(app)

#Attempts to load an image file. If this fails, errors. 
# Otherwise formats photo to display in photo stage
def photoInit(app):
    if (app.selectedFile != None):
        try:
            app.rawPhoto = cv.imread(app.selectedFile)
            app.photoArray = resizeImage(app)
            
            #References:
            # https://docs.opencv.org/3.4.13/d8/d01/group__imgproc__color__
            # conversions.html#gga4e0972be5de079fed4e3a10e24ef5ef0a95d70bf0c
            # 1b5aa58d1eb8abf03161f21
            app.photoArray = cv.cvtColor(app.photoArray, cv.COLOR_BGR2RGBA)
            app.photo = Image.fromarray(app.photoArray)
            app.photoSelected = True
        except:
            print("An error occurred")

        #Add handle specific exceptions


#Handle Click##############################################################################

#Handles mouse click
def mousePressed(app, event):
    cx, cy = event.x, event.y

    #If a photo has been selected, check for clicks in larger photo stage and play button
    if (app.photoSelected):
        if ((app.widthMargin < cx < app.width - app.widthMargin) and
            (app.heightMargin * 2 < cy < app.maxHeight)):

            #Select photo
            selectFile(app)

        elif ((app.widthMargin < cx < app.width - app.widthMargin) and
            (app.heightMargin * 0.25 + app.maxHeight < cy 
            < app.heightMargin + app.maxHeight)):

            #Play puzzle
            main.playPuzzle(app.selectedFile, app.algorithm)

    #Otherwise check for clicks in smaller photo stage
    else:
        if ((app.widthMargin < cx < app.width - app.widthMargin) and
            (app.heightMargin * 2 < cy < app.maxHeight - app.heightMargin)):

            #Select photo
            selectFile(app)

    #Check for algorithm toggle clicks
    if ((app.widthMargin * 2 < cx < app.maxWidth - app.widthMargin) and
        (app.height - app.heightMargin * 1.5 < cy < app.height - app.heightMargin)):

        if (app.algorithm):
            app.algorithm = False
        else:
            app.algorithm = True


#Draw##############################################################################

#Draws start menu on the canvas
def redrawAll(app, canvas):

    #Main menu text
    canvas.create_text(app.width / 2, app.heightMargin, text="Main Menu", fill="black",
        font=app.titleFont)

    #Layout if photo has been selected
    if (app.photoSelected):

        #Photo stage
        canvas.create_rectangle(app.widthMargin, app.heightMargin * 2, app.maxWidth, 
            app.maxHeight, fill="orange", outline="orange")
        
        canvas.create_image(app.width // 2, app.height // 2 - app.heightMargin, 
            image=ImageTk.PhotoImage(app.photo))

        #Play button
        canvas.create_rectangle(app.widthMargin, app.heightMargin * 0.25 + app.maxHeight, 
        app.maxWidth, app.heightMargin + app.maxHeight, fill="purple", outline="purple")

        canvas.create_text(app.width / 2, (app.maxHeight  + app.heightMargin * 0.25)  + 
        ((app.heightMargin + app.maxHeight) - (app.heightMargin * 0.25 + app.maxHeight)) / 2, 
        text="Play", fill="white", font=app.font)

    else:
        #Photo stage
        canvas.create_rectangle(app.widthMargin, app.heightMargin * 2, app.maxWidth, 
            app.maxHeight - app.heightMargin, fill="orange", outline="orange")
        
        canvas.create_text(app.width / 2, 
        app.heightMargin + (app.maxHeight - app.heightMargin) / 2,
        text="Click to upload photo", fill="white", font=app.font)
        
        #Play button
        canvas.create_rectangle(app.widthMargin, 
        app.heightMargin * 0.25 + app.maxHeight - app.heightMargin, 
        app.maxWidth, app.maxHeight, fill="gray", outline="gray")

        canvas.create_text(app.width / 2, 
        (app.maxHeight - app.heightMargin * 1.5 + app.heightMargin * 0.25)  + 
        ((app.maxHeight) - 
        (app.heightMargin * 0.25 + app.maxHeight - app.heightMargin * 2)) / 2, 
        text="Play", fill="black", font=app.font)

    
    #Algorithm toggle
    canvas.create_rectangle(app.widthMargin * 2, 
        app.height - app.heightMargin * 1.5, 
        app.maxWidth - app.widthMargin, app.height - app.heightMargin, fill="purple", outline="purple")

    if (not app.algorithm):
        canvas.create_text(app.width / 2, app.height - (app.heightMargin + app.heightMargin * 1.5) / 2, 
        text="Median Cut Algorithm", fill="white")
    else:
        canvas.create_text(app.width / 2, app.height - (app.heightMargin + app.heightMargin * 1.5) / 2, 
        text="Modified Median Cut Algorithm", fill="white")



def playGame():
    runApp(width=800, height=850)
    

playGame()


#Add algorithm toggle