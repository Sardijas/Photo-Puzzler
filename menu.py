from PIL import Image
import tkinter.font
import math, copy, random, time, tkinter, string
import cv2 as cv
import requests, io
import numpy as np

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
    app.errorFont = tkinter.font.Font(family='Helvetica', size=20, weight='bold')
    app.titleFont = tkinter.font.Font(family='Helvetica', size=48, weight='bold')

    #Artwork storage
    app.artworkJson = None

    #Photo init
    app.selectedFile = None
    app.photoSelected = False
    app.rawPhoto = None
    app.photoArray = None
    app.photo = None

    #Algorithm chooser init
    app.algorithm = False

    #Error handler
    app.errorText = ""



#Random Artwork##############################################################################

#Gets a random artwork from SMK gallery's API
# References:
# https://www.smk.dk/en/article/smk-api/
# Note: It looks like the api is fairly new, and its random function 
# currently only gets semi-random paintings from a set of 4 or 5. :(
def getRandomArtwork(app):

    #Send request for random artwork
    #References:
    # https://docs.python-requests.org/en/latest/
    # https://realpython.com/api-integration-in-python/
    # https://www.educative.io/edpresso/how-to-generate-a-random-string-in-python
    # https://www.w3schools.com/python/ref_requests_get.asp

    randomKey = ''.join(random.choice(string.ascii_letters) for i in range(random.randint(1,5)))

    url = 'https://api.smk.dk/api/v1/art/search/?'
    param = {"keys":"*", "randomHighlights":randomKey, "rows":1, "lang":"en"}
    request = requests.get(url, params=param)

    #Store response
    app.artworkJson = request.json()

    #Get artwork url from response
    artworkUrl = app.artworkJson['items'][0]['image_thumbnail']

    #Get image from artwork url
    # References:
    # https://www.kite.com/python/answers/how-to-read-an-image-data-from-a-url-in-python
    artwork = requests.get(artworkUrl)
    artwork_bytes = io.BytesIO(artwork.content)
    app.rawPhoto = np.array(Image.open(artwork_bytes))
    


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
    photoInit(app, 0)


#Attempts to load an image file. If this fails, errors. 
# Otherwise formats photo to display in photo stage
def photoInit(app, mode):
    #Handles user uploaded photos
    if (app.selectedFile != None and mode == 0):
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
            app.errorText = ""
        except:
            app.errorText = "Incorrect file type"

    #Handles photos downloaded from SMK api
    elif (mode == 1):
        app.photoArray = resizeImage(app)
        app.photo = Image.fromarray(app.photoArray)
        app.photoSelected = True
        app.errorText = ""


#Handle Click##############################################################################

#Handles mouse click
def mousePressed(app, event):
    cx, cy = event.x, event.y

    #If a photo has been selected, check for clicks in larger photo stage and play buttons
    if (app.photoSelected):
        #Check for upload file button
        if ((app.widthMargin < cx < app.width - app.widthMargin) and
            (app.heightMargin * 2 < cy < app.maxHeight)):

            #Select photo
            selectFile(app)

        #Check for easy mode button
        elif ((app.widthMargin < cx < (app.widthMargin + app.maxWidth) / 3) and
            (app.heightMargin * 0.25 + app.maxHeight < cy 
            < app.heightMargin + app.maxHeight)):

            #Play puzzle
            main.playPuzzle(app.selectedFile, app.algorithm, 0, app.artworkJson)

        #Check for medium mode button
        elif (((2 * app.widthMargin + app.maxWidth) / 3 < cx < 
        2 * (((app.widthMargin / 2 + app.maxWidth) / 3))) and
            (app.heightMargin * 0.25 + app.maxHeight < cy 
            < app.heightMargin + app.maxHeight)):

            #Play puzzle
            main.playPuzzle(app.selectedFile, app.algorithm, 1, app.artworkJson)

        #Check for hard mode button
        elif ((app.widthMargin < cx < app.maxWidth) and
            (2 * ((app.widthMargin + app.maxWidth) / 3) < cy 
            < app.heightMargin + app.maxHeight)):

            #Play puzzle
            main.playPuzzle(app.selectedFile, app.algorithm, 2, app.artworkJson)

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

    #Check for artist mode button
    elif ((app.widthMargin < cx < app.maxWidth) and
        (app.heightMargin * 0.25 + app.maxHeight < cy < app.heightMargin * 1.25 + app.maxHeight)):
    
        getRandomArtwork(app)
        photoInit(app, 1)
        main.playPuzzle(app.rawPhoto, app.algorithm, 3, app.artworkJson)



#Draw##############################################################################

#Draws start menu on the canvas
def redrawAll(app, canvas):

    #Background
    canvas.create_rectangle(0, 0, app.width, app.height, fill="#cce0ff", outline="#cce0ff")

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
        canvas.create_rectangle(app.widthMargin, 
        app.heightMargin * 0.25 + app.maxHeight, 
        (app.widthMargin + app.maxWidth) / 3, app.heightMargin + app.maxHeight, 
        fill="#382985", outline="#382985")

        #Easy mode button
        canvas.create_text((app.widthMargin + (app.widthMargin  + app.maxWidth) / 3) / 2, 
        (app.maxHeight  + app.heightMargin * 0.25)  + 
        ((app.heightMargin + app.maxHeight) - (app.heightMargin * 0.25 + app.maxHeight)) / 2, 
        text="Easy", fill="white", font=app.font)

        canvas.create_text((app.widthMargin + (app.widthMargin + app.maxWidth) / 3) / 2, 
        app.heightMargin * 0.25 + app.maxHeight - app.heightMargin, fill="red")

        #Medium mode button
        canvas.create_rectangle((2 * app.widthMargin + app.maxWidth) / 3, 
        app.heightMargin * 0.25 + app.maxHeight, 
        2 * (((app.widthMargin / 2 + app.maxWidth) / 3)), app.heightMargin + app.maxHeight, 
        fill="#382985", outline="#382985")

        canvas.create_text(((app.widthMargin + app.maxWidth) / 3 + 
        2 * ((app.widthMargin + app.maxWidth) / 3)) / 2, 
        (app.maxHeight  + app.heightMargin * 0.25)  + 
        ((app.heightMargin + app.maxHeight) - (app.heightMargin * 0.25 + app.maxHeight)) / 2, 
        text="Medium", fill="white", font=app.font)

        #Hard mode button
        canvas.create_rectangle(2 * ((app.widthMargin + app.maxWidth) / 3), 
        app.heightMargin * 0.25 + app.maxHeight, 
        app.maxWidth, app.heightMargin + app.maxHeight, fill="#382985", outline="#382985")

        canvas.create_text((2 * ((app.widthMargin + app.maxWidth) / 3) + app.maxWidth) / 2, 
        (app.maxHeight  + app.heightMargin * 0.25)  + 
        ((app.heightMargin + app.maxHeight) - (app.heightMargin * 0.25 + app.maxHeight)) / 2, 
        text="Hard", fill="white", font=app.font)

    else:
        #Photo stage
        canvas.create_rectangle(app.widthMargin, app.heightMargin * 2, app.maxWidth, 
            app.maxHeight - app.heightMargin, fill="orange", outline="orange")
        
        canvas.create_text(app.width / 2, 
        app.heightMargin + (app.maxHeight - app.heightMargin) / 2,
        text="Click to upload photo", fill="white", font=app.font)
        
        #Easy mode button
        canvas.create_rectangle(app.widthMargin, 
        app.heightMargin * 0.25 + app.maxHeight - app.heightMargin, 
        (app.widthMargin + app.maxWidth) / 3, app.maxHeight, fill="gray", outline="gray")

        canvas.create_text((app.widthMargin + (app.widthMargin  + app.maxWidth) / 3) / 2, 
        (app.maxHeight - app.heightMargin * 1.5 + app.heightMargin * 0.25)  + 
        ((app.maxHeight) - 
        (app.heightMargin * 0.25 + app.maxHeight - app.heightMargin * 2)) / 2, 
        text="Easy", fill="black", font=app.font)

        #Medium mode button
        canvas.create_rectangle((2 * app.widthMargin + app.maxWidth) / 3, 
        app.heightMargin * 0.25 + app.maxHeight - app.heightMargin, 
        2 * (((app.widthMargin / 2 + app.maxWidth) / 3)), app.maxHeight, 
        fill="gray", outline="gray")

        canvas.create_text(((app.widthMargin + app.maxWidth) / 3 + 
        2 * ((app.widthMargin + app.maxWidth) / 3)) / 2, 
        (app.maxHeight - app.heightMargin * 1.5 + app.heightMargin * 0.25)  + 
        ((app.maxHeight) - 
        (app.heightMargin * 0.25 + app.maxHeight - app.heightMargin * 2)) / 2, 
        text="Medium", fill="black", font=app.font)

        #Hard mode button
        canvas.create_rectangle(2 * ((app.widthMargin + app.maxWidth) / 3), 
        app.heightMargin * 0.25 + app.maxHeight - app.heightMargin, 
        app.maxWidth, app.maxHeight, fill="gray", outline="gray")

        canvas.create_text((2 * ((app.widthMargin + app.maxWidth) / 3) + app.maxWidth) / 2, 
        (app.maxHeight - app.heightMargin * 1.5 + app.heightMargin * 0.25)  + 
        ((app.maxHeight) - 
        (app.heightMargin * 0.25 + app.maxHeight - app.heightMargin * 2)) / 2, 
        text="Hard", fill="black", font=app.font)

        #Error text
        canvas.create_text(app.width // 2, app.height // 2 - 3.25 * app.heightMargin, 
        text=app.errorText, fill="red", font=app.errorFont)

        #Random artwork (artist mode) button
        canvas.create_rectangle(app.widthMargin, 
        app.heightMargin * 0.25 + app.maxHeight, app.maxWidth, 
        app.heightMargin * 1.25 + app.maxHeight, fill="#382985", outline="#382985")

        canvas.create_text(app.width / 2, (app.heightMargin * 1.5 + 2 * app.maxHeight) / 2,
        text="Artist", fill="white", font=app.font)

    
    #Algorithm toggle
    canvas.create_rectangle(app.widthMargin * 2, 
        app.height - app.heightMargin * 1.5, 
        app.maxWidth - app.widthMargin, app.height - app.heightMargin, fill="#382985", outline="#382985")

    if (not app.algorithm):
        canvas.create_text(app.width / 2, app.height - (app.heightMargin + app.heightMargin * 1.5) / 2, 
        text="Median Cut Algorithm", fill="white")
    else:
        canvas.create_text(app.width / 2, app.height - (app.heightMargin + app.heightMargin * 1.5) / 2, 
        text="Modified Median Cut Algorithm", fill="white")



def playGame():
    runApp(width=800, height=850)
    

playGame()
