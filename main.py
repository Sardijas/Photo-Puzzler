from collections import Counter
from PIL import Image
from PIL import ImageColor
import tkinter.font
import math, copy, random, time, tkinter
import requests
import cv2 as cv
import numpy as np

#This module is just being used for color algorithm comparison :)
from colorthief import ColorThief

from cmu_112_graphics import *

import menu

# Credit for inspiration and debugging tips:
# Xinyi Luo, Lauren Sands, Cole Savage, 
# Zara Mansoor, Shannon Yang, Christine Li


#Init##########################################################################
def appStarted(app):
    #Viewport
    app.widthMargin = app.width // 7
    app.heightMargin = app.height // 10
    app.maxWidth = app.width - 2 * app.widthMargin
    app.maxHeight = app.height - 2 * app.heightMargin

    #Mode init
    if (app.myMode == 0):
        app.rows = 20
        app.cols = 15
        app.paletteSize = 4
        app.rawPhoto = cv.imread(app.filename)

    elif (app.myMode == 1):
        app.rows = 30
        app.cols = 25
        app.paletteSize = 5
        app.rawPhoto = cv.imread(app.filename)

    elif (app.myMode == 2):
        app.rows = 40
        app.cols = 35
        app.paletteSize = 7
        app.rawPhoto = cv.imread(app.filename)

    else:
        app.rows = 55
        app.cols = 50
        app.paletteSize = 8
        #In this case filename input from menu is rawPhoto
        #Temporarily convert to BGR
        app.rawPhoto = cv.cvtColor(app.filename, cv.COLOR_BGR2RGB)


    #Photo init
    #Reference:
    # https://www.cs.cmu.edu/~112/notes/notes-animations-part4.html#loadImageUsingFile
    app.photoArray = resizeImage(app)
    app.photo = Image.fromarray(app.photoArray)

    #Palette init

    if (not app.algorithm):
        app.palette = getPaletteMedianCut(app)
    else:
        app.palette = getPaletteModifiedMedianCut(app, app.filename)
    #^ used for calculating colors in solutionGrid

    app.usedPalette = set([])
    #^ used for drawing color buckets
    
    app.colorCounter = {}
    #^ tracks frequency of each palette color

    #Maybe make this controllable, max 48/48
    #Set up grid


    app.answerGrid = getGrid(app)
    app.solutionGrid = getSolution(app)
    app.emptyGrid = getGrid(app)

    #Trim palette to paletteSize
    trimPalette(app)

    #Final palette init
    app.radius = (app.heightMargin * (2/3)) / 2
    app.colorData = placeColors(app)

    #Hints init
    app.hintRows, app.hintCols, app.hRSorted, app.hCSorted = getHints(app)

    #Brush init
    app.brushColor = None

    #Error init
    app.errors = set([])

    #Win state init
    app.isWin = False
    app.blurredImageArray = blurImage(app)
    app.blurredImage = Image.fromarray(app.blurredImageArray)
    app.titleFont = tkinter.font.Font(family='Helvetica', size=48, weight='bold')

    #Solve init
    app.isSolve = False

    #Hint Fonts
    # References:
    # https://docs.python.org/3/library/tkinter.font.html
    # https://www.tutorialspoint.com/modify-the-default-font-in-python-tkinter#:~:text=Running%20the%20above%20code%20will,widgets%20that%20uses%20textual%20information
    app.font = tkinter.font.Font(family='Lucida', size=16, weight='bold')
    app.smallFont = tkinter.font.Font(family='Lucida', size=10)

    #Tutorial init
    app.isTutorial = False


#Image processing ##########################################################################

#Creates a blurred version of the input image to be displayed on win
def blurImage(app):
    
    #References:
    # https://docs.opencv.org/3.4/d4/d13/tutorial_py_filtering.html
    rgbImage = cv.cvtColor(app.photoArray, cv.COLOR_BGR2RGBA)
    blurredImage = cv.GaussianBlur(rgbImage, (21, 21), 0)

    return blurredImage


#Resizes an image to fit puzzle gameboard area
#References:
# https://docs.opencv.org/3.4/da/d54/group__imgproc__
# transform.html#ga47a974309e9102f5f08231edc7e7529d
# https://www.pyimagesearch.com/2021/01/20/opencv-resize-image-cv2-resize/
# https://www.tutorialspoint.com/using-opencv-with-tkinter
def resizeImage(app):

    #Size down interpolation
    if (len(app.rawPhoto) > app.maxHeight 
        or len(app.rawPhoto[0]) > app.maxWidth):
        interpolation = cv.INTER_AREA

    #Size up interpolation
    else:
        interpolation = cv.INTER_CUBIC

    #Create destination array
    app.stage = (app.maxWidth, app.maxHeight)

    #Resize image into destination array
    img = cv.resize(app.rawPhoto, app.stage, 
         interpolation)

    return  img

#Get palette#########################################################################

#Trims default 8-color palette (median cut can only do powers of 2) to
# app.paletteSize
def trimPalette(app):
    colorList = []

    #Get frequency of used colors
    for color in app.colorCounter:
        colorList.append((color, app.colorCounter[color]))

    #Sort used colors by frequency
    colorList = sorted(colorList, key=lambda element: element[1])

    #If too many colors were used, cut out the least frequent ones
    if (len(colorList) > app.paletteSize):
        difference = len(colorList) - app.paletteSize 
        colorList = colorList[:len(colorList) - difference]

    #Cut frequency data out from colorList
    colorList = [color[0] for color in colorList]

    colorSet = set(colorList)

    #Find cells in solutionGrid that used trimmed colors,
    # and fill them with the closest remaining color 
    for i in range(len(app.solutionGrid)):
        for j in range(len(app.solutionGrid[i])):

            oldColor = app.solutionGrid[i][j]

            #If trimmed color was used
            if (oldColor not in colorSet):
                #References:
                # #https://pillow.readthedocs.io/en/stable/reference/ImageColor.html
                oldColor = ImageColor.getrgb(oldColor)

                red = oldColor[0]
                green = oldColor[1]
                blue = oldColor[2]

                #Replace with closest remaining color
                closestColor = getClosestColor(red, green, blue, getRgbPalette(colorList))
                app.solutionGrid[i][j] = rgbToHex(closestColor)
    
    app.usedPalette = list(colorSet)


#Gets a palette of rgb colors from a palette of hex colors
def getRgbPalette(palette):
    rgbPalette = []

    for color in palette:
        rgbPalette.append(ImageColor.getrgb(color))

    return rgbPalette


#Gets palette using regular median cut algorithm
# (written by me!)
def getPaletteMedianCut(app):
    
    bucket = []
    for i in range(len(app.photoArray)):
        for j in range(len(app.photoArray[i])):
            pixel = app.photoArray[i][j]
            bucket.append([pixel[0], pixel[1], pixel[2]])

    palette = medianCut(np.array(bucket), 3)

    return palette


#Gets palette using module modified median cut algorithm 
# for comparison to regular median cut algorithm
#References:
# https://github.com/fengsp/color-thief-py
def getPaletteModifiedMedianCut(app, filename):
    color_thief = ColorThief(filename)
    return color_thief.get_palette(color_count=app.paletteSize)


#Get random palette#########################################################################

#Gets a random color palette using colormind.io api
#References:
# http://colormind.io/api-access/
def getRandomPalette(app):
    palette = []

    #Send request for random palette
    #References:
    # https://docs.python-requests.org/en/latest/
    # https://realpython.com/api-integration-in-python/
    url = 'http://colormind.io/api/'
    param = {"model":"default"}
    request = requests.post(url, json=param)

    #Construct palette from response
    for color in (request.json())['result']:
        palette.append(rgbToHex(color))

    if (app.paletteSize > 5):
        request = requests.post(url, json=param)
        for color in (request.json())['result']:
            palette.append(rgbToHex(color))

    #Adjust solution and answer grids to match new palette
    usedPalette = adjustGrid(app, palette)

    #Set color palette and color bucket fields to match new palette
    app.palette = palette
    app.usedPalette = set(usedPalette)
    app.colorData = placeColors(app)
    #Reset hints to match new palette
    app.hintRows, app.hintCols, app.hRSorted, app.hCSorted = getHints(app)
    syncHints(app, 0, 0)
    

#Adjusts the colors in solutionGrid and answerGrid to match the
# new color palette
def adjustGrid(app, newPalette):
    colorDict = {}
    #Will contain colors from returned palette that are
    # actually used
    newUsedPalette = []

    #Get old palette
    oldUsedPaletteList = list(app.usedPalette)

    #Map old palette colors to new palette colors
    for i in range(len(oldUsedPaletteList)):
        colorDict[oldUsedPaletteList[i]] = newPalette[i]
        newUsedPalette.append(newPalette[i])

    #Edit solutionGrid and answerGrid cell colors to match
    # new palette
    for row in range(len(app.solutionGrid)):
        for col in range(len(app.solutionGrid[row])):

            #Adjust solution grid cell
            oldColor = app.solutionGrid[row][col]
            app.solutionGrid[row][col] = colorDict[oldColor]

            #Adjust answer grid cell
            oldColor = app.answerGrid[row][col]

            if ((oldColor != None)):
                app.answerGrid[row][col] = colorDict[oldColor]

    #Return new palette
    return newUsedPalette


#Median Cut Algorithm#########################################################################

#Gets the top 2^n colors from a photo
#References:
# https://link.springer.com/referenceworkentry/10.1007%2F978-0-387-78414-4_36
# http://cis.csuohio.edu/~sschung/CIS660/Chapter3_SCMedianCutNotes.pdf
# https://en.wikipedia.org/wiki/Median_cut
# https://muthu.co/reducing-the-number-of-colors-of-an-image-using-median-cut-algorithm/
# https://github.com/fengsp/color-thief-py/blob/master/colorthief.py
# http://leptonica.org/papers/mediancut.pdf
def medianCut(bucket, target):

    #If a bucket is empty, just return it
    if (bucket == []):
        return
    #If the target depth has been reached, return the average
    # rgb value of the current bucket
    elif (target == 0):
        return [getAverage(np.array(bucket))] 
    else:
        #Sort bucket along greatest rgb channel axis
        sortedBucket = sortBucket(np.array(bucket))

        medianIndex = (len(sortedBucket) + 1) // 2

        #Divide sorted bucket along median index and recurse
        return (medianCut(sortedBucket[:medianIndex], target - 1) + 
            medianCut(sortedBucket[medianIndex:], target - 1))


#Gets the average pixel color from a bucket
def getAverage(bucket):
    #References:
    # https://muthu.co/reducing-the-number-of-colors-
    # of-an-image-using-median-cut-algorithm/
    # https://stackoverflow.com/questions/15884527/how-can-i-prevent-the-
    # typeerror-list-indices-must-be-integers-not-tuple-when-c
    blue = np.mean(bucket[:, 0])
    green = np.mean(bucket[:, 1])
    red = np.mean(bucket[:, 2])

    return [red, green, blue]


#Sorts pixels in a bucket along the greatest channel axis
def sortBucket(bucket):

    #Get rgb channel ranges
    #References:
    # https://muthu.co/reducing-the-number-of-colors-
    # of-an-image-using-median-cut-algorithm/
    redRange = np.max(bucket[:, 0]) - np.min(bucket[:, 0])
    greenRange = np.max(bucket[:, 1]) - np.min(bucket[:, 1])
    blueRange = np.max(bucket[:, 2]) - np.min(bucket[:, 2])

    #Get channel with greatest range
    if (redRange > greenRange and redRange > blueRange):
        greatestRange = 0
    elif (greenRange > redRange and greenRange > blueRange):
        greatestRange = 1
    else:
        greatestRange = 2

    #Sorts pixels along greatest channel axis
    #References:
    # https://www.w3schools.com/python/ref_func_sorted.asp
    # https://docs.python.org/3/howto/sorting.html
    sortedList = sorted(bucket, key=lambda element: element[greatestRange])

    return sortedList


#Construct grids#########################################################################

#Constructs a 2D list gameboard grid according to parameters
def getGrid(app):
    grid = []

    for row in range(app.rows):
        currentRow = []
        for col in range(app.cols):
            currentRow.append(None)

        grid.append(currentRow)

    return grid


#Puts together the puzzle solution
def getSolution(app):
    solution = []

    for row in range(app.rows):
        currentRow = []
        for col in range(app.cols):
            currentRow.append(getColor(app, row, col))

        solution.append(currentRow)

    return solution

#Color#########################################################################

#Converts rgb color values to hexadecimal color values
#Reference:
# https://docs.python.org/3/library/functions.html#hex
def rgbToHex(rgb):

    red = '%X' % int(rgb[0])
    green = '%X' % int(rgb[1])
    blue = '%X' % int(rgb[2])

    if (len(red) < 2):
        red = "0" + red

    if (len(blue) < 2):
        blue = "0" + blue

    if (len(green) < 2):
        green = "0" + green

    hex = "#" + red + green + blue

    return hex


#Gets a pixel region corresponding to a grid cell
def getCell(app, row, col):

    #Calculate cell parameters
    cellWidth = app.maxWidth / app.cols
    cellHeight = app.maxHeight / app.rows

    x1 = col * cellWidth
    y1 = row * cellHeight

    x2 = x1 + cellWidth
    y2 = y1 + cellHeight

    cx = int(x1 + (x2 - x1) // 2)
    cy = int(y1 + (y2 - y1) // 2)

    #Get pixel region
    cell = cv.getRectSubPix(app.photoArray, 
        (int(cellWidth), int(cellHeight)), (cx, cy))

    return cell


#Get closest palette color to a provided color
def getClosestColor(red, green, blue, palette):

    closestColor = None
    closestDistance = None

    #Get rid of euclidean - try chi squared, 
    #Reference:
    # https://en.wikipedia.org/wiki/Color_difference
    for color in palette:
        distance = math.sqrt( (color[0] - red)**2 + (color[1] - green)**2 
            + (color[2] - blue)**2)

        if (closestColor == None or distance < closestDistance):
            closestColor = color
            closestDistance = distance

    return closestColor


#Get average color of pixel region for a grid cell
def getColor(app, row, col):

    #Get pixel region
    cell = getCell(app, row, col)

    #Get region channel averages
    #Reference:
    # https://muthu.co/reducing-the-number-of-
    # colors-of-an-image-using-median-cut-algorithm/
    red = np.mean(cell[:, 0])
    blue = np.mean(cell[:, 1])
    green = np.mean(cell[:, 2])

    #Get closest palette color to average
    closestColor = getClosestColor(red, green, blue, app.palette)
    newColor = rgbToHex(closestColor)

    #Update number of occurances of newColor
    if (app.colorCounter.get(newColor, None) == None):
        app.colorCounter[newColor] = 1
    else:
        app.colorCounter[newColor] += 1

    #Mark color as used
    app.usedPalette.add(newColor)
    return newColor


#Handle click########################################################################

#Handle mouse drag
def mouseDragged(app, event):
    cx = event.x
    cy = event.y

    #Don't paint after win
    if (not app.isWin):
        paint(app, cx, cy, True)


#Handle mouse click
def mousePressed(app, event):
    cx = event.x
    cy = event.y

    #If the user is in the tutorial and clicks the exit button, closes the tutorial
    if (app.isTutorial and 
        (app.width - app.widthMargin + app.widthMargin / 4 < cx < app.width - app.widthMargin / 4)
        and (app.heightMargin / 4 < cy < app.heightMargin - app.heightMargin / 4)):
        app.isTutorial = False

    #If the user clicked the shuffle palette button, shuffle the palette
    elif ((app.widthMargin / 4 < cx < app.widthMargin - app.widthMargin / 4)
    and (app.heightMargin / 4 < cy < app.heightMargin - app.heightMargin / 4)):

        getRandomPalette(app)

    #If the user clicked the solve button, solves the puzzle
    elif ((app.width - app.widthMargin + app.widthMargin / 4 < cx < app.width - app.widthMargin / 4)
    and (app.heightMargin / 4 < cy < app.heightMargin - app.heightMargin / 4)):

        solve(app)

    #If the user clicked the reset button, resets the puzzle
    elif ((app.width - app.widthMargin + app.widthMargin / 4 < cx < app.width - app.widthMargin / 4)
    and (app.heightMargin / 4 + app.heightMargin < cy < 2 * app.heightMargin - app.heightMargin / 4)):

        app.isWin, app.isSolve = False, False
        app.answerGrid = getGrid(app)
        syncHints(app, 0, 0)

    #If the user clicked the menu button, goes to the main menu
    elif ((app.width - app.widthMargin + app.widthMargin / 4 < cx < app.width - app.widthMargin / 4)
    and (app.heightMargin / 4 + 2 * app.heightMargin < cy < 3 * app.heightMargin - app.heightMargin / 4)):
        menu.playGame() 

    #If the user clicked the tutorial button, opens the tutorial
    elif ((app.width - app.widthMargin + app.widthMargin / 4 < cx < app.width - app.widthMargin / 4)
    and (app.heightMargin / 4 + 3 * app.heightMargin < cy < 4 * app.heightMargin - app.heightMargin / 4)):
        app.isTutorial = True

    #Otherwise try paint. Don't paint after win
    elif (not app.isWin):
        paint(app, cx, cy, False)


#Handles painting-related clicks 
# (change brush color, paint, erase)
def paint(app, cx, cy, drag):

    #Checks if the user clicked on a paint bucket
    if (cy > app.height - app.heightMargin):

        #If so, change brush color to match paint bucket
        for color in app.colorData:
            if ((color[0] <= cx <= color[2]) and 
                (color[1] <= cy <= color[3])):
                app.brushColor = color[4]

    #Checks if the user clicked on the gameboard grid
    elif ((app.heightMargin < cy < app.height - app.heightMargin) and
        (app.widthMargin < cx < app.width - app.widthMargin)):

        #Calculate the cell that was clicked on
        cellWidth = app.maxWidth / app.cols
        cellHeight = app.maxHeight / app.rows

        row = int( (cy - app.heightMargin) // cellHeight )
        col = int( (cx - app.widthMargin) // cellWidth ) 

        #Paint a cell a new color
        if ( (app.brushColor != None) and (app.answerGrid[row][col] != app.brushColor) ):
            app.answerGrid[row][col] = app.brushColor

        #Erase a cell's color (by clicking a cell painted the same color as the brush color)
        elif ((not drag) and (app.brushColor != None) and 
            (app.answerGrid[row][col] == app.brushColor) ):
            app.answerGrid[row][col] = None

        #Check for incorrectly painted cells
        verify(app)
        
        #Update hints to reflect solved and unsolved hints
        syncHints(app, row, col)

    #Check for win state
    if (app.answerGrid == app.solutionGrid):
        app.isWin = True


#Checks for incorrectly painted cells
def verify(app):

    for i in range(app.rows):
        for j in range(app.cols):

            #If a cell is incorrectly painted, add its 
            # location to the set of errors
            if ( (app.answerGrid[i][j] != None) 
            and (app.answerGrid[i][j] != app.solutionGrid[i][j])):
                app.errors.add((i, j))

            #If a cell was incorrectly painted but has been
            # fixed, remove its location from the set of errors
            elif( (i, j) in app.errors):
                app.errors.remove((i, j))


#Draw grid##########################################################################

#Draws the gameboard grid
def drawGrid(app, canvas, grid):
    for row in range(app.rows):
        for col in range(app.cols):
            drawCell(app, canvas, grid, row, col)


#Draw a single cell on the gameboard grid
def drawCell(app, canvas, grid, row, col):
    #Gets cell fill from model
    color = grid[row][col]

    #Calculates cell dimensions
    cellWidth = app.maxWidth / app.cols
    cellHeight = app.maxHeight / app.rows

    x1 = app.widthMargin + col * cellWidth
    y1 = app.heightMargin + row * cellHeight
    x2 = x1 + cellWidth
    y2 = y1 + cellHeight

    #Draw filled cell
    if (color != None):
        canvas.create_rectangle(x1, y1, x2, y2, width = 2, fill = color)

        #Incorrect fill, highlight error cell
        if ((row, col) in app.errors):
            canvas.create_rectangle(x1 + 2, y1 + 2, x2 - 2, y2 - 2, width = 2, outline = "red")

    #Draw unfilled cell
    else:
        canvas.create_rectangle(x1, y1, x2, y2, width = 2)


#Hints##############################################################################

#Calculates puzzle hints from puzzle solution
def getHints(app):

    #Set up arrays
    hintRows, hintRowsSorted, hintCols, colHelper = [], [], [], []

    #Initializes col hint data for every col 
    # (allows column hints and row hints to be constructed simultaneously)
    for col in range(app.cols):
        hintCols.append([])

        #[currentColColor, currentColCount, startRow]
        colHelper.append([None, 0, 0])

    for row in range(app.rows):
        #Initializes row hint data
        currentRowColor, currentRowCount, startCol = None, 0, 0
    
        rowCollector = []

        for col in range(app.cols):
            cellColor = app.solutionGrid[row][col]

            #Tracks ongoing row color run
            if (cellColor == currentRowColor):
                currentRowCount += 1

            #A color run was just completed (or a row just began)
            else:
                #Append finished color run
                if (currentRowColor != None):
                    rowCollector.append([currentRowColor, currentRowCount, startCol, 1])

                #Reset row hint data
                currentRowColor, currentRowCount, startCol = cellColor, 1, col

            #If this was the last cell in the row, add the final hint to rowCollector
            if (col == app.cols - 1):
                rowCollector.append([currentRowColor, currentRowCount, startCol, 1])


            #Tracks ongoing col color run
            if (cellColor == colHelper[col][0]):
                colHelper[col][1] += 1

            #A color run was just completed (or a column just began)
            else:
                #Append finished color run
                if (colHelper[col][0] != None):
                    hintCols[col].append([colHelper[col][0], colHelper[col][1], colHelper[col][2], 1])

                #Reset column hint data
                colHelper[col][0], colHelper[col][1], colHelper[col][2] = cellColor, 1, row

            #If this was the last cell in the column, add the final hint to colHelper
            if (row == app.rows - 1):
                    hintCols[col].append([colHelper[col][0], colHelper[col][1], colHelper[col][2], 1])

        #Append original ordered rowCollector to hintRowsSorted to be used in syncHints
        hintRowsSorted.append(copy.copy(rowCollector))

        #Append randomized rowCollector to be used in drawHints
        random.shuffle(rowCollector)
        hintRows.append(rowCollector)

    #Save original ordered hintCols to hintColsSorted to be used in syncHints
    hintColsSorted = copy.deepcopy(hintCols)

    #Shuffle column hints in hintCols to be used in drawHints
    for col in hintCols:
        random.shuffle(col)

    return hintRows, hintCols, hintRowsSorted, hintColsSorted


#Shows unsolved hints and disappears solved hints
def syncHints(app, row, col):

    #Sync row hints
    for row in range(app.rows):
        for hint in app.hintRows[row]:
            #If hint is solved, hide hint
            if ((app.answerGrid[row])[hint[2]:hint[2] + hint[1]] ==
                (app.solutionGrid[row])[hint[2]:hint[2] + hint[1]]):
                hint[3] = 0
            
            #Else show hint
            else: 
                hint[3] = 1

    #Sync col hints
    for col in range(app.cols):
        for hint in app.hintCols[col]:
            answerCol = []
            solutionCol = []

            for row in range(app.rows):
                #Build columns to check hints against
                answerCol.append(app.answerGrid[row][col])
                solutionCol.append(app.solutionGrid[row][col])

            #If hint is solved hide hint
            if (answerCol[hint[2]:hint[2] + hint[1]] ==
                solutionCol[hint[2]:hint[2] + hint[1]]):
                hint[3] = 0

            #Else show hint
            else:
                hint[3] = 1


#Draws hints
def drawHints(app, canvas):

    showRowHints = []
    showColHints = []

    #Collects unsolved row hints that need to be drawn
    for row in app.hintRows:
        rowCollector = []
        for hint in row:
            if (hint[3] == 1 and app.myMode != 3):
                rowCollector.append(hint)
            elif (hint[3] == 1 and app.myMode == 3 and hint[0] == app.brushColor):
                rowCollector.append(hint)

        showRowHints.append(rowCollector)

    #Collects unsolved column hints that need to be drawn
    for col in app.hintCols:
        colCollector = []
        for hint in col:
            if (hint[3] == 1 and app.myMode != 3):
                colCollector.append(hint)
            elif (hint[3] == 1 and app.myMode == 3 and hint[0] == app.brushColor):
                colCollector.append(hint)

        showColHints.append(colCollector)


    #Calculates parameters for drawing hints
    rowHintWidth = app.widthMargin / 5
    rowHintHeight = app.maxHeight / app.rows

    colHintHeight = app.heightMargin / 4
    colHintWidth = app.maxWidth / app.cols

    #Draws row hints
    for i in range(len(showRowHints)):
        hintCount = 0

        y1 = app.heightMargin + rowHintHeight * (i + 1) - rowHintHeight // 2

        for hint in showRowHints[i]:
            x1 = app.widthMargin - rowHintWidth * hintCount - rowHintWidth // 2

            if (hint[0] == app.brushColor and app.myMode == 3):
                canvas.create_text(x1, y1, text=hint[1], fill=hint[0], font=app.smallFont)
            elif (hint[0] == app.brushColor):
                canvas.create_text(x1, y1, text=hint[1], fill=hint[0], font=app.font)
            else:
                canvas.create_text(x1, y1, text=hint[1], fill=hint[0])

            hintCount += 1
    
    #Draws column hints
    for i in range(len(showColHints)):
        hintCount = 0

        x1 = app.widthMargin + colHintWidth * (i + 1) - colHintWidth // 2

        for hint in showColHints[i]:
            y1 = app.heightMargin - colHintHeight * hintCount - colHintHeight // 2

            if (hint[0] == app.brushColor and app.myMode == 3):
                canvas.create_text(x1, y1, text=hint[1], fill=hint[0], font=app.smallFont)
            elif (hint[0] == app.brushColor):
                canvas.create_text(x1, y1, text=hint[1], fill=hint[0], font=app.font)
            else:
                canvas.create_text(x1, y1, text=hint[1], fill=hint[0])

            hintCount += 1


#Color Buckets##############################################################################

#Calculate color bucket locations
def placeColors(app):
    colorData = []

    #Calculations for multicolor palettes
    if (len(app.usedPalette) > 1):
        circleX = app.maxWidth / ( len(app.usedPalette) - 1)
        colorCount = 0    
    
    #Calculations for one-color palettes
    else:
        circleX = app.maxWidth / 2
        colorCount = 1 

    #Calculate bucket locations
    y1 = app.height - app.radius * 2
    for color in app.usedPalette:
        x1 = app.widthMargin + circleX * colorCount - app.radius / 2
        colorData.append((x1, y1, x1 + app.radius, y1 + app.radius, color))

        colorCount += 1

    return colorData


#Draws color buckets
def drawColors(app, canvas):

    for color in app.colorData:
        canvas.create_oval(color[0], color[1], color[2], color[3], fill=color[4])

#Solver#########################################################################

#Speeds up solver by making color placement deductions using hints
def deduce(app):
    canDeduce = True

    while (canDeduce):
        canDeduce = False
        unsolvedRows = []
        unsolvedCols = []

        for row in app.hintRows:
            currentRow = []
            for hint in row:
                if (hint[3] == 1):
                    currentRow.append(hint)
            unsolvedRows.append(currentRow)

        for col in app.hintCols:
            currentCol = []
            for hint in col:
                if (hint[3] == 1):
                    currentCol.append(hint)
            unsolvedCols.append(currentCol)

        #Only-one-color-left deductions
        for hints in unsolvedRows:
            if (len(hints) == 1):
                canDeduce = True
                color = hints[0][0]
                count = hints[0][1]
                start = hints[0][2]

                for i in range(count):
                    app.answerGrid[unsolvedRows.index(hints)][start + i] = color
                
        for hints in unsolvedCols:
            if (len(hints) == 1):
                canDeduce = True
                color = hints[0][0]
                count = hints[0][1]
                start = hints[0][2]

                for i in range(count):
                    app.answerGrid[start + i][unsolvedCols.index(hints)] = color

        #Size deductions
        for hints in unsolvedRows:
            for hint in hints:
                color = hint[0]
                count = hint[1]
                if (count > app.cols // 2):
                    for i in range(app.cols - (app.cols - count) * 2):
                        if (app.answerGrid[unsolvedRows.index(hints)][(app.cols - count) + i] == None):
                            canDeduce = True

                        app.answerGrid[unsolvedRows.index(hints)][(app.cols - count) + i] = color

        syncHints(app, 0, 0)


#Solves a puzzle with a part brute-force, part deduction algorithm
def solve(app):

    while (not app.isWin):

        #First, make deductions
        deduce(app)

        #Brute force next square
        coordinates = getNextSquare(app)

        if (coordinates != None):
            for color in app.usedPalette:
                if (app.answerGrid[coordinates[0]][coordinates[1]] == None):
                    app.answerGrid[coordinates[0]][coordinates[1]] = color

                    verify(app)

                    if ((coordinates[0], coordinates[1]) not in app.errors):
                        continue
                    
                    app.answerGrid[coordinates[0]][coordinates[1]] = None

        #Update hints
        syncHints(app, 0, 0)

        #Once the puzzle has been solved, end the loop
        if (app.answerGrid == app.solutionGrid):
            app.isWin = True
            app.isSolve = True

        
#Get the next unpainted square in the answer grid
def getNextSquare(app):
    for i in range(app.rows):
            for j in range(app.cols):
                if (app.answerGrid[i][j] == None):
                    return (i, j)
    

#Draw##############################################################################


#Draws game on the canvas
def redrawAll(app, canvas):
    
    #Draw gameboard, hints, color buckets
    drawGrid(app, canvas, app.answerGrid)
    drawHints(app, canvas)
    drawColors(app, canvas)

    #Draw shuffle button
    canvas.create_rectangle(app.widthMargin / 4, app.heightMargin / 4, app.widthMargin - app.widthMargin / 4,
        app.heightMargin - app.heightMargin / 4, fill="orange", outline="orange")
    canvas.create_text(app.widthMargin / 2, app.heightMargin / 2, text="Shuffle", fill="white")

    #Draw solve button
    canvas.create_rectangle( app.width - app.widthMargin + app.widthMargin / 4, 
        app.heightMargin / 4, app.width - app.widthMargin / 4,
        app.heightMargin - app.heightMargin / 4, fill="#382985", outline="#382985")
    canvas.create_text(app.width - app.widthMargin / 2, app.heightMargin / 2, text="Solve", fill="white")

    #Draw reset button
    canvas.create_rectangle( app.width - app.widthMargin + app.widthMargin / 4, 
        app.heightMargin / 4 + app.heightMargin, app.width - app.widthMargin / 4,
        2 * app.heightMargin - app.heightMargin / 4, fill="red", outline="red")
    canvas.create_text(app.width - app.widthMargin / 2, app.heightMargin / 2 + app.heightMargin, 
        text="Reset", fill="white")

    #Draw menu button
    canvas.create_rectangle( app.width - app.widthMargin + app.widthMargin / 4, 
        app.heightMargin / 4 + 2*app.heightMargin, app.width - app.widthMargin / 4,
        3 * app.heightMargin - app.heightMargin / 4, fill="blue", outline="blue")
    canvas.create_text(app.width - app.widthMargin / 2, app.heightMargin / 2 + 2*app.heightMargin, 
        text="Menu", fill="white")

    #Draw tutorial button
    canvas.create_rectangle(app.width - app.widthMargin + app.widthMargin / 4, 
        app.heightMargin / 4 + 3*app.heightMargin, app.width - app.widthMargin / 4,
        4 * app.heightMargin - app.heightMargin / 4, fill="green", outline="green")
    canvas.create_text(app.width - app.widthMargin / 2, app.heightMargin / 2 + 3*app.heightMargin, 
        text="Help", fill="white")


    #Draw tutorial
    if (app.isTutorial):
        #Background
        canvas.create_rectangle(0, 0, app.width, app.height, fill="#cce0ff", outline="#cce0ff")

        #Title
        canvas.create_text(app.width // 2, app.heightMargin // 2, text="Tutorial",
        fill="black", font=app.titleFont)

        #Instructions
        canvas.create_text(app.width // 2, app.heightMargin // 2 + app.heightMargin, 
        text="Click on one of the buckets at the bottom of the screen to select a color", fill="black",
        font=app.font)

        canvas.create_text(app.width // 2, app.heightMargin // 2 + app.heightMargin * 1.25, 
        text="Click or drag on the game grid to paint squares and reveal a hidden picture", fill="black",
        font=app.font)

        canvas.create_text(app.width // 2, app.heightMargin // 2 + app.heightMargin * 1.5, 
        text="Click a painted square with the same color to erase", fill="black",
        font=app.font)

        canvas.create_text(app.width // 2, app.heightMargin // 2 + app.heightMargin * 1.75, 
        text="Incorrectly painted squares will be highlighted in red", 
        fill="black", font=app.font)

        canvas.create_text(app.width // 2, app.heightMargin // 2 + app.heightMargin * 2.5, 
        text="There are hints for colors in both columns and rows", fill="black",
        font=app.font)

        canvas.create_text(app.width // 2, app.heightMargin // 2 + app.heightMargin * 2.75, 
        text="After clicking on a bucket, corresponding hints will be bolded", fill="black",
        font=app.font)

        canvas.create_text(app.width // 2, app.heightMargin // 2 + app.heightMargin * 3.5, 
        text="Row hints display runs of a certain color in a certain row", fill="black",
        font=app.font)

        canvas.create_text(app.width // 2, app.heightMargin // 2 + app.heightMargin * 3.75, 
        text="For example, a green 10 means that there is a run of 10 green squares in that row", 
        fill="black", font=app.font)

        canvas.create_text(app.width // 2, app.heightMargin // 2 + app.heightMargin * 4, 
        text="Column hints display runs of a certain color in a certain column", fill="black",
        font=app.font)

        canvas.create_text(app.width // 2, app.heightMargin // 2 + app.heightMargin * 4.25, 
        text="Using these hints, you can figure out what color goes in which square to solve the puzzle", 
        fill="black", font=app.font)

        canvas.create_text(app.width // 2, app.heightMargin // 2 + app.heightMargin * 4.75, 
        text="But beware! Rows with more than 5 distinct color runs and columns with more than 4", 
        fill="black", font=app.font)

        canvas.create_text(app.width // 2, app.heightMargin // 2 + app.heightMargin * 5, 
        text="distinct color runs may have hidden hints. You must solve the puzzle further to reveal these hints.", 
        fill="black", font=app.font)

        #Draw exit button
        canvas.create_rectangle( app.width - app.widthMargin + app.widthMargin / 4, 
            app.heightMargin / 4, app.width - app.widthMargin / 4,
            app.heightMargin - app.heightMargin / 4, fill="red", outline="red")
        canvas.create_text(app.width - app.widthMargin / 2, app.heightMargin / 2, text="Exit", fill="white")

        #Draw additional artist mode tutorial notes
        if (app.myMode == 3):
            canvas.create_text(app.width // 2, app.heightMargin // 2 + app.heightMargin * 6, 
        text="You are playing in artist mode! Congratulations for daring the challenge.", 
        fill="red", font=app.font)

            canvas.create_text(app.width // 2, app.heightMargin // 2 + app.heightMargin * 6.5, 
        text="Artist mode has one special feature designed to make it especially difficult:", 
        fill="red", font=app.font)
        
            canvas.create_text(app.width // 2, app.heightMargin // 2 + app.heightMargin * 6.75, 
        text="When you click on a bucket, only the hints corresponding to that color will show up", 
        fill="red", font=app.font)



    #Draw win state
    if (app.isWin and not app.isSolve):
        canvas.create_image(app.width // 2, app.height // 2, image=ImageTk.PhotoImage(app.blurredImage))

        drawGrid(app, canvas, app.emptyGrid)

        canvas.create_rectangle(app.maxWidth / 2 - app.widthMargin / 2, 
            app.height / 2 - app.heightMargin / 2, 
            app.widthMargin * 2.5 + app.maxWidth / 2, 
            app.height / 2 + app.heightMargin / 2, fill="white")

        canvas.create_text(app.width // 2, app.height // 2, text="YOU WIN!", fill="black", font=app.titleFont)

    #Draw artwork title, artist name if in artist mode
    if (app.isWin and app.myMode == 3):
        canvas.create_text(app.width / 2, 6 * (app.heightMargin / 16), text=app.json['items'][0]['titles'][0]['title'],
            fill="black", font=app.font)
        canvas.create_text(app.width / 2, 10 * (app.heightMargin / 16), text=app.json['items'][0]['artist'][0],
            fill="black", font=app.font)


def playPuzzle(file, alg, mod, artJson):
    runApp(width=800, height=850, filename=file, algorithm=alg, mode=mod, json=artJson)
    