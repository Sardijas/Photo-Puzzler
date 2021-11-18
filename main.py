from inspect import currentframe
from PIL import Image
#Will get rid of later
from colorthief import ColorThief
import math, copy, random, time, tkinter
import cv2 as cv
# import numpy as np

from cmu_112_graphics import *

import menu

#Inspired by notes
# https://www.cs.cmu.edu/~112/notes/notes-animations-part4.html#loadImageUsingFile

# For resizing:
# https://docs.opencv.org/4.5.4/da/d54/group__imgproc__t
# ransform.html#ga47a974309e9102f5f08231edc7e7529d
# https://www.pyimagesearch.com/2021/01/20/opencv-resize-image-cv2-resize/
# https://www.tutorialspoint.com/using-opencv-with-tkinter

#Talked to greater cole
#Talked to xinyi, lauren

def appStarted(app):
    #Viewport
    app.widthMargin = app.width // 7
    app.heightMargin = app.height // 10
    app.maxWidth = app.width - 2 * app.widthMargin
    app.maxHeight = app.height - 2 * app.heightMargin

    #Photo init
    app.rawPhoto = cv.imread(app.filename)
    app.photoArray = resizeImage(app)
    app.photo = Image.fromarray(app.photoArray)

    #Palette init
    app.paletteSize = 5
    app.palette = getPalette(app, app.filename)
    app.usedPalette = set([])

    #Maybe make this controllable, max 48/48
    #Set up grid
    app.rows = 30
    app.cols = 25
    app.answerGrid = getGrid(app)
    app.solutionGrid = getSolution(app)
    app.emptyGrid = getGrid(app)

    #Final palette init
    app.radius = (app.heightMargin * (2/3)) / 2
    app.colorData = placeColors(app)


    #gradient in a pixel
    #Gaussian blur??

    #Hints init
    app.hintRows, app.hintCols, app.hRSorted, app.hCSorted = getHints(app)

    #Brush init
    app.brushColor = None

    #Error init
    app.isError = False
    app.errors = set([])

    #Win state init
    app.isWin = False
    # app.rawWinCover = cv.imread('blank.png')
    # app.winCoverArray = resizeWinCover(app, app.rawWinCover)
    app.blurredImageArray = blurImage(app)
    app.blurredImage = Image.fromarray(app.blurredImageArray)
    app.titleFont = tkinter.font.Font(family='Helvetica', size=48, weight='bold')

#Init##########################################################################

#https://docs.opencv.org/3.4/d4/d13/tutorial_py_filtering.html
def blurImage(app):
    #Create destination array

    # resizedOriginal = resizeWinCover(app, app.photoArray)

    # alphaImage = cv.addWeighted(app.winCoverArray, 0.2, resizedOriginal, 0.6, 0, app.stage)
    rgbImage = cv.cvtColor(app.photoArray, cv.COLOR_BGR2RGBA)
    blurredImage = cv.GaussianBlur(rgbImage, (21, 21), 0)

    # for row in blurredImage:
    #     for pixel in row:
    #         pixel.append(0.5)

    #     print(row)

    return blurredImage

# #Functionality tho
# def resizeWinCover(app, toResize):
#     #Size up??
#     if (len(toResize) > app.height
#         or len(toResize) > app.width):
#         interpolation = cv.INTER_AREA

#     #Size down
#     else:
#         interpolation = cv.INTER_CUBIC

#     #Create destination array
#     app.stage = (app.width, app.height)

#     #Resize image into destination array
#     img = cv.resize(toResize, app.stage, 
#          interpolation)

#     return  img

def resizeImage(app):

    #Size up???
    if (len(app.rawPhoto) > app.maxHeight 
        or len(app.rawPhoto[0]) > app.maxWidth):
        interpolation = cv.INTER_AREA

    #Size down
    else:
        interpolation = cv.INTER_CUBIC

    #Create destination array
    app.stage = (app.maxWidth, app.maxHeight)

    #Resize image into destination array
    img = cv.resize(app.rawPhoto, app.stage, 
         interpolation)

    return  img

#Goal: replace w/ my own algorithm
#https://github.com/fengsp/color-thief-py
#https://muthu.co/reducing-the-number-of-colors-of-an-image-using-median-cut-algorithm/
def getPalette(app, filename):

    # bucket = []
    # for row in app.photoArray:
    #     bucket.extend(row)

    # num = len(app.photoArray[0])

    # yes = medianCut(bucket, 0, 4)

    # return yes

    color_thief = ColorThief(filename)
    return color_thief.get_palette(color_count=app.paletteSize)

#http://leptonica.org/papers/mediancut.pdf
def medianCut(bucket, count, target):

    if (2**count == target):
        return [getAverage(bucket)]
    elif (bucket == []):
        return 0
    else:
        sortedBucket = sortBucket(bucket)

        medianIndex = ( len(sortedBucket) + 1) // 2

        return (medianCut(bucket[:medianIndex], count + 1, target) + 
            medianCut(bucket[medianIndex:], count + 1, target))


def getAverage(bucket):
    red = 0
    green = 0
    blue = 0

    count = 0

    for pixel in bucket:
        red += pixel[0]
        green += pixel[1]
        blue += pixel[2]

        count += 1

    red //= count
    green //= count
    blue //= count

    return [blue, green, red]


def sortBucket(bucket):

    red = [None, None]
    green = [None, None]
    blue = [None, None]

    #Longer but more efficient than list comprehensions
    for pixel in bucket:
        if (red[0] == None):
            red[0] = pixel[0]
            red[1] = pixel[0]
        else:
            if (pixel[0] < red[0]):
                red[0] = pixel[0]
            elif (pixel[0] > red[1]):
                red[1] = pixel[0]

        if (green[0] == None):
            green[0] = pixel[1]
            green[1] = pixel[1]
        else:
            if (pixel[1] < green[0]):
                green[0] = pixel[1]
            elif (pixel[1] > green[1]):
                green[1] = pixel[1]

        if (blue[0] == None):
            blue[0] = pixel[2]
            blue[1] = pixel[2]
        else:
            if (pixel[2] < blue[0]):
                blue[0] = pixel[2]
            elif (pixel[2] > blue[1]):
                blue[1] = pixel[2]

    redRange = red[1] - red[0]
    greenRange = green[1] - green[0]
    blueRange = blue[1] - blue[0]

    if (redRange > greenRange and redRange > blueRange):
        greatestRange = 0
    elif (greenRange > redRange and greenRange > blueRange):
        greatestRange = 1
    else:
        greatestRange = 2

    #https://www.w3schools.com/python/ref_func_sorted.asp
    #https://docs.python.org/3/howto/sorting.html
    sortedList = sorted(bucket, key=lambda element: element[greatestRange])

    return sortedList

def getGrid(app):
    grid = []

    for row in range(app.rows):
        currentRow = []
        for col in range(app.cols):
            currentRow.append(None)

        grid.append(currentRow)

    return grid

def getSolution(app):
    solution = []

    for row in range(app.rows):
        currentRow = []
        for col in range(app.cols):
            currentRow.append(getColor(app, row, col))

        solution.append(currentRow)

    return solution


def getColor(app, row, col):
    cellWidth = app.maxWidth / app.cols
    cellHeight = app.maxHeight / app.rows

    x1 = col * cellWidth
    y1 = row * cellHeight

    x2 = x1 + cellWidth
    y2 = y1 + cellHeight

    cx = int(x1 + (x2 - x1) // 2)
    cy = int(y1 + (y2 - y1) // 2)

    cell = cv.getRectSubPix(app.photoArray, 
        (int(cellWidth), int(cellHeight)), (cx, cy))

    blue = 0
    green = 0
    red = 0

    count = 0

    #numpy for pixel averages - efficient could be complex
    for row in range(len(cell)):
        for col in range(len(cell[0])):
            pixel = cell[row][col]
            blue += pixel[0]
            green += pixel[1]
            red += pixel[2]

            count += 1

    blue //= count
    green //= count
    red //= count

    closestColor = None
    closestDistance = None

    #Get rid of euclidean - try chi squared, 
    #https://en.wikipedia.org/wiki/Color_difference
    for color in app.palette:
        distance = math.sqrt( (color[0] - red)**2 + (color[1] - green)**2 
            + (color[2] - blue)**2)

        if (closestColor == None or distance < closestDistance):
            closestColor = color
            closestDistance = distance

    newColor = rgbToHex(closestColor)
    app.usedPalette.add(newColor)
    return newColor


def getHints(app):

    hintRows = []
    hintRowsSorted = []
    hintCols = []
    hintColsSorted = []

    colHelper = []
    
    for col in range(app.cols):
        hintCols.append([])
        #Appends currentColColor, currentColCount, startRow
        colHelper.append([None, 0, 0])

    for row in range(app.rows):
        currentRowColor = None
        currentRowCount = 0
        startCol = 0
    
        rowCollector = []

        for col in range(app.cols):
            cellColor = app.solutionGrid[row][col]

            if (cellColor == currentRowColor):
                currentRowCount += 1
                if (col == app.cols - 1):
                    rowCollector.append([currentRowColor, currentRowCount, startCol, 1])

            else:
                if (currentRowColor != None):
                    rowCollector.append([currentRowColor, currentRowCount, startCol, 1])

                currentRowColor = cellColor
                currentRowCount = 1
                startCol = col

            if (cellColor == colHelper[col][0]):
                colHelper[col][1] += 1
                if (row == app.rows - 1):
                    hintCols[col].append([colHelper[col][0], colHelper[col][1], colHelper[col][2], 1])

            else:
                if (colHelper[col][0] != None):
                    hintCols[col].append([colHelper[col][0], colHelper[col][1], colHelper[col][2], 1])

                colHelper[col][0] = cellColor
                colHelper[col][1] = 1
                colHelper[col][2] = row

        hintRowsSorted.append(copy.copy(rowCollector))

        random.shuffle(rowCollector)
        hintRows.append(rowCollector)

    hintColsSorted = copy.deepcopy(hintCols)
    for col in hintCols:
        random.shuffle(col)

    return hintRows, hintCols, hintRowsSorted, hintColsSorted

def getOddHint(row):
    hintPriorityList = sorted(row, key=lambda element: element[1])
    return hintPriorityList[0]

def trimHints(app):
    problemRows = []
    problemCols = []

    for i in range(len(app.hintRows)):
        if (len(app.hintRows[i]) > 5):
            oddRowCopy = copy.copy(app.hintRows[i])
            for j in range(len(app.hintRows[i]) - 5):
                oddHint = getOddHint(oddRowCopy)
                problemRows.append((i, oddHint))
                oddRowCopy.remove(oddHint)

    
    for i in range(len(app.hintCols)):
        if (len(app.hintCols[i]) > 5):
            oddColCopy = copy.copy(app.hintCols[i])
            for j in range(len(app.hintCols[i]) - 5):
                oddHint = getOddHint(oddColCopy)
                problemCols.append((i, oddHint))
                oddColCopy.remove(oddHint)

    # for problemHint in problemRows:
    #     #My thought is change to a neighboring color that is in both row and col



#Model#########################################################################


#https://docs.python.org/3/library/functions.html#hex
def rgbToHex(rgb):

    red = '%X' % rgb[0]
    green = '%X' % rgb[1]
    blue = '%X' % rgb[2]

    if (len(red) < 2):
        red = "0" + red

    if (len(blue) < 2):
        blue = "0" + blue

    if (len(green) < 2):
        green = "0" + green

    hex = "#" + red + green + blue

    return hex

def placeColors(app):
    colorData = []

    if (len(app.usedPalette) > 1):
        circleX = app.maxWidth / ( len(app.usedPalette) - 1)
        colorCount = 0    
    else:
        circleX = app.maxWidth / 2
        colorCount = 1 


    y1 = app.height - app.radius * 2
    for color in app.usedPalette:
        x1 = app.widthMargin + circleX * colorCount - app.radius / 2
        colorData.append((x1, y1, x1 + app.radius, y1 + app.radius, color))

        colorCount += 1

    return colorData

def verify(app):
    for i in range(app.rows):
        for j in range(app.cols):

            if ( (app.answerGrid[i][j] != None) and (app.answerGrid[i][j] != app.solutionGrid[i][j])):
                app.errors.add((i, j))

            elif( (i, j) in app.errors):
                app.errors.remove((i, j))

    if (len(app.errors) == 0):
        app.isError = False
    else:
        app.isError = True

def paint(app, cx, cy, drag):
    if (cy > app.height - app.heightMargin):

        for color in app.colorData:
            if ((color[0] <= cx <= color[2]) and 
                (color[1] <= cy <= color[3])):
                app.brushColor = color[4]


    elif ((app.heightMargin < cy < app.height - app.heightMargin) and
        (app.widthMargin < cx < app.width - app.widthMargin)):

        cellWidth = app.maxWidth / app.cols
        cellHeight = app.maxHeight / app.rows

        row = int( (cy - app.heightMargin) // cellHeight )
        col = int( (cx - app.widthMargin) // cellWidth ) 

        if ( (app.brushColor != None) and (app.answerGrid[row][col] != app.brushColor) ):
            app.answerGrid[row][col] = app.brushColor
        elif ((not drag) and (app.brushColor != None) and 
            (app.answerGrid[row][col] == app.brushColor) ):
            app.answerGrid[row][col] = None

        verify(app)
        if (not (row, col) in app.errors):
            syncHints(app, app.brushColor, row, col)

    if (app.answerGrid == app.solutionGrid):
        app.isWin = True

#Can't tell if you overwrite something
def syncHints(app, color, row, col):
    for row in range(app.rows):
        for hint in app.hintRows[row]:
            if ((app.answerGrid[row])[hint[2]:hint[2] + hint[1]] ==
                (app.solutionGrid[row])[hint[2]:hint[2] + hint[1]]):
                hint[3] = 0
            else: 
                hint[3] = 1

    for col in range(app.cols):
        for hint in app.hintCols[col]:
            answerCol = []
            solutionCol = []
            for row in range(app.rows):
                answerCol.append(app.answerGrid[row][col])
                solutionCol.append(app.solutionGrid[row][col])

            if (answerCol[hint[2]:hint[2] + hint[1]] ==
                solutionCol[hint[2]:hint[2] + hint[1]]):
                hint[3] = 0
            else:
                hint[3] = 1

#Events########################################################################

def mouseDragged(app, event):
    cx = event.x
    cy = event.y

    if (not app.isWin):
        paint(app, cx, cy, True)

def mousePressed(app, event):
    cx = event.x
    cy = event.y

    if (not app.isWin):
        paint(app, cx, cy, False)


#Draw##########################################################################


def drawGrid(app, canvas, grid):
    for row in range(app.rows):
        for col in range(app.cols):
            drawCell(app, canvas, grid, row, col)


def drawCell(app, canvas, grid, row, col):
    color = grid[row][col]

    cellWidth = app.maxWidth / app.cols
    cellHeight = app.maxHeight / app.rows

    x1 = app.widthMargin + col * cellWidth
    y1 = app.heightMargin + row * cellHeight
    x2 = x1 + cellWidth
    y2 = y1 + cellHeight

    if (color != None):
        canvas.create_rectangle(x1, y1, x2, y2, width = 2, fill = color)

        if ((row, col) in app.errors):
            canvas.create_rectangle(x1 + 2, y1 + 2, x2 - 2, y2 - 2, width = 2, outline = "red")

    else:
        canvas.create_rectangle(x1, y1, x2, y2, width = 2)


#Doesn't work well at all for photos with lots of '1' patterns
def drawHints(app, canvas):

    showRowHints = []
    longestRow = None
    showColHints = []
    longestCol = None

    for row in app.hintRows:
        rowCollector = []
        for hint in row:
            if (hint[3] == 1):
                rowCollector.append(hint)

        if (longestRow == None or longestRow < len(rowCollector)):
            longestRow = len(rowCollector)

        showRowHints.append(rowCollector)

    for col in app.hintCols:
        colCollector = []
        for hint in col:
            if (hint[3] == 1):
                colCollector.append(hint)

        if (longestCol == None or longestCol < len(colCollector)):
            longestCol = len(colCollector)

        showColHints.append(colCollector)


    if (longestRow != None and longestRow != 0):
        rowHintWidth = app.widthMargin / longestRow
    else:
        rowHintWidth = 0

    rowHintHeight = app.maxHeight / app.rows

    if (longestCol != None and longestCol != 0):
        colHintHeight = app.heightMargin / longestCol
    else:
        colHintHeight = 0
    
    colHintWidth = app.maxWidth / app.cols

    for i in range(len(showRowHints)):
        hintCount = 0
        y1 = app.heightMargin + rowHintHeight * (i + 1) - rowHintHeight // 2
        for hint in showRowHints[i]:
            x1 = app.widthMargin - rowHintWidth * hintCount - rowHintWidth // 2

            canvas.create_text(x1, y1, text=hint[1], fill=hint[0])

            hintCount += 1
    
    for i in range(len(showColHints)):
        hintCount = 0
        x1 = app.widthMargin + colHintWidth * (i + 1) - colHintWidth // 2
        for hint in showColHints[i]:
            y1 = app.heightMargin - colHintHeight * hintCount - colHintHeight // 2

            canvas.create_text(x1, y1, text=hint[1], fill=hint[0])

            hintCount += 1

def drawColors(app, canvas):

    for color in app.colorData:
        canvas.create_oval(color[0], color[1], color[2], color[3], fill=color[4])


def redrawAll(app, canvas):
    canvas.create_image(app.width // 2, app.height // 2, image=ImageTk.PhotoImage(app.photo))
    
    drawGrid(app, canvas, app.answerGrid)
    drawHints(app, canvas)
    drawColors(app, canvas)

    if (app.isWin):
        canvas.create_image(app.width // 2, app.height // 2, image=ImageTk.PhotoImage(app.blurredImage))

        drawGrid(app, canvas, app.emptyGrid)

        canvas.create_rectangle(app.maxWidth / 2 - app.widthMargin / 2, 
            app.height / 2 - app.heightMargin / 2, 
            app.widthMargin * 2.5 + app.maxWidth / 2, 
            app.height / 2 + app.heightMargin / 2, fill="white")

        canvas.create_text(app.width // 2, app.height // 2, text="YOU WIN!", fill="black", font=app.titleFont)

def playPuzzle(file):
    runApp(width=800, height=850, filename=file)
    





#Notes
#win-state: gridlines disappear? de-pixelize to become regular picture 
# - blur? simulate process would be complex w/ gaussian, maybe 5 or 6 filters
#Maybe get grid and things to resize and move
#Goal: solver by MVP (even bad solver)
#Clues disappear - on click? automatically?




#Check numpy to make things quicker, loop through board and check neigbors