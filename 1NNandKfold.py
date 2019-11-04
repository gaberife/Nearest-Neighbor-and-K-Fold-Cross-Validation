import pandas as pd
from scipy.spatial import distance 
import timeit


# This function is changed slightly from hw04!
def readData(numRows = None):
    inputCols = ["Alcohol", "Malic Acid", "Ash", "Alcalinity of Ash", "Magnesium", "Total Phenols", "Flavanoids", "Nonflavanoid Phenols", "Proanthocyanins", "Color Intensity", "Hue", "Diluted", "Proline"]
    outputCol = 'Class'
    colNames = [outputCol] + inputCols  # concatenate two lists into one
    wineDF = pd.read_csv("data/wine.data", header=None, names=colNames, nrows = numRows)
    
    return wineDF, inputCols, outputCol

def foldsTest(trainInputDF, trainOutputSeries, testInputDF, testOutputSeries):
    print("================================")
    print("Train input:\n", list(trainInputDF.index))  # convert to list to print all contents
    print("Train output:\n", list(trainOutputSeries.index))  # convert to list to print all contents
    
    print("Test input:\n", testInputDF.index)
    print("Test output:\n", testOutputSeries.index)

    return 0


def null():
    return None
# ---------------------------------
# Given
    
def test():
    df, inputCols, outputCol = readData()
    #kFoldCVManual(3, df.loc[:, inputCols], df.loc[:, outputCol], foldsTest)
    kFoldCVManual(3, df.loc[:, inputCols], df.loc[:, outputCol], null)

    
def test05():
    df, inputCols, outputCol = readData()
    #kFoldCVManual(3, df.loc[:, inputCols], df.loc[:, outputCol], foldsTest)
    kFoldCVManual(3, df.loc[:, inputCols], df.loc[:, outputCol], null)
    
    startTime = timeit.default_timer()
    for i in range(1000):
        findNearestLoop(df.iloc[100:110, :], df.iloc[90, :])
        elapsedTime = (timeit.default_timer() - startTime)
        i+=1
    print(findNearestLoop(df.iloc[100:110, :], df.iloc[90, :]))
    print(elapsedTime)
    
    startTime = timeit.default_timer()
    for i in range(1000):
        findNearestHOF(df.iloc[100:110, :], df.iloc[90, :])
        elapsedTime = (timeit.default_timer() - startTime)
        i+=1
    print(findNearestHOF(df.iloc[100:110, :], df.iloc[90, :]))
    print(elapsedTime)
    
    #findNearestHOF is faster than findNearestLoop because we're not looping 
    #though the whole data frame and not comparing individual values 


def kFoldCVManual (k, inputDF, outputSeries, null):
    numberOfElements = inputDF.shape[0]  #finds size of the inputDF
    print("numberOfElements = " , numberOfElements)
    foldSize = numberOfElements / k  #determines the size of each fold 
    print("foldSize = " , foldSize)
    results = [] #init array to store results of folds test
    
    for i in range(k): #runs through k folds 
        start = int(i*foldSize) # iteration * size of each fold cast as int 
        print("start = " , start)
        upToNotIncluding = int((i+1)*foldSize) #gives us the boundaries of the folds
        print("upToNotIncluding = " , upToNotIncluding)

        #determins the boundaries of each fold left off from the last iteration
        testInputDF = inputDF.iloc[start:upToNotIncluding] #the input columns for the testing set
        #print("testInputDF = " , testInputDF)

        #prepare the sets to concatinate so that they can become the training set 
        #concatinate means link (things) together in a chain or series.

        #training set implementation - Confused about why we do this?????

        trainingSet = inputDF.iloc[:start,:] #everything after start 
        print("trainingSet = " , trainingSet)
        
        theRest= inputDF.iloc[upToNotIncluding:,:] #everything after upToNotIncluding
        #print("theRest = " , theRest)

        trainInputDF = pd.concat([trainingSet,theRest]) #the input columns for the training set
        #print("trainInputDF = " , trainInputDF)
        
        #testing set implementation
        testOutputSeries = outputSeries.iloc[start:upToNotIncluding] #the output column for the testing set
        #print("testOutputSeries = " , testOutputSeries)
        upperSeries= outputSeries.iloc[:start]
        #print("upperSeries = " , upperSeries)
        lowerSeries= outputSeries.iloc[upToNotIncluding:]
        #print("lowerSeries = " , lowerSeries)
        trainOutputSeries = pd.concat([upperSeries,lowerSeries]) #the output column for the training set
        #print("trainOutputSeries = " , trainOutputSeries)
        
        
        result = foldsTest(trainInputDF, trainOutputSeries, testInputDF, testOutputSeries)

        results.append(result)
    return sum(results)/len(results)
    
    
def findNearestLoop(df, testRow):
    minDist = float('inf')
    minID = 0
    for rowID in df.index:
        row = df.loc[rowID, :]
        euDistance = distance.euclidean (row, testRow)
        if minDist > euDistance :
            minDist = euDistance
            minID = rowID
    return minID


def findNearestHOF(df,testRow):
    nearestHOF = df.apply(lambda row: distance.euclidean(row, testRow),axis = 1)
    return nearestHOF.idxmin()    
    
    
    
