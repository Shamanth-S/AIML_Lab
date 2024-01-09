import csv
import math
import random

def loadcsv(filename):
    with open(filename, "r") as file:
        lines = csv.reader(file)
        dataset = list(lines)[1:]
        for i in range(len(dataset)):
            dataset[i] = [float(x) for x in dataset[i]]
    return dataset

def splitDataset(dataset, splitRatio):
    trainSize = int(len(dataset) * splitRatio)
    trainSet = []
    copy = list(dataset)
    while len(trainSet) < trainSize:
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    return [trainSet, copy]

def separateByClass(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if vector[-1] not in separated:
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    return separated

def mean(numbers):
    return round(sum(numbers) / float(len(numbers)), 2)

def stdev(numbers):
    if len(numbers) <= 1:
        return 0
    avg = mean(numbers)
    variance = sum([pow(x - avg, 2) for x in numbers]) / float(len(numbers) - 1)
    return round(math.sqrt(variance), 2)

def summarize(dataset):
    summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
    del summaries[-1]
    return summaries

def summarizeByClass(dataset):
    separated = separateByClass(dataset)
    summaries = {}
    for classValue, instances in separated.items():
        summaries[classValue] = summarize(instances)
    return summaries

def calculateProbability(x, mean, stdev):
    exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
    return round((1 / (math.sqrt(2 * math.pi) * stdev)) * exponent, 2)

def calculateClassProbabilities(summaries, inputVector):
    probabilities = {}
    for classValue, classSummaries in summaries.items():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, stdev = classSummaries[i]
            x = inputVector[i]
            probabilities[classValue] *= calculateProbability(x, mean, stdev)
    return probabilities

def predict(summaries, inputVector):
    probabilities = calculateClassProbabilities(summaries, inputVector)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.items():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
    return bestLabel

def getPredictions(summaries, testSet):
    predictions = []
    for i in range(len(testSet)):
        result = predict(summaries, testSet[i])
        predictions.append(result)
    return predictions

def getAccuracy(testSet, predictions):
    correct = 0
    for i in range(len(testSet)):
        if testSet[i][-1] == predictions[i]:
            correct += 1
    return round((correct / float(len(testSet))) * 100.0, 2)

def main():
    filename = "E:\\Python\\DataSets\\_06_Sixth.csv"
    splitRatio = 0.67
    dataset = loadcsv(filename)

    print("The length of the dataset:", len(dataset))
    print("The dataset splitting into training and testing")

    trainingSet, testSet = splitDataset(dataset, splitRatio)

    print("Number of rows in the training set:", len(trainingSet))
    print("Number of rows in the testing set:", len(testSet))
    print("First five rows of the testing set:")
    for i in range(min(5, len(testSet))):
        print(testSet[i])

    summaries = summarizeByClass(trainingSet)
    print("Model summaries:")
    for classValue, classSummaries in summaries.items():
        print(classValue, ":", [(round(mean, 2), round(stdev, 2)) for mean, stdev in classSummaries])

    predictions = getPredictions(summaries, testSet)
    print("Predictions:", predictions)

    accuracy = getAccuracy(testSet, predictions)
    print("Accuracy:", accuracy, "%")

main()
