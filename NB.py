import unicodecsv
import math
import numpy as np


class NB:
    def getData(self, filename):
        with open(filename, 'rb') as f:
            reader = unicodecsv.reader(f)
            i_data = list(reader)
            for i in i_data:
                if i[-1] == 'yes':
                        i[-1] = 1
                else:
                        i[-1] = 0
            for i in range(len(i_data)):
                i_data[i] = [float(x) for x in i_data[i]]
            return i_data

    def processNBData(self,daatset):
        for i in daatset:
            if i[-1] == 'yes':
                i[-1] = 1
            else:
                i[-1] = 0
        for i in range(len(daatset)):
            daatset[i] = [float(x) for x in daatset[i]]
        return daatset

    def splitAttributeLabels(self,dataset):
        label_data = []
        for i in dataset:
            label_data.append(i[-1])
            del i[-1]
        return dataset, label_data

    def getFoldsData(self,filename):
        with open(filename,'rb') as f:
            reader = unicodecsv.reader(f)
            i_data = list(reader)
            dataset = []
            fold1 = i_data[1:78]
            fold2 = i_data[80:157]
            fold3 = i_data[159:236]
            fold4 = i_data[238:315]
            fold5 = i_data[317:394]
            fold6 = i_data[396:473]
            fold7 = i_data[475:552]
            fold8 = i_data[554:631]
            fold9 = i_data[633:709]
            fold10 = i_data[711:787]
            dataset.append(fold1)
            dataset.append(fold2)
            dataset.append(fold3)
            dataset.append(fold4)
            dataset.append(fold5)
            dataset.append(fold6)
            dataset.append(fold7)
            dataset.append(fold8)
            dataset.append(fold9)
            dataset.append(fold10)
            # for i in dataset:
            #     if i[8] == 'yes':
            #         i[8] = 1
            #     else:
            #         i[8] = 0
            # for j in range(len(dataset)):
            #     for i in dataset[j]:
            #         dataset[j][i] = [float(x) for x in dataset[j][i]]
            return dataset

    def mean(self, numbers):
        numbers
        return sum(numbers) / float(len(numbers))


    def stdev(self, numbers):
        avg = self.mean(numbers)
        variance = sum([pow(x - avg, 2) for x in numbers]) / float(len(numbers) - 1)
        return math.sqrt(variance)

    def separateByClass(self, dataset):
        separated = {}
        for i in range(len(dataset)):
            vector = dataset[i]
            if (vector[-1] not in separated):
                separated[vector[-1]] = []
            separated[vector[-1]].append(vector)
        return separated

    def summarize(self, dataset):
        summaries = [(np.mean(attribute), np.std(attribute)) for attribute in zip(*dataset)]
        del summaries[-1]
        return summaries

    def summarizeByClass(self,dataset):
        separated = self.separateByClass(dataset)
        summaries = {}
        for classValue, instances in separated.items():
            summaries[classValue] = self.summarize(instances)
        return summaries

    def calculateProbability(self, x, mean, stdev):
        exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
        # np.random.normal()
        return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent

    def calculateClassProbabilities(self,summaries, inputVector,prior):
        probabilities = {}
        for classValue, classSummaries in summaries.items():
            probabilities[classValue] = 1
            for i in range(len(classSummaries)):
                mean, stdev = classSummaries[i]
                x = inputVector[i]
                probabilities[classValue] *= self.calculateProbability(x, mean, stdev)
            p = prior[classValue]
            probabilities[classValue] = probabilities[classValue] * p
        return probabilities

    def prior(self,trainSet):
        prior = {}
        n = len(trainSet)
        yes_count = 0
        no_count = 0
        for i in trainSet:
            if i[-1] == 1:
                yes_count += 1
            else:
                no_count += 1
        prior[1.0] = (yes_count/n)
        prior[0.0] = (no_count/n)
        return prior

    def getPredictions(self,testSet,trainSet):
        summaries = self.summarizeByClass(trainSet)
        prior = self.prior(trainSet)
        for i in testSet:
            probabilities = self.calculateClassProbabilities(summaries, i, prior)
            bestLabel, bestProb = None, -1
            for classValue, probability in probabilities.items():
                if bestLabel is None or probability > bestProb:
                    bestProb = probability
                    bestLabel = classValue
            i.append(bestLabel)
        return testSet

    def accurancy(self, results):
        correct = 0
        for i in results:
            n = len(i)
            if i[n-2] == i[n-1]:
                correct += 1
        accurancy = float(correct) / len(results) * 100
        return accurancy

    def cross_validation(self,filename):
        accurancies = []
        for i in range(10):
            dataset = self.getFoldsData(filename)
            test_dataset = dataset[i]
            dataset.remove(test_dataset)
            train_dataset = []
            for folds in dataset:
                for instances in folds:
                    train_dataset.append(instances)
            train_data = self.processNBData(train_dataset)
            test_data = self.processNBData(test_dataset)
            results = self.getPredictions(train_data,test_data)
            accurancies.append(self.accurancy(results))
        average_accurancy = sum(accurancies) / float(len(accurancies))
        return average_accurancy