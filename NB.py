import unicodecsv
import math
import numpy as np


class NB:
    def getTrainData(self, filename):
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

    def getTestData(self, filename):
        with open(filename, 'rb') as f:
            reader = unicodecsv.reader(f)
            i_data = list(reader)
            for i in range(len(i_data)):
                i_data[i] = [float(x) for x in i_data[i]]
            return i_data

    def splitAttributeLabels(self,dataset):
        label_data = []
        for i in dataset:
            label_data.append(i[-1])
            del i[-1]
        return dataset, label_data

    def mean(self, numbers):
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
        return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent

    def calculateClassProbabilities(self,summaries, inputVector, prior):
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

    def get_stratification_fold(self, dataset):
        seperated = self.separateByClass(dataset)
        # yes_folds_sizes = (len(seperated[1]) // 10) * np.ones(10, dtype=np.int)
        # yes_folds_sizes[:len(seperated[1]) % 10] += 1
        # no_folds_sizes = (len(seperated[0]) // 10) * np.ones(10, dtype=np.int)
        # no_folds_sizes[:len(seperated[0]) % 10] += 1
        yes_folds = []
        no_folds = list(list(a) for a in zip(*[iter(seperated[0])]*50))
        yes_folds.append(seperated[1][:27])
        yes_folds.append(seperated[1][27:54])
        yes_folds.append(seperated[1][54:81])
        yes_folds.append(seperated[1][81:108])
        yes_folds.append(seperated[1][108:135])
        yes_folds.append(seperated[1][135:162])
        yes_folds.append(seperated[1][162:189])
        yes_folds.append(seperated[1][189:216])
        yes_folds.append(seperated[1][216:242])
        yes_folds.append(seperated[1][242:268])
        folds = []
        for i in range(10):
            folds.append(yes_folds[i] + no_folds[i])
        return folds

    def cross_validation(self,filename):
        accurancies = []
        for i in range(10):
            traindata = self.getTrainData(filename)
            dataset = self.get_stratification_fold(traindata)
            test_dataset = dataset[i]
            dataset.remove(test_dataset)
            train_dataset = []
            for folds in dataset:
                for instances in folds:
                    train_dataset.append(instances)
            results = self.getPredictions(test_dataset,train_dataset)
            accurancies.append(self.accurancy(results))
        average_accurancy = sum(accurancies) / float(len(accurancies))
        return average_accurancy