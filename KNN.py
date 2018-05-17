from collections import Counter
import unicodecsv
import numpy as np


class KNN:

    def getData(self, filename):
        with open(filename, 'rb') as f:
            reader = unicodecsv.reader(f)
            i_data = list(reader)
            return i_data

    def splitAttributeLabels(self, dataset):
        label_data = []
        for i in dataset:
            label_data.append(i[-1])
            del i[-1]
        return dataset, label_data

    def writeStratification(self,from_filename,to_filename):
        foldsdata = self.getData(from_filename)
        dataset = self.get_stratification_fold(foldsdata)
        with open(to_filename, "wb") as csvfile:
            writer = unicodecsv.writer(csvfile)
            for i in dataset:
                writer.writerow(["fold"+str(dataset.index(i)+1)])
                for a in i:
                    writer.writerow(a)
                writer.writerow("")

    def separateByClass(self, dataset):
        separated = {}
        for i in range(len(dataset)):
            vector = dataset[i]
            if (vector[-1] not in separated):
                separated[vector[-1]] = []
            separated[vector[-1]].append(vector)
        return separated

    def get_stratification_fold(self, dataset):
        seperated = self.separateByClass(dataset)
        # prior = self.prior(dataset)
        # yes_folds_sizes = (len(seperated[1]) // 10) * np.ones(10, dtype=np.int)
        # yes_folds_sizes[:len(seperated[1]) % 10] += 1
        # no_folds_sizes = (len(seperated[0]) // 10) * np.ones(10, dtype=np.int)
        # no_folds_sizes[:len(seperated[0]) % 10] += 1
        yes_folds = []
        no_folds = list(list(a) for a in zip(*[iter(seperated['no'])]*50))
        yes_folds.append(seperated['yes'][:27])
        yes_folds.append(seperated['yes'][27:54])
        yes_folds.append(seperated['yes'][54:81])
        yes_folds.append(seperated['yes'][81:108])
        yes_folds.append(seperated['yes'][108:135])
        yes_folds.append(seperated['yes'][135:162])
        yes_folds.append(seperated['yes'][162:189])
        yes_folds.append(seperated['yes'][189:216])
        yes_folds.append(seperated['yes'][216:242])
        yes_folds.append(seperated['yes'][242:268])
        folds = []
        for i in range(10):
            folds.append(yes_folds[i] + no_folds[i])
        return folds


    def cross_validation(self,k,filename):
        accurancies = []
        for i in range(10):
            foldsdata = self.getData(filename)
            dataset = self.get_stratification_fold(foldsdata)
            test_data = dataset[i]
            dataset.remove(test_data)
            train_dataset = []
            for folds in dataset:
                for instances in folds:
                    train_dataset.append(instances)
            train_data, train_label_data = self.splitAttributeLabels(train_dataset)
            test_set, test_label_data = self.splitAttributeLabels(test_data)
            results = self.knn_predict(train_data,train_label_data,test_data,k)
            accurancies.append(self.accurancy(results,test_label_data))
        average_accurancy = sum(accurancies) / float(len(accurancies))
        return average_accurancy

    def accurancy(self, results, test_label_data):
        correct = 0
        compare = zip(*(results,test_label_data))
        for i in compare:
            if i[0] == i[1]:
                correct += 1
        accurancy = float(correct) / len(results) * 100
        return accurancy

    def distance(self, instance1, instance2):
        # just in case, if the instances are lists or tuples:
        instance1 = np.array(instance1, dtype=float)
        instance2 = np.array(instance2, dtype=float)
        return np.linalg.norm(instance1 - instance2)

    def get_neighbors(self, training_set,labels, test_instance, k):
        distances = []
        for index in range(len(training_set)):
            dist = self.distance(test_instance, training_set[index])
            distances.append((training_set[index], dist, labels[index]))
        distances.sort(key=lambda x: x[1])
        neighbors = distances[:k]
        return neighbors

    def vote(self, neighbors):
        class_counter = Counter()
        for neighbor in neighbors:
            class_counter[neighbor[2]] += 1
        return class_counter.most_common(1)[0][0]

    def knn_predict(self, training_set, labels, test_set, k):
        results = []
        for i in range(len(test_set)):
            neighbors = self.get_neighbors(training_set, labels, test_set[i], k)
            results.append(self.vote(neighbors))
        return results





