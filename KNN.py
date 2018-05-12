from collections import Counter
import unicodecsv
import numpy as np


class KNN:

    def getData(self, filename):
        with open(filename,'rb') as f:
            reader = unicodecsv.reader(f)
            i_data = list(reader)
            return i_data

    # def getTrainData(self, filename):
    #     with open(filename, 'rb') as f:
    #         reader = unicodecsv.reader(f)
    #         train_data = list(reader)
    #         label_data = []
    #         for i in train_data:
    #             label_data.append(i[-1])
    #             del i[-1]
    #     return train_data, label_data

    def splitAttributeLabels(self,dataset):
        label_data = []
        for i in dataset:
            label_data.append(i[-1])
            del i[-1]
        return dataset, label_data

    def fold_stratified(self, train_data):
        train_data_yes = []
        train_data_no = []
        folds_data_yes = []
        folds_data_no = []
        for i in train_data:
            if i[8] == 'yes':
                train_data_yes.append(i)
            else:
                train_data_no.append(i)
        fold_sizes = (len(train_data) // 10) * np.ones(10, dtype=np.int)
        fold_sizes[:len(train_data) % 10] += 1
        for i in range(len(fold_sizes)):
            if i == 0:
                folds_data_yes.append(train_data_yes[:fold_sizes[i]])
            else:
                folds_data_yes.append(train_data_yes[fold_sizes[i-1]:fold_sizes[i-1]+fold_sizes[i]])
        for i in range(len(fold_sizes)):
            if i == 0:
                folds_data_no.append(train_data_no[:fold_sizes[i]])
            else:
                folds_data_no.append(train_data_no[fold_sizes[i-1]:fold_sizes[i-1]+fold_sizes[i]])
        return folds_data_yes,folds_data_no

    def getFoldsData(self, filename):
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
            return dataset

    def cross_validation(self,k,filrname):
        accurancies = []
        for i in range(10):
            dataset = self.getFoldsData(filrname)
            test_data = dataset[i]
            dataset.remove(test_data)
            train_dataset = []
            for folds in dataset:
                for instances in folds:
                    train_dataset.append(instances)
            train_data, train_label_data = self.splitAttributeLabels(train_dataset)
            test_set, test_label_data = self.splitAttributeLabels(test_data)
            results = self.knn_predict(train_data,train_label_data,test_data,k)
            # results = self.knn_predict_2(train_data, train_label_data, test_data, k)
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
    # def accurancy(self, results):
    #     correct = 0
    #     for i in results:
    #         if i[8] == i[9]:
    #             correct += 1
    #     accurancy = float(correct) / len(results) * 100
    #     return accurancy

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
            neighbors = self.get_neighbors(training_set,labels, test_set[i], k)
            results.append(self.vote(neighbors))
        return results

    # def knn_predict_2(self, training_set, labels, test_set, k):
    #     for i in range(len(test_set)):
    #         neighbors = self.get_neighbors(training_set,labels, test_set[i], k)
    #         i.append(self.vote(neighbors))
    #     return test_set

    # def euclideanDist(self,x,xi):
    #     d = 0.0
    #     for i in range(len(x) - 1):
    #         d += pow((float(xi[i]) - float(x[i])), 2)  # euclidean distance
    #     d = math.sqrt(d)
    #     return d
    # def knn_predict(self, test_data, train_data, k_value):
    #     for i in test_data:
    #         eu_Distance = []
    #         yes = 0
    #         no = 0
    #         # print(i[0])
    #         for j in train_data:
    #             eu_dist = self.euclideanDist(i, j)
    #             eu_Distance.append((j[8], eu_dist))
    #             eu_Distance.sort(key=operator.itemgetter(1))
    #             k_nn = eu_Distance[:k_value]
    #             for k in k_nn:
    #                 if k[0] == 'yes':
    #                     yes += 1
    #                 else:
    #                     no += 1
    #         if yes > no:
    #             i.append('yes')
    #         elif yes < no:
    #             i.append('no')
    #         else:
    #             i.append('NaN')
    #
    # def getNeighbors(self, trainingSet,testInstance, k):
    #     eu_Distance = []
    #     for x in range(len(trainingSet)):
    #         dist = self.distance(testInstance, trainingSet[x])
    #         eu_Distance.append((trainingSet[x], dist))
    #     eu_Distance.sort(key=operator.itemgetter(1))
    #     # neighbors = eu_Distance[:k]
    #     neighbors = []
    #     for x in range(k):
    #         neighbors.append(eu_Distance[x][0])
    #     return neighbors
    #
    # def getResponse(self, neighbors):
    #     classVotes = {}
    #     classVotes['yes'] = 0
    #     classVotes['no'] = 0
    #     for x in range(len(neighbors)):
    #         response = neighbors[x][-1]
    #         if response == 'yes':
    #             classVotes['yes'] += 1
    #         else:
    #             classVotes['no'] += 1
    #     sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    #     return sortedVotes[0][0]
    #
    # def predict(self, X_train, y_train, x_test, k):
    #     # create list for distances and targets
    #     distances = []
    #     targets = []
    #     for i in range(len(X_train)):
    #         distance = np.sqrt(np.sum(np.square(x_test - X_train[i,:])))
    #         # add it to list of distances
    #         distances.append([distance, i])
    #     distances = sorted(distances)
    #     # make a list of the k neighbors' targets
    #     for i in range(k):
    #         index = distances[i][1]
    #         targets.append(y_train[index])
    #     # return most common target
    #     return Counter(targets).most_common(1)[0][0]
    #
    # def kNearestNeighbor(self, X_train, y_train, X_test, predictions, k):
    #     # loop over all observations
    #     for i in range(len(X_test)):
    #         predictions.append(self.predict(X_train, y_train, X_test[i,:], k))



