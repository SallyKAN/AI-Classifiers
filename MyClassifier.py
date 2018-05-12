import sys
from KNN import KNN
from NB import NB


def main():
    train_filename = sys.argv[1]
    test_filename = sys.argv[2]
    if sys.argv[3][1] == 'N':
        k_value = int(sys.argv[3][0])
        knn = KNN()
        train_dataset = knn.getData(train_filename)
        test_data = knn.getData(test_filename)
        train_data, label_data = knn.splitAttributeLabels(train_dataset)
        # results = knn.knn_predict(train_data, label_data, test_data, k_value)
        # for i in results:
        #     print(i)
        print(knn.cross_validation(k_value, 'pima-CFS-folds.csv'))

    else:
        nb = NB()
        train_data = nb.getData(train_filename)
        test__data = nb.getData(test_filename)
        result = nb.getPredictions(test__data, train_data)
        # separated = nb.separateByClass(train_data)
        # print(separated[1][separated[1]=='yes'].count())
        # print(result[20][8],result[21][8],result[22][8])
        print(nb.cross_validation('pima-folds.csv'))
        # for i in range(len(result)):
        #     if result[i][8] == 1:
        #         print('yes')
        #     else:
        #         print('no')


if __name__ == '__main__':
    main()
